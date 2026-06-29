"""
ingest_candidates.py

Adapted from ingest_enhanced.py for the Redrob Intelligent Candidate Discovery Challenge.

Processes 100K candidate profiles from candidates.jsonl:
1. Builds rich search_text from profile, career history, skills, education
2. Computes behavioral_score from redrob_signals (availability, responsiveness, activity)
3. Generates FAISS vector index + BM25 index
4. Saves processed candidates + indices for use by rank_candidates.py

Run: python ingest_candidates.py
"""

import json
import pickle
import os
import math
from datetime import datetime, date
from tqdm import tqdm

# Search Libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
INDICES_DIR = os.path.join(BASE_DIR, "data", "indices")

CANDIDATES_JSONL = os.path.join(DATA_DIR, "candidates.jsonl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Recency reference date (close to dataset creation)
REFERENCE_DATE = date(2026, 6, 1)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(INDICES_DIR, exist_ok=True)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def clean(val, fallback=""):
    """Safe string clean."""
    if val is None:
        return fallback
    s = str(val).strip()
    return s if s not in ("", "None", "null", "nan") else fallback


def days_since(date_str):
    """Days between date_str (YYYY-MM-DD) and REFERENCE_DATE. Returns None on error."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return None


# ─── Search Text Builder ────────────────────────────────────────────────────────

def build_search_text(c):
    """
    Build rich search text from all semantically meaningful fields.
    Stored UPPERCASE (matches BM25 tokenization convention from original codebase).
    """
    parts = []

    # Profile
    p = c.get("profile", {})
    parts += [
        clean(p.get("headline")),
        clean(p.get("summary")),
        clean(p.get("current_title")),
        clean(p.get("current_company")),
        clean(p.get("current_industry")),
        clean(p.get("location")),
        clean(p.get("country")),
    ]

    # Career history
    for job in c.get("career_history", []):
        parts += [
            clean(job.get("title")),
            clean(job.get("company")),
            clean(job.get("industry")),
            clean(job.get("description")),
        ]

    # Skills
    for skill in c.get("skills", []):
        parts.append(clean(skill.get("name")))
        # Weight advanced/expert skills by repeating them
        prof = clean(skill.get("proficiency", ""))
        if prof in ("advanced", "expert"):
            parts.append(clean(skill.get("name")))  # duplicate for BM25 weight

    # Education
    for edu in c.get("education", []):
        parts += [
            clean(edu.get("degree")),
            clean(edu.get("field_of_study")),
            clean(edu.get("institution")),
        ]

    # Certifications
    for cert in c.get("certifications", []):
        parts += [clean(cert.get("name")), clean(cert.get("issuer"))]

    # Skill assessment scores (key names are skill names)
    signals = c.get("redrob_signals", {})
    for skill_name in signals.get("skill_assessment_scores", {}).keys():
        parts.append(clean(skill_name))

    # Join and uppercase
    text = " ".join(x for x in parts if x)
    return text.upper()


# ─── Behavioral Score ────────────────────────────────────────────────────────

def compute_behavioral_score(c):
    """
    Compute a 0–1 behavioral score from redrob_signals.
    
    Reflects: availability, responsiveness, platform activity, reliability.
    Weights tuned to the JD (urgent hire, sub-30d notice preferred, active candidates).
    """
    s = c.get("redrob_signals", {})
    score_components = []

    # 1. Availability (is actively looking?)
    open_to_work = 1.0 if s.get("open_to_work_flag", False) else 0.3
    score_components.append(("open_to_work", open_to_work, 0.15))

    # 2. Recency of last activity (penalize inactive candidates heavily)
    days_inactive = days_since(s.get("last_active_date", ""))
    if days_inactive is None:
        activity_score = 0.3
    elif days_inactive <= 14:
        activity_score = 1.0
    elif days_inactive <= 30:
        activity_score = 0.85
    elif days_inactive <= 60:
        activity_score = 0.65
    elif days_inactive <= 90:
        activity_score = 0.45
    elif days_inactive <= 180:
        activity_score = 0.25
    else:
        activity_score = 0.1
    score_components.append(("last_active", activity_score, 0.20))

    # 3. Recruiter response rate (key signal per JD hint)
    response_rate = s.get("recruiter_response_rate", 0)
    score_components.append(("response_rate", float(response_rate), 0.15))

    # 4. Notice period (JD prefers sub-30 days)
    notice = s.get("notice_period_days", 90)
    if notice <= 0:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.6
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2
    score_components.append(("notice_period", notice_score, 0.10))

    # 5. Profile completeness
    completeness = s.get("profile_completeness_score", 50) / 100.0
    score_components.append(("profile_completeness", completeness, 0.10))

    # 6. Interview completion rate (reliability signal)
    icr = s.get("interview_completion_rate", 0.5)
    score_components.append(("interview_completion", float(icr), 0.10))

    # 7. GitHub activity (relevant for AI Engineer role)
    github = s.get("github_activity_score", -1)
    if github == -1:
        github_score = 0.3  # no github linked — slightly negative for this role
    else:
        github_score = float(github) / 100.0
    score_components.append(("github", github_score, 0.10))

    # 8. Verifications (trust signals)
    verified = (
        (1 if s.get("verified_email", False) else 0) +
        (1 if s.get("verified_phone", False) else 0) +
        (1 if s.get("linkedin_connected", False) else 0)
    ) / 3.0
    score_components.append(("verified", verified, 0.05))

    # 9. Saved by recruiters (market validation)
    saved = min(s.get("saved_by_recruiters_30d", 0) / 10.0, 1.0)
    score_components.append(("saved_by_recruiters", saved, 0.05))

    # Weighted sum
    total_weight = sum(w for _, _, w in score_components)
    weighted_score = sum(v * w for _, v, w in score_components) / total_weight

    return round(weighted_score, 4), {name: round(val, 3) for name, val, _ in score_components}


# ─── Candidate Flattening ─────────────────────────────────────────────────────

def flatten_candidate(c):
    """
    Flatten a raw candidate JSON into the processed format used by the search engine.
    Mirrors the structure of advisors_enhanced.json from the original codebase.
    """
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})

    behavioral_score, behavioral_breakdown = compute_behavioral_score(c)

    # Compute total experience months from career history
    total_months = sum(
        job.get("duration_months", 0)
        for job in c.get("career_history", [])
    )

    # Count AI-core skills (skills with assessment scores = platform-verified)
    ai_assessment_count = len(s.get("skill_assessment_scores", {}))

    # Skill proficiency summary
    skill_levels = {}
    for skill in c.get("skills", []):
        name = clean(skill.get("name"))
        if name:
            skill_levels[name] = clean(skill.get("proficiency", "beginner"))

    return {
        "id": c["candidate_id"],

        # Profile snapshot (for result display)
        "profile": {
            "headline": clean(p.get("headline")),
            "current_title": clean(p.get("current_title")),
            "current_company": clean(p.get("current_company")),
            "current_industry": clean(p.get("current_industry")),
            "location": clean(p.get("location")),
            "country": clean(p.get("country")),
            "years_of_experience": p.get("years_of_experience", 0),
            "summary_snippet": clean(p.get("summary", ""))[:300],
        },

        # Career history (raw, for display)
        "career_history": c.get("career_history", []),

        # Skills
        "skills": c.get("skills", []),
        "skill_levels": skill_levels,

        # Education
        "education": c.get("education", []),

        # Certifications
        "certifications": c.get("certifications", []),

        # Computed fields for filtering/ranking
        "computed": {
            "total_experience_months": total_months,
            "total_experience_years": round(total_months / 12, 1),
            "ai_assessment_count": ai_assessment_count,
            "skill_assessment_scores": s.get("skill_assessment_scores", {}),
        },

        # Behavioral signals (pre-scored)
        "behavioral": {
            "score": behavioral_score,
            "breakdown": behavioral_breakdown,
            "open_to_work": s.get("open_to_work_flag", False),
            "last_active_date": s.get("last_active_date", ""),
            "recruiter_response_rate": s.get("recruiter_response_rate", 0),
            "notice_period_days": s.get("notice_period_days", 90),
            "github_activity_score": s.get("github_activity_score", -1),
            "profile_completeness_score": s.get("profile_completeness_score", 0),
            "interview_completion_rate": s.get("interview_completion_rate", 0),
            "offer_acceptance_rate": s.get("offer_acceptance_rate", -1),
            "verified_email": s.get("verified_email", False),
            "verified_phone": s.get("verified_phone", False),
            "linkedin_connected": s.get("linkedin_connected", False),
            "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d", 0),
            "expected_salary_lpa": s.get("expected_salary_range_inr_lpa", {}),
            "preferred_work_mode": s.get("preferred_work_mode", ""),
            "willing_to_relocate": s.get("willing_to_relocate", False),
            # v2 signals — needed by compute_behavioral_score_v2()
            "avg_response_time_hours": s.get("avg_response_time_hours", None),
            "connection_count": s.get("connection_count", 0),
        },

        # For BM25 + vector search
        "search_text": "",  # Filled below
    }


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(limit=None):
    """
    Full ingestion pipeline.
    
    Args:
        limit: If set, only process first N candidates (for testing).
    """
    print("\n" + "=" * 60)
    print("REDROB CANDIDATE INGESTION")
    print("=" * 60)

    # ── Step 1: Stream + flatten candidates ──────────────────────
    print(f"\n[1/4] Loading candidates from {CANDIDATES_JSONL}...")
    candidates = []
    
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Reading JSONL", total=limit or 100_000)):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                flat = flatten_candidate(raw)
                flat["search_text"] = build_search_text(raw)
                candidates.append(flat)
            except Exception as e:
                print(f"   ⚠️  Skipping line {i}: {e}")
                continue

    print(f"   ✓ Processed {len(candidates)} candidates")

    # ── Step 2: Generate embeddings ──────────────────────────────
    print(f"\n[2/4] Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("   Generating embeddings (MiniLM: ~25-35 mins for 100K on CPU)...")
    search_texts = [c["search_text"] for c in candidates]
    # all-MiniLM-L6-v2 does not need the "passage:" prefix unlike e5 models
    keyed_texts = search_texts

    batch_size = 512  # MiniLM is lightweight — larger batches are fine on CPU
    embeddings = model.encode(
        keyed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )

    print(f"   ✓ Embeddings shape: {embeddings.shape}")

    # ── Step 3: Build FAISS index ─────────────────────────────────
    print("\n[3/4] Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity (pre-normalized)
    index.add(embeddings)
    print(f"   ✓ FAISS index: {index.ntotal} vectors, dim={dimension}")

    # ── Step 4: Build BM25 index ──────────────────────────────────
    print("\n[4/4] Building BM25 index...")
    tokenized_corpus = [text.split() for text in search_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"   ✓ BM25 index built for {len(tokenized_corpus)} documents")

    # ── Save artifacts ────────────────────────────────────────────
    print("\n💾 Saving artifacts...")

    candidates_path = os.path.join(PROCESSED_DIR, "candidates_processed.json")
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f)
    print(f"   ✓ {candidates_path} ({len(candidates)} candidates)")

    faiss_path = os.path.join(INDICES_DIR, "candidates.faiss")
    faiss.write_index(index, faiss_path)
    print(f"   ✓ {faiss_path}")

    bm25_path = os.path.join(INDICES_DIR, "candidates_bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"   ✓ {bm25_path}")

    # Summary stats
    behavioral_scores = [c["behavioral"]["score"] for c in candidates]
    print(f"\n{'='*60}")
    print("✅ INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"   Total candidates:     {len(candidates):,}")
    print(f"   Avg behavioral score: {sum(behavioral_scores)/len(behavioral_scores):.3f}")
    print(f"   Open to work:         {sum(1 for c in candidates if c['behavioral']['open_to_work']):,}")
    print(f"\n💡 Next: Run rank_candidates.py to generate submission")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        print(f"⚠️  Test mode: processing first {limit} candidates only")
    run_ingestion(limit=limit)
