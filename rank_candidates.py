"""
rank_candidates.py  —  v2 (upgraded)

Key fixes over v1:
  1. Career domain gate  — multiplier kills non-AI careers regardless of skill list
  2. Deep skill scorer   — uses duration_months + endorsements + assessment score
  3. Response time       — avg_response_time_hours folded into behavioral
  4. Education tier      — tier_1/2/3/4 credibility signal
  5. Product company prestige — career at Swiggy/Zomato vs TCS/Infosys
  6. Rebalanced weights  — deep_skill 25%, career_domain 20%, semantic 30%

Run:
  python rank_candidates.py                   → submission.csv (top 100)
  python rank_candidates.py --show 20         → print top 20 to terminal
  python rank_candidates.py --jd-only         → show parsed JD and exit
"""

import json, pickle, os, csv, argparse
from datetime import date, datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
INDICES_DIR   = os.path.join(BASE_DIR, "data", "indices")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
RRF_K           = 60
RETRIEVAL_TOP_K = 500
FINAL_TOP_N     = 100
REFERENCE_DATE  = date(2026, 6, 1)

# ── JD ─────────────────────────────────────────────────────────────────────────
JD_TEXT = """
Job: Senior AI Engineer — Founding Team at Redrob AI (Series A)
Location: Pune/Noida, India (Hybrid).
Experience: 5–9 years preferred (4–5 in applied ML/AI at product companies).

MUST HAVE:
- Production embeddings-based retrieval (sentence-transformers, E5, BGE, OpenAI)
- Production vector databases / hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch, OpenSearch)
- Strong Python; code quality matters
- Evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing)
- Shipped end-to-end ranking, search, or recommendation system at a product company

NICE TO HAVE:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost, neural LTR)
- HR-tech, recruiting tech, marketplace products
- Distributed systems / large-scale inference

DISQUALIFIERS:
- Pure research without production deployment
- AI experience only from LLM/LangChain tutorials with no pre-LLM ML background
- Career entirely at IT services (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- No production code in last 18 months
- Primary expertise computer vision / speech / robotics without NLP/IR background
"""

# ── JD-relevant skills (for deep skill scoring) ────────────────────────────────
JD_CORE_SKILLS = {
    # Must-have — NLP / IR / ranking focused (JD explicitly wants this, not CV/speech)
    "faiss", "pinecone", "weaviate", "qdrant", "elasticsearch", "opensearch",
    "milvus", "chroma", "vespa",                          # vector DBs
    "sentence transformers", "sentence-transformers", "embeddings", "vector search",
    "hybrid search", "information retrieval", "ranking", "recommendation",
    "ndcg", "mrr", "map", "learning to rank", "retrieval", "semantic search",
    "nlp", "natural language processing", "transformers", "bert", "llm",
    "python", "pytorch", "tensorflow", "hugging face", "huggingface",
    "reranking", "cross-encoder", "bi-encoder", "dense retrieval",
    # Nice-to-have
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "rag",
    "xgboost", "lightgbm", "mlflow", "kubeflow", "mlops",
    "kafka", "spark", "airflow", "kubernetes",
    # REMOVED intentionally — JD disqualifies pure CV/speech background:
    # "opencv", "gans", "object detection", "image classification",
    # "speech recognition", "tts", "computer vision"
}

# ── Career domain keywords ──────────────────────────────────────────────────────
AI_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "nlp engineer",
    "data scientist", "search engineer", "recommendation", "applied scientist",
    "research engineer", "applied ml", "applied ai", "deep learning engineer",
    "mlops", "ml researcher", "nlp researcher",
]

AI_DESC_KEYWORDS = [
    "faiss", "vector search", "semantic search", "embedding model",
    "retrieval system", "ranking model", "recommendation model",
    "fine-tuning", "neural ranker", "learning to rank",
    "information retrieval", "sentence transformer",
]

IT_SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "persistent systems", "coforge",
}

PRODUCT_COMPANIES_INDIA = {
    "swiggy", "zomato", "flipkart", "meesho", "cred", "razorpay", "paytm",
    "phonepe", "ola", "uber", "google", "microsoft", "amazon", "meta",
    "apple", "netflix", "salesforce", "adobe", "oracle", "atlassian",
    "freshworks", "zoho", "cleartax", "groww", "zerodha", "upstox",
    "byju", "unacademy", "sharechat", "moj", "dream11", "myntra",
    "nykaa", "policybazaar", "lenskart", "urban company", "dunzo",
    "mad street den", "sarvam", "krutrim", "sarvam ai",
}

# ── Helpers ─────────────────────────────────────────────────────────────────────
def days_since(date_str):
    try:
        return (REFERENCE_DATE - datetime.strptime(date_str, "%Y-%m-%d").date()).days
    except:
        return None

def clean(v):
    s = str(v).strip() if v else ""
    return s if s not in ("None", "null", "nan") else ""


# ── LLM JD parser ───────────────────────────────────────────────────────────────
def parse_jd(jd_text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_jd()
    client = Groq(api_key=api_key)
    prompt = """Parse this job description and return ONLY valid JSON (no markdown):
{
  "must_have_keywords": ["15-25 critical technical terms"],
  "nice_to_have_keywords": ["10-15 secondary terms"],
  "preferred_titles": ["realistic job titles a good candidate would have"],
  "disqualifier_signals": ["3-5 things that disqualify a candidate"],
  "search_query": "150-200 word natural language description of the ideal candidate",
  "min_years_experience": 5,
  "max_years_experience": 9
}
Include both acronyms and full forms. Focus on production skills, not just knowledge."""
    try:
        r = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"system","content":prompt},
                      {"role":"user","content":f"Parse:\n\n{jd_text}"}],
            temperature=0.1,
        )
        raw = r.choices[0].message.content.replace("```json","").replace("```","").strip()
        result = json.loads(raw)
        result.setdefault("must_have_keywords", [])
        result.setdefault("nice_to_have_keywords", [])
        result.setdefault("preferred_titles", [])
        result.setdefault("disqualifier_signals", [])
        result.setdefault("search_query", jd_text[:400])
        result.setdefault("min_years_experience", 5)
        result.setdefault("max_years_experience", 9)
        return result
    except Exception as e:
        print(f"   ⚠ LLM error: {e} — using fallback")
        return _fallback_jd()

def _fallback_jd():
    return {
        "must_have_keywords": [
            "embeddings","FAISS","vector search","semantic search","hybrid search",
            "sentence-transformers","information retrieval","ranking","NDCG","MRR",
            "recommendation system","Python","NLP","natural language processing",
            "Elasticsearch","Pinecone","RAG","transformer","BERT","retrieval",
        ],
        "nice_to_have_keywords": [
            "LoRA","QLoRA","PEFT","fine-tuning","learning to rank","XGBoost",
            "A/B testing","MLflow","Spark","distributed systems",
        ],
        "preferred_titles": [
            "AI Engineer","ML Engineer","Machine Learning Engineer","Search Engineer",
            "NLP Engineer","Applied Scientist","Data Scientist","Research Engineer",
            "Recommendation Systems Engineer",
        ],
        "disqualifier_signals": ["TCS","Infosys","Wipro","Accenture","Cognizant","Capgemini"],
        "search_query": (
            "Senior AI Engineer with 5-9 years production experience in embeddings, "
            "vector search, hybrid retrieval systems. Built ranking or recommendation "
            "systems at product companies. Strong Python. FAISS, Elasticsearch, "
            "sentence-transformers. Evaluation frameworks NDCG, A/B testing. "
            "NLP and information retrieval background required."
        ),
        "min_years_experience": 5,
        "max_years_experience": 9,
    }


# ── Scoring functions ────────────────────────────────────────────────────────────

def compute_deep_skill_score(candidate):
    """
    NEW v2 signal: weighted combination of
      - platform-verified assessment score (if taken)
      - duration_months of use
      - peer endorsements
    Only for JD-relevant skills.
    """
    skills = candidate.get("skills", [])
    assessments = candidate.get("computed", {}).get("skill_assessment_scores", {})
    
    skill_scores = []
    for sk in skills:
        name = sk.get("name", "").lower()
        if not any(kw in name for kw in JD_CORE_SKILLS):
            continue
        
        # Assessment component (0–1)
        assess_score = 0.0
        for ak, av in assessments.items():
            if ak.lower() in name or name in ak.lower():
                assess_score = float(av) / 100.0
                break
        
        # Duration component: 60 months (5 years) = full score
        duration = sk.get("duration_months", 0) or 0
        duration_score = min(duration / 60.0, 1.0)
        
        # Endorsements component: 50+ endorsements = full score
        endorsements = sk.get("endorsements", 0) or 0
        endorse_score = min(endorsements / 50.0, 1.0)
        
        # Proficiency bonus
        prof_bonus = {"expert": 0.15, "advanced": 0.08, "intermediate": 0.0, "beginner": -0.05}.get(
            sk.get("proficiency", "intermediate"), 0.0
        )
        
        # Composite: if assessment taken it dominates, else duration+endorsements
        if assess_score > 0:
            composite = 0.5 * assess_score + 0.3 * duration_score + 0.2 * endorse_score + prof_bonus
        else:
            composite = 0.5 * duration_score + 0.35 * endorse_score + 0.15 + prof_bonus
        
        skill_scores.append(min(max(composite, 0.0), 1.0))
    
    if not skill_scores:
        return 0.0
    
    # Top-5 average (don't penalise for having unrelated skills)
    skill_scores.sort(reverse=True)
    return round(sum(skill_scores[:5]) / 5, 4)


def compute_career_domain_score(candidate):
    """
    NEW v2: How AI/ML is the actual career history?
    Returns (score 0-1, multiplier 0.25-1.0)
    """
    jobs = candidate.get("career_history", [])
    if not jobs:
        return 0.3, 0.25
    
    ai_job_count = 0
    product_company_count = 0
    services_count = 0
    total_ai_months = 0
    
    for job in jobs:
        title = (job.get("title", "") or "").lower()
        company = (job.get("company", "") or "").lower()
        desc = (job.get("description", "") or "").lower()
        duration = job.get("duration_months", 0) or 0
        
        # Does the title/description indicate AI/ML work?
        is_ai_job = (any(kw in title for kw in AI_TITLE_KEYWORDS) or
                     any(kw in desc[:300] for kw in AI_DESC_KEYWORDS))
        if is_ai_job:
            ai_job_count += 1
            total_ai_months += duration
        
        # Product company check
        if any(pc in company for pc in PRODUCT_COMPANIES_INDIA):
            product_company_count += 1
        
        # IT services check
        if any(s in company for s in IT_SERVICES_FIRMS):
            services_count += 1
    
    # Score based on depth of AI career
    total_months = sum(j.get("duration_months", 0) or 0 for j in jobs)
    ai_fraction = total_ai_months / max(total_months, 1)

    # ── Career pivot detection ──────────────────────────────────────────────
    # Some candidates do real AI work under non-AI job titles (Backend Eng, Data Eng).
    # If they have 2+ platform-verified AI assessments OR 3+ expert JD-core skills,
    # treat them as a "verified career pivot" — don't punish them like a Civil Engineer.
    skills = candidate.get("skills", [])
    assessments = candidate.get("computed", {}).get("skill_assessment_scores", {})

    AI_ASSESS_KEYS = [
        "nlp", "faiss", "embeddings", "fine-tuning", "llm", "transformers",
        "bert", "recommendation", "retrieval", "ranking", "deep learning",
        "machine learning", "pytorch", "tensorflow", "milvus", "rag",
        "information retrieval", "semantic search", "vector", "sentence",
    ]
    verified_ai_assessments = sum(
        1 for k in assessments
        if any(a in k.lower() for a in AI_ASSESS_KEYS)
    )
    expert_jd_skills = sum(
        1 for sk in skills
        if sk.get("proficiency") in ("advanced", "expert")
        and any(kw in (sk.get("name", "") or "").lower() for kw in JD_CORE_SKILLS)
    )
    # Require at least 1 platform-verified assessment — prevents a Project Manager
    # who self-reports "expert Embeddings" with zero assessments from gaming the pivot tier.
    is_career_pivot = (
        (verified_ai_assessments >= 2) or
        (expert_jd_skills >= 3 and verified_ai_assessments >= 1)
    )

    # Base score
    if ai_job_count == 0:
        if is_career_pivot:
            # Doing AI work under a non-AI title — treated as 1 weak AI role
            base = 0.30
            multiplier = 0.55
        else:
            base = 0.05
            multiplier = 0.35      # raised from 0.25 → gives deep_skill more room
    elif ai_job_count == 1:
        base = 0.40
        multiplier = 0.65
    elif ai_job_count == 2:
        base = 0.70
        multiplier = 0.90
    else:
        base = 0.90
        multiplier = 1.0
    
    # Boost for product company experience
    if product_company_count > 0:
        base = min(base + 0.08, 1.0)
    
    # Penalty for services-only
    if services_count > 0 and product_company_count == 0:
        multiplier = min(multiplier, 0.55)
    
    return round(base, 4), multiplier


def compute_behavioral_score_v2(candidate):
    """
    v2: Adds avg_response_time_hours, connection_count.
    """
    b = candidate.get("behavioral", {})
    
    # 1. Open to work
    open_score = 1.0 if b.get("open_to_work") else 0.3
    
    # 2. Activity recency
    days = days_since(b.get("last_active_date", ""))
    if days is None:       act = 0.3
    elif days <= 14:       act = 1.0
    elif days <= 30:       act = 0.85
    elif days <= 60:       act = 0.65
    elif days <= 90:       act = 0.45
    elif days <= 180:      act = 0.25
    else:                  act = 0.1
    
    # 3. Response rate
    rr = float(b.get("recruiter_response_rate", 0))
    
    # 4. Response time (NEW)
    rt_hours = b.get("avg_response_time_hours", 120)
    if rt_hours is None:       rt = 0.5
    elif rt_hours <= 12:       rt = 1.0
    elif rt_hours <= 24:       rt = 0.85
    elif rt_hours <= 48:       rt = 0.65
    elif rt_hours <= 120:      rt = 0.4
    else:                      rt = 0.2
    
    # 5. Notice period
    notice = b.get("notice_period_days", 90) or 90
    if notice <= 0:        np_score = 1.0
    elif notice <= 30:     np_score = 1.0
    elif notice <= 60:     np_score = 0.6
    elif notice <= 90:     np_score = 0.4
    else:                  np_score = 0.2
    
    # 6. Interview completion
    icr = float(b.get("interview_completion_rate", 0.5) or 0.5)
    
    # 7. GitHub
    gh = b.get("github_activity_score", -1)
    github = max(0, float(gh)) / 100.0 if gh and gh >= 0 else 0.2
    
    # 8. Verifications
    verified = (
        (1 if b.get("verified_email") else 0) +
        (1 if b.get("verified_phone") else 0) +
        (1 if b.get("linkedin_connected") else 0)
    ) / 3.0
    
    # 9. Connection count (NEW — proxy for professional network)
    conns = b.get("connection_count", 0) or 0
    connection = min(conns / 500.0, 1.0)
    
    weights = [
        (act,        0.22),   # activity recency — most important availability signal
        (rr,         0.16),   # response rate
        (rt,         0.10),   # response time
        (open_score, 0.10),   # open to work flag
        (np_score,   0.12),   # notice period — JD cares about this
        (icr,        0.08),   # interview completion (reliability)
        (github,     0.12),   # github — meaningful for AI engineer role
        (verified,   0.06),   # trust signals
        (connection, 0.04),   # network size
    ]
    score = sum(v * w for v, w in weights)
    return round(score, 4)


def compute_education_score(candidate):
    """NEW v2: education tier + institution prestige."""
    edu_list = candidate.get("education", [])
    if not edu_list:
        return 0.5
    tier_scores = {"tier_1": 1.0, "tier_2": 0.78, "tier_3": 0.58, "tier_4": 0.42}
    best = max(tier_scores.get(e.get("tier", "tier_4"), 0.42) for e in edu_list)
    return round(best, 4)


def compute_experience_fit(candidate, jd):
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    lo, hi = jd.get("min_years_experience", 5), jd.get("max_years_experience", 9)
    if lo <= yoe <= hi:  return 1.0
    if yoe < lo:         return max(0.0, 1.0 - (lo - yoe) * 0.12)
    return max(0.5, 1.0 - (yoe - hi) * 0.04)


def compute_location_fit(candidate):
    country = (candidate.get("profile", {}).get("country", "") or "").lower()
    location = (candidate.get("profile", {}).get("location", "") or "").lower()
    if country != "india":
        return 0.65
    preferred = ["pune", "noida", "delhi", "gurugram", "gurgaon",
                 "hyderabad", "mumbai", "bangalore", "bengaluru"]
    return 1.0 if any(c in location for c in preferred) else 0.88


def build_reasoning(candidate, scores, jd):
    p = candidate.get("profile", {})
    b = candidate.get("behavioral", {})
    title = p.get("current_title", "Unknown")
    yoe   = p.get("years_of_experience", 0)
    
    # JD skill hits
    search_text = candidate.get("search_text", "").lower()
    must_hits = sum(1 for kw in jd.get("must_have_keywords", []) if kw.lower() in search_text)
    
    parts = [f"{title} with {yoe:.1f} yrs exp"]
    parts.append(f"{must_hits}/{len(jd.get('must_have_keywords',[]))} core skills")
    parts.append(f"career domain {scores['career_domain']:.0%}")
    parts.append(f"deep skill {scores['deep_skill']:.0%}")
    if b.get("open_to_work"):  parts.append("open to work")
    notice = b.get("notice_period_days", 90)
    if notice <= 30:  parts.append(f"notice {notice}d")
    gh = b.get("github_activity_score", -1)
    if gh and gh > 30:  parts.append(f"github {gh:.0f}")
    rt = b.get("avg_response_time_hours", 999)
    if rt and rt < 24:  parts.append(f"responds in {rt:.0f}h")
    return "; ".join(parts) + "."


# ── Main ranker ──────────────────────────────────────────────────────────────────
class CandidateRanker:
    def __init__(self):
        print("⚙  Loading candidates...")
        with open(os.path.join(PROCESSED_DIR, "candidates_processed.json")) as f:
            self.candidates = json.load(f)
        print(f"   ✓ {len(self.candidates):,} candidates")
        
        self.faiss_index = faiss.read_index(os.path.join(INDICES_DIR, "candidates.faiss"))
        print(f"   ✓ FAISS index ({self.faiss_index.ntotal:,} vectors)")
        
        with open(os.path.join(INDICES_DIR, "candidates_bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        print("   ✓ BM25 index")
        
        print(f"   Loading {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Ranker ready\n")

    def _vector_search(self, query, k):
        emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.faiss_index.search(emb, k)
        return [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0])) if I[0][i] >= 0]

    def _bm25_search(self, query, k):
        scores = self.bm25.get_scores(query.upper().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in ranked[:k]]

    def _rrf(self, vec, bm25):
        rrf = {}
        for rank, (doc_id, score) in enumerate(vec, 1):
            rrf.setdefault(doc_id, {"rrf":0.0,"vs":0.0,"bs":0.0,"vr":9999,"br":9999})
            rrf[doc_id]["rrf"] += 1.0 / (RRF_K + rank)
            rrf[doc_id]["vs"] = score;  rrf[doc_id]["vr"] = rank
        for rank, (doc_id, score) in enumerate(bm25, 1):
            rrf.setdefault(doc_id, {"rrf":0.0,"vs":0.0,"bs":0.0,"vr":9999,"br":9999})
            rrf[doc_id]["rrf"] += 1.0 / (RRF_K + rank)
            rrf[doc_id]["bs"] = score;  rrf[doc_id]["br"] = rank
        return sorted(rrf.items(), key=lambda x: x[1]["rrf"], reverse=True)

    def rank(self, jd, top_n=FINAL_TOP_N):
        # Build query
        must = " ".join(jd.get("must_have_keywords", []))
        nice = " ".join(jd.get("nice_to_have_keywords", []))
        titles = " ".join(jd.get("preferred_titles", []))
        query = f"{jd['search_query']} {must} {nice} {titles}"
        
        print(f"   Running vector search (top {RETRIEVAL_TOP_K})...")
        vec = self._vector_search(query, RETRIEVAL_TOP_K)
        print(f"   Running BM25 search (top {RETRIEVAL_TOP_K})...")
        bm25 = self._bm25_search(query, RETRIEVAL_TOP_K)
        print("   Fusing with RRF...")
        fused = self._rrf(vec, bm25)
        
        pool = min(3000, len(fused))
        print(f"   Scoring top {pool} candidates with v2 signals...")
        
        scored = []
        for doc_idx, rrf_sc in fused[:pool]:
            c = self.candidates[doc_idx]
            
            # --- All five signal components ---
            rrf_norm        = min(rrf_sc["rrf"] * 500, 1.0)
            deep_skill      = compute_deep_skill_score(c)
            career_domain, career_mult = compute_career_domain_score(c)
            behavioral      = compute_behavioral_score_v2(c)
            exp_fit         = compute_experience_fit(c, jd)
            edu_score       = compute_education_score(c)
            location        = compute_location_fit(c)
            
            # Combined experience/edu/location
            misc = 0.5 * exp_fit + 0.3 * edu_score + 0.2 * location
            
            # Weighted sum
            # 30% semantic · 25% deep skill · 20% career domain · 15% behavioral · 10% misc
            raw = (
                0.30 * rrf_norm    +
                0.25 * deep_skill  +
                0.20 * career_domain +
                0.15 * behavioral  +
                0.10 * misc
            )
            
            # Apply career domain gate multiplier (the anti-gaming mechanism)
            final = raw * career_mult
            
            scores = {
                "rrf_norm": round(rrf_norm, 4),
                "deep_skill": deep_skill,
                "career_domain": career_domain,
                "career_mult": career_mult,
                "behavioral": behavioral,
                "exp_fit": exp_fit,
                "edu_score": edu_score,
                "location": location,
                "vector_rank": rrf_sc["vr"],
                "bm25_rank": rrf_sc["br"],
            }
            
            scored.append({
                **c,
                "final_score": round(final, 4),
                "score_breakdown": scores,
                "reasoning": "",  # filled below
            })
        
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        
        for c in scored:
            c["reasoning"] = build_reasoning(c, c["score_breakdown"], jd)
        
        return scored[:top_n]


def write_submission(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, c in enumerate(results, 1):
            w.writerow([c["id"], rank, f"{c['final_score']:.4f}", c["reasoning"]])
    print(f"   ✓ {path}")


def display(results, n=10):
    print(f"\n{'='*70}\nTOP {n} CANDIDATES\n{'='*70}")
    for i, c in enumerate(results[:n], 1):
        p = c.get("profile", {})
        sb = c.get("score_breakdown", {})
        b  = c.get("behavioral", {})
        print(f"\n#{i:>3}  {c['id']}  —  Final: {c['final_score']:.4f}")
        print(f"     {p.get('current_title','')} @ {p.get('current_company','')} | {p.get('years_of_experience',0):.1f}yrs | {p.get('location','')}")
        print(f"     Semantic {sb.get('rrf_norm',0):.0%}  Deep-skill {sb.get('deep_skill',0):.0%}  Career {sb.get('career_domain',0):.0%} (×{sb.get('career_mult',1):.2f})  Behav {sb.get('behavioral',0):.0%}")
        print(f"     ↳ {c.get('reasoning','')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",     type=int, default=FINAL_TOP_N)
    parser.add_argument("--show",    type=int, default=0)
    parser.add_argument("--output",  default="submission.csv")
    parser.add_argument("--jd-only", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("REDROB CANDIDATE RANKING ENGINE  —  v2")
    print("="*60)

    print("\n[1/3] Parsing JD...")
    jd = parse_jd(JD_TEXT)
    print(f"   ✓ {len(jd['must_have_keywords'])} must-have · {len(jd['nice_to_have_keywords'])} nice-to-have")

    if args.jd_only:
        print(json.dumps(jd, indent=2));  return

    print("\n[2/3] Loading indices...")
    ranker = CandidateRanker()

    print(f"[3/3] Ranking (top {args.top})...")
    results = ranker.rank(jd, top_n=args.top)
    print(f"\n✅ Done — {len(results)} candidates ranked")

    out = os.path.join(BASE_DIR, args.output)
    write_submission(results, out)

    # Save full JSON for Streamlit
    slim = [{k:v for k,v in c.items() if k != "search_text"} for c in results]
    full_path = os.path.join(PROCESSED_DIR, "ranked_results.json")
    with open(full_path, "w") as f:
        json.dump({"jd": jd, "results": slim,
                   "generated_at": datetime.now().isoformat()}, f)
    print(f"   ✓ {full_path}  (for Streamlit)")

    if args.show > 0:
        display(results, args.show)

    print(f"\n  Submission → {out}")
    print(f"  Validate  → python validate_submission.py {out}\n")


if __name__ == "__main__":
    main()
