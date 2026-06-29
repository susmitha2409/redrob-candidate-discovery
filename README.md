# Redrob Intelligent Candidate Discovery
## Track 01 — Data & AI Challenge

A hybrid AI ranking system that finds the top 100 most relevant candidates from 100,000 profiles for a Senior AI Engineer role.

---

## Project Structure

```
hackathon/
├── ingest_candidates.py       ← Step 1: Build search indices (run once)
├── rank_candidates.py         ← Step 2: Score and rank candidates
├── streamlit_app.py           ← Step 3: Interactive dashboard
├── requirements.txt           ← All Python dependencies
├── .env                       ← Your API key goes here (you create this)
└── data/                      ← Auto-created when you run ingest
    ├── processed/
    │   ├── candidates_processed.json
    │   ├── ontology_enhanced.json
    │   └── ranked_results.json
    └── indices/
        ├── candidates.faiss
        └── candidates_bm25.pkl
```

---

## Prerequisites

- Python 3.9 or higher
- 8 GB RAM minimum (16 GB recommended for full 100K dataset)
- Free Groq API key from https://console.groq.com
- The dataset folder: `[PUB] India_runs_data_and_ai_challenge/` in the same directory

---

## Step-by-Step Instructions

### Step 1 — Install dependencies

Open your terminal, navigate to this folder, and run:

```bash
pip install -r requirements.txt
```

This installs everything: sentence-transformers, FAISS, BM25, Groq, Streamlit, Plotly, etc.

> **Note:** This will download the `intfloat/e5-large-v2` model (~1.3 GB) on first run. Make sure you have internet.

---

### Step 2 — Set up your Groq API key

1. Go to https://console.groq.com and sign up (free)
2. Click **API Keys** → **Create API Key** → copy it
3. Create a file called `.env` in this folder with this content:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

> The key is only used once during ranking to parse the job description. If you skip this, the system uses a built-in fallback — it still works, just slightly less accurate.

---

### Step 3 — Run ingestion (build the search indices)

```bash
python ingest_candidates.py
```

This reads `candidates.jsonl`, builds FAISS + BM25 indices, and pre-computes behavioral scores.

**Expected time:**
- 500 candidates (test mode): ~2 minutes
- Full 100,000 candidates: ~25–35 minutes

**To test with 500 candidates first (recommended):**
```bash
python ingest_candidates.py 500
```

When done you will see:
```
✅ INGESTION COMPLETE
   Total candidates: 100,000
   Open to work: 31,204
```

---

### Step 4 — Run the ranker (generate submission.csv)

```bash
python rank_candidates.py
```

This parses the job description with the LLM, runs hybrid search, scores all candidates, and outputs `submission.csv`.

**Expected time:** 2–5 minutes

**To preview the top 20 results in the terminal:**
```bash
python rank_candidates.py --show 20
```

When done you will see:
```
✅ Done — 100 candidates ranked
   ✓ submission.csv
   ✓ data/processed/ranked_results.json (for Streamlit)
```

---

### Step 5 — Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

Then open your browser to: **http://localhost:8501**

The dashboard opens automatically. It shows:
- 📋 **Rankings tab** — all 100 candidates with score bars, pills, expandable details
- 📊 **Analytics tab** — 6 charts (score distribution, skill vs behavioral, heatmap, etc.)
- 📄 **Job Description tab** — parsed JD requirements as colored keyword pills
- ⬇️ **Download tab** — export submission.csv and filtered lists

> **Demo mode:** If you haven't run `rank_candidates.py` yet, the dashboard still works — it loads `sample_candidates.json` and runs a live preview of the scoring.

---

### Step 6 — Validate your submission

```bash
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## Full Run Order (copy-paste)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Create .env with your Groq key (do this manually — see Step 2)

# 3. Quick test with 500 candidates
python ingest_candidates.py 500
python rank_candidates.py --show 10

# 4. Full run
python ingest_candidates.py
python rank_candidates.py

# 5. Dashboard
streamlit run streamlit_app.py

# 6. Validate
python "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```

---

## How the Scoring Works

```
Final Score = (
    30% × Semantic match      (FAISS vector + BM25 keyword, fused via RRF)
  + 25% × Deep skill score    (assessment score + duration_months + endorsements)
  + 20% × Career domain       (AI/ML job titles in career history)
  + 15% × Behavioral signals  (activity, response rate, notice period, GitHub)
  + 10% × Misc                (experience fit + education tier + location)
) × career_domain_multiplier  (0.35x–1.0x penalty based on AI career depth)
```

**Career pivot detection:** Candidates with no AI job title but 2+ platform-verified AI assessments get a 0.55x multiplier instead of 0.35x — recognising real AI work done under non-AI titles.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'faiss'` | Run `pip install faiss-cpu` |
| `FileNotFoundError: candidates.jsonl` | Make sure the dataset folder is in the same directory as these files |
| `FileNotFoundError: candidates_processed.json` | Run `python ingest_candidates.py` first |
| `GROQ_API_KEY not set` | Create `.env` file with your key (Step 2). System uses fallback if missing. |
| Streamlit shows "Demo mode" | Run `python rank_candidates.py` first to generate `ranked_results.json` |
| Out of memory during ingest | Run `python ingest_candidates.py 10000` to test with 10K candidates |

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Semantic search | `intfloat/e5-large-v2` + FAISS | Meaning-based candidate retrieval |
| Keyword search | BM25Okapi (rank-bm25) | Exact term matching |
| Fusion | Reciprocal Rank Fusion (RRF) | Combine both search results |
| JD parsing | Groq API (Llama 3.3 70B) | Extract structured requirements from JD |
| Dashboard | Streamlit + Plotly | Interactive recruiter UI |
