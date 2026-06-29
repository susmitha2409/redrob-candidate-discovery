"""
streamlit_app.py

Redrob Intelligent Candidate Discovery — Interactive Dashboard

Run:
  streamlit run streamlit_app.py

Requirements:
  pip install streamlit plotly pandas
  (Also needs rank_candidates.py outputs — run rank_candidates.py first OR
   use the demo mode which loads sample data)
"""

import json
import os
import math
import csv
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Redrob · Candidate Discovery",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_PATH  = os.path.join(PROCESSED_DIR, "ranked_results.json")
SUBMISSION_CSV = os.path.join(BASE_DIR, "submission.csv")

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Palette ── */
  :root {
    --bg:        #0d0f14;
    --surface:   #161b27;
    --surface2:  #1e2535;
    --border:    #2a3347;
    --accent:    #6c63ff;
    --accent2:   #a78bfa;
    --green:     #22c55e;
    --yellow:    #f59e0b;
    --red:       #ef4444;
    --text:      #e2e8f0;
    --muted:     #94a3b8;
  }

  /* ── Base ── */
  .stApp { background: var(--bg); color: var(--text); }
  .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* ── Metric cards ── */
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .metric-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
                  text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
  .metric-value { font-size: 2rem; font-weight: 700; color: var(--text); line-height: 1; }
  .metric-sub   { font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }

  /* ── Candidate cards ── */
  .cand-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
  }
  .cand-card:hover { border-color: var(--accent); }
  .cand-rank {
    display: inline-block;
    width: 2.2rem; height: 2.2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 8px; text-align: center; line-height: 2.2rem;
    font-weight: 700; font-size: 0.85rem; color: white;
    margin-right: 0.75rem;
  }
  .cand-rank.top3 {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
  }
  .cand-id   { font-size: 0.72rem; color: var(--muted); font-family: monospace; }
  .cand-title { font-size: 1rem; font-weight: 600; color: var(--text); }
  .cand-meta  { font-size: 0.8rem; color: var(--muted); margin-top: 0.15rem; }
  .score-bar-wrap { background: var(--surface2); border-radius: 4px; height: 6px; flex: 1; overflow: hidden; }
  .score-bar { height: 100%; border-radius: 4px;
               background: linear-gradient(90deg, var(--accent), var(--accent2)); }

  /* ── Score pill ── */
  .pill {
    display: inline-block; border-radius: 999px;
    padding: 0.15rem 0.65rem; font-size: 0.72rem; font-weight: 600;
    margin-right: 0.35rem; margin-top: 0.35rem;
  }
  .pill-green  { background: rgba(34,197,94,.15);  color: #4ade80; border: 1px solid rgba(34,197,94,.3); }
  .pill-blue   { background: rgba(108,99,255,.15); color: #818cf8; border: 1px solid rgba(108,99,255,.3); }
  .pill-yellow { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
  .pill-red    { background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }
  .pill-gray   { background: rgba(148,163,184,.1); color: var(--muted); border: 1px solid var(--border); }

  /* ── Section header ── */
  .section-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
  }

  /* ── Search bar ── */
  .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
  }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: var(--surface2) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
  }

  /* ── Plotly transparent bg ── */
  .js-plotly-plot { background: transparent !important; }
  .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_results():
    """Load ranked results from JSON."""
    if not os.path.exists(RESULTS_PATH):
        return None, None
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data.get("results", []), data.get("jd", {})


@st.cache_data
def load_sample_candidates():
    """Load sample candidates for demo mode."""
    sample_path = os.path.join(
        BASE_DIR,
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "sample_candidates.json",
    )
    if not os.path.exists(sample_path):
        return []
    with open(sample_path) as f:
        return json.load(f)


def make_demo_results(sample_candidates):
    """Score sample candidates using real v2 scoring logic for demo mode."""
    from datetime import date, datetime
    REF = date(2026, 6, 1)
    JD_CORE = {
        "faiss","pinecone","milvus","weaviate","qdrant","elasticsearch",
        "embeddings","vector search","information retrieval","ranking",
        "recommendation","ndcg","mrr","retrieval","semantic search",
        "nlp","natural language processing","transformers","bert","llm",
        "python","pytorch","tensorflow","hugging face","sentence transformers",
        "lora","qlora","fine-tuning","rag","xgboost","mlflow","mlops",
    }
    AI_TITLES = ["machine learning","ml engineer","ai engineer","nlp engineer",
        "data scientist","search engineer","recommendation","applied scientist",
        "research engineer","applied ml","applied ai","deep learning engineer"]
    AI_DESC   = ["faiss","vector search","embedding model","retrieval system",
        "ranking model","recommendation model","fine-tuning","learning to rank",
        "information retrieval","sentence transformer"]
    AI_ASSESS = ["nlp","faiss","embeddings","fine-tuning","llm","transformers",
        "recommendation","retrieval","ranking","machine learning",
        "pytorch","tensorflow","milvus","rag","information retrieval"]
    PRODUCT_COS = {"swiggy","zomato","flipkart","meesho","cred","razorpay",
        "paytm","phonepe","ola","uber","google","microsoft","amazon",
        "freshworks","zoho","groww","sharechat","myntra","mad street den"}
    IT_SVC = {"tcs","infosys","wipro","accenture","cognizant","capgemini","hcl"}

    def _ds(date_str):
        try: return (REF - datetime.strptime(date_str, "%Y-%m-%d").date()).days
        except: return None

    def _is_ai_job(job):
        t = (job.get("title","") or "").lower()
        d = (job.get("description","") or "").lower()[:300]
        return any(kw in t for kw in AI_TITLES) or any(kw in d for kw in AI_DESC)

    results = []
    for c in sample_candidates[:100]:
        p = c.get("profile", {})
        s = c.get("redrob_signals", {})
        yoe = p.get("years_of_experience", 0) or 0
        jobs = c.get("career_history", [])
        skills_list = c.get("skills", [])
        assessments = s.get("skill_assessment_scores", {})

        # Career domain
        ai_count = sum(1 for j in jobs if _is_ai_job(j))
        prod = sum(1 for j in jobs if any(pc in (j.get("company","") or "").lower() for pc in PRODUCT_COS))
        svc  = sum(1 for j in jobs if any(sv in (j.get("company","") or "").lower() for sv in IT_SVC))
        verified_ai = sum(1 for k in assessments if any(a in k.lower() for a in AI_ASSESS))
        expert_jd   = sum(1 for sk in skills_list if sk.get("proficiency") in ("advanced","expert")
                          and any(kw in (sk.get("name","") or "").lower() for kw in JD_CORE))
        is_pivot = (verified_ai >= 2) or (expert_jd >= 3)
        if ai_count == 0:
            cd, cm = (0.30, 0.55) if is_pivot else (0.05, 0.35)
        elif ai_count == 1: cd, cm = 0.40, 0.65
        elif ai_count == 2: cd, cm = 0.70, 0.90
        else:               cd, cm = 0.90, 1.0
        if prod > 0: cd = min(cd + 0.08, 1.0)
        if svc  > 0 and prod == 0: cm = min(cm, 0.55)

        # Deep skill
        sk_scores = []
        for sk in skills_list:
            name = (sk.get("name","") or "").lower()
            if not any(kw in name for kw in JD_CORE): continue
            assess = 0.0
            for ak, av in assessments.items():
                if ak.lower() in name or name in ak.lower(): assess = av/100.0; break
            dur = min((sk.get("duration_months",0) or 0)/60.0, 1.0)
            end = min((sk.get("endorsements",0) or 0)/50.0, 1.0)
            pb  = {"expert":0.15,"advanced":0.08,"intermediate":0.0,"beginner":-0.05}.get(sk.get("proficiency","intermediate"),0)
            comp = (0.5*assess+0.3*dur+0.2*end+pb) if assess > 0 else (0.5*dur+0.35*end+0.15+pb)
            sk_scores.append(min(max(comp,0),1))
        sk_scores.sort(reverse=True)
        deep_skill = round(sum(sk_scores[:5])/5, 4) if sk_scores else 0.0

        # Behavioral
        d   = _ds(s.get("last_active_date",""))
        act = 0.3 if d is None else (1.0 if d<=14 else 0.85 if d<=30 else 0.65 if d<=60 else 0.45 if d<=90 else 0.25 if d<=180 else 0.1)
        rr  = float(s.get("recruiter_response_rate",0))
        rt  = s.get("avg_response_time_hours",120) or 120
        rt_s= 1.0 if rt<=12 else 0.85 if rt<=24 else 0.65 if rt<=48 else 0.4 if rt<=120 else 0.2
        otw = 1.0 if s.get("open_to_work_flag") else 0.3
        np_ = s.get("notice_period_days",90) or 90
        np_s= 1.0 if np_<=30 else 0.6 if np_<=60 else 0.4 if np_<=90 else 0.2
        icr = float(s.get("interview_completion_rate",0.5) or 0.5)
        gh  = s.get("github_activity_score",-1)
        gh_s= max(0,float(gh))/100.0 if gh and gh >= 0 else 0.2
        vf  = ((1 if s.get("verified_email") else 0)+(1 if s.get("verified_phone") else 0)+(1 if s.get("linkedin_connected") else 0))/3.0
        cn  = min((s.get("connection_count",0) or 0)/500.0, 1.0)
        behavioral = round(act*0.22+rr*0.16+rt_s*0.10+otw*0.10+np_s*0.12+icr*0.08+gh_s*0.12+vf*0.06+cn*0.04, 3)

        # Misc
        ef    = 1.0 if 5<=yoe<=9 else max(0.0,1.0-(5-yoe)*0.12) if yoe<5 else max(0.5,1.0-(yoe-9)*0.04)
        tmap  = {"tier_1":1.0,"tier_2":0.78,"tier_3":0.58,"tier_4":0.42}
        edu_s = max((tmap.get(e.get("tier","tier_4"),0.42) for e in c.get("education",[])), default=0.5)
        locs  = p.get("location","").lower()
        lf    = 1.0 if any(x in locs for x in ["pune","noida","delhi","gurugram","hyderabad","mumbai","bangalore","bengaluru"]) else (0.88 if p.get("country","").lower()=="india" else 0.65)
        misc  = 0.5*ef + 0.3*edu_s + 0.2*lf

        # Pseudo-semantic
        text     = " ".join([p.get("headline",""), p.get("current_title",""),
                             " ".join((j.get("description","") or "") for j in jobs)]).lower()
        rrf_norm = min(sum(1 for kw in JD_CORE if kw in text)/12.0, 1.0)

        raw   = 0.30*rrf_norm + 0.25*deep_skill + 0.20*cd + 0.15*behavioral + 0.10*misc
        final = round(raw * cm, 4)
        pivot_flag = is_pivot and ai_count == 0

        results.append({
            "id": c["candidate_id"],
            "profile": {
                "headline": p.get("headline", ""),
                "current_title": p.get("current_title", ""),
                "current_company": p.get("current_company", ""),
                "current_industry": p.get("current_industry", ""),
                "location": p.get("location", ""),
                "country": p.get("country", ""),
                "years_of_experience": yoe,
                "summary_snippet": p.get("summary", "")[:300],
            },
            "career_history": c.get("career_history", []),
            "skills": c.get("skills", []),
            "education": c.get("education", []),
            "computed": {
                "total_experience_years": yoe,
                "ai_assessment_count": len(assessments),
                "skill_assessment_scores": assessments,
            },
            "behavioral": {
                "score": behavioral,
                "open_to_work": s.get("open_to_work_flag", False),
                "last_active_date": s.get("last_active_date", ""),
                "recruiter_response_rate": s.get("recruiter_response_rate", 0),
                "notice_period_days": s.get("notice_period_days", 90),
                "github_activity_score": s.get("github_activity_score", -1),
                "avg_response_time_hours": s.get("avg_response_time_hours", None),
                "connection_count": s.get("connection_count", 0),
                "profile_completeness_score": s.get("profile_completeness_score", 0),
                "interview_completion_rate": s.get("interview_completion_rate", 0),
                "willing_to_relocate": s.get("willing_to_relocate", False),
                "preferred_work_mode": s.get("preferred_work_mode", ""),
                "verified_email": s.get("verified_email", False),
                "verified_phone": s.get("verified_phone", False),
                "linkedin_connected": s.get("linkedin_connected", False),
                "saved_by_recruiters_30d": s.get("saved_by_recruiters_30d", 0),
                "expected_salary_lpa": s.get("expected_salary_range_inr_lpa", {}),
            },
            "final_score": final,
            "score_breakdown": {
                "rrf_norm": round(rrf_norm, 4),
                "deep_skill": deep_skill,
                "skill_match": deep_skill,       # alias for dashboard compatibility
                "career_domain": round(cd, 4),
                "career_mult": cm,
                "behavioral": behavioral,
                "experience_fit": round(ef, 4),
                "edu_score": round(edu_s, 4),
                "location_fit": round(lf, 4),
                "vector_rank": 999,
                "bm25_rank": 999,
            },
            "reasoning": (
                f"{p.get('current_title','?')} · {yoe:.1f}yrs · "
                f"career domain {cd:.0%}(×{cm}) · deep skill {deep_skill:.0%} · "
                + ("PIVOT candidate · " if pivot_flag else "")
                + f"behavioral {behavioral:.0%}"
            ),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results



def score_color(v):
    if v >= 0.75: return "pill-green"
    if v >= 0.50: return "pill-blue"
    if v >= 0.30: return "pill-yellow"
    return "pill-red"


def days_since(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date(2026, 6, 1) - d).days
    except:
        return None


def activity_pill(last_active):
    days = days_since(last_active)
    if days is None: return '<span class="pill pill-gray">unknown</span>'
    if days <= 14:   return f'<span class="pill pill-green">active {days}d ago</span>'
    if days <= 30:   return f'<span class="pill pill-green">active {days}d ago</span>'
    if days <= 60:   return f'<span class="pill pill-yellow">active {days}d ago</span>'
    if days <= 90:   return f'<span class="pill pill-yellow">dormant {days}d</span>'
    return f'<span class="pill pill-red">inactive {days}d</span>'


def render_candidate_card(c, rank, expanded=False):
    p   = c.get("profile", {})
    b   = c.get("behavioral", {})
    sb  = c.get("score_breakdown", {})

    rank_cls = "top3" if rank <= 3 else ""
    score_pct = int(c["final_score"] * 100)
    bar_w = min(100, score_pct)

    title_safe = p.get("current_title", "Unknown").replace("<", "&lt;")
    company = p.get("current_company", "")
    location = f"{p.get('location','')}, {p.get('country','')}".strip(", ")
    yoe = p.get("years_of_experience", 0)
    headline = p.get("headline", "")[:80]

    # pills
    pills = []
    if b.get("open_to_work"):
        pills.append('<span class="pill pill-green">Open to Work</span>')
    notice = b.get("notice_period_days", 90)
    if notice <= 30:
        pills.append(f'<span class="pill pill-green">Notice {notice}d</span>')
    elif notice <= 60:
        pills.append(f'<span class="pill pill-yellow">Notice {notice}d</span>')
    else:
        pills.append(f'<span class="pill pill-red">Notice {notice}d</span>')
    pills.append(activity_pill(b.get("last_active_date", "")))
    work_mode = b.get("preferred_work_mode", "")
    if work_mode:
        pills.append(f'<span class="pill pill-gray">{work_mode}</span>')
    if b.get("willing_to_relocate"):
        pills.append('<span class="pill pill-blue">Relocatable</span>')

    github = b.get("github_activity_score", -1)
    if github > 0:
        gc = "pill-green" if github >= 50 else "pill-yellow" if github >= 20 else "pill-gray"
        pills.append(f'<span class="pill {gc}">GitHub {github:.0f}</span>')

    html = f"""
    <div class="cand-card">
      <div style="display:flex;align-items:flex-start;gap:0.75rem;">
        <span class="cand-rank {rank_cls}">#{rank}</span>
        <div style="flex:1;min-width:0;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
            <div>
              <div class="cand-title">{title_safe}
                <span class="cand-id" style="margin-left:0.5rem;">{c['id']}</span>
              </div>
              <div class="cand-meta">
                {f'<b>{company}</b> &nbsp;·&nbsp;' if company else ''}{yoe:.1f} yrs experience &nbsp;·&nbsp; {location}
              </div>
              {'<div class="cand-meta" style="margin-top:0.2rem;color:#94a3b8;">'+headline+'</div>' if headline else ''}
            </div>
            <div style="text-align:right;white-space:nowrap;">
              <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;">{score_pct}</div>
              <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;">score</div>
            </div>
          </div>

          <!-- score bar -->
          <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.75rem;">
            <div class="score-bar-wrap">
              <div class="score-bar" style="width:{bar_w}%;"></div>
            </div>
            <div style="font-size:0.72rem;color:#94a3b8;white-space:nowrap;min-width:90px;">
              Skill {sb.get('deep_skill', sb.get('skill_match',0)):.0%} &nbsp; Career {sb.get('career_domain',0):.0%}
            </div>
          </div>

          <!-- pills -->
          <div style="margin-top:0.6rem;">{''.join(pills)}</div>
        </div>
      </div>
    </div>
    """
    return html


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(results, jd, demo_mode=False):
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1.25rem;">
          <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0;">🎯 Redrob</div>
          <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;">Candidate Discovery</div>
        </div>
        """, unsafe_allow_html=True)

        # Score range banner
        if results:
            max_s = max(c["final_score"] for c in results)
            min_s = min(c["final_score"] for c in results)
            st.markdown(
                f'<div style="font-size:11px;color:#94a3b8;background:#1e2535;'
                f'border-radius:6px;padding:8px 10px;margin-bottom:10px;">'
                f'📊 Score range: '
                f'<b style="color:#e2e8f0;">{min_s:.3f}</b> → '
                f'<b style="color:#22c55e;">{max_s:.3f}</b>'
                + (f'<br><span style="color:#f59e0b;">⚠ Demo: {len(results)} sample candidates</span>' if demo_mode else '')
                + '</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Filter & Sort**")

        search = st.text_input("Search by title, company, skill",
                               placeholder="e.g. Recommendation, FAISS, Backend")

        # Cap slider max to actual max score so user can see all results
        _max_s = max((c["final_score"] for c in results), default=1.0)
        min_score = st.slider("Minimum score", 0.0, round(_max_s, 2), 0.0, 0.01)

        open_only = st.checkbox("Open to work only", value=False)

        relocate_only = st.checkbox("Willing to relocate", value=False)

        work_modes = ["All"] + sorted(set(
            c.get("behavioral", {}).get("preferred_work_mode", "") for c in results
            if c.get("behavioral", {}).get("preferred_work_mode", "")
        ))
        work_mode_filter = st.selectbox("Work mode", work_modes)

        notice_max = st.slider("Max notice period (days)", 0, 180, 180, step=10)

        yoe_range = st.slider(
            "Years of experience",
            0.0, 20.0, (0.0, 20.0), step=0.5
        )

        st.markdown("---")
        st.markdown("**Sort by**")
        sort_by = st.radio("", ["Final Score", "Deep Skill", "Behavioral", "Experience"], index=0)

        st.markdown("---")
        if st.button("🔄 Reload Results", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        return {
            "search": search,
            "min_score": min_score,
            "open_only": open_only,
            "relocate_only": relocate_only,
            "work_mode": work_mode_filter,
            "notice_max": notice_max,
            "yoe_range": yoe_range,
            "sort_by": sort_by,
        }


# ─── Filter & Sort ─────────────────────────────────────────────────────────────

def apply_filters(results, filters):
    out = results[:]

    if filters["search"]:
        q = filters["search"].lower()
        def _matches(c):
            p = c.get("profile", {})
            # Check profile fields
            if q in (p.get("current_title", "") or "").lower(): return True
            if q in (p.get("current_company", "") or "").lower(): return True
            if q in (p.get("headline", "") or "").lower(): return True
            if q in (p.get("current_industry", "") or "").lower(): return True
            # Check skills
            if any(q in (sk.get("name", "") or "").lower() for sk in c.get("skills", [])): return True
            # Check past job titles
            if any(q in (j.get("title", "") or "").lower() for j in c.get("career_history", [])): return True
            return False
        out = [c for c in out if _matches(c)]

    if filters["min_score"] > 0:
        out = [c for c in out if c["final_score"] >= filters["min_score"]]

    if filters["open_only"]:
        out = [c for c in out if c.get("behavioral", {}).get("open_to_work", False)]

    if filters["relocate_only"]:
        out = [c for c in out if c.get("behavioral", {}).get("willing_to_relocate", False)]

    if filters["work_mode"] != "All":
        out = [c for c in out if c.get("behavioral", {}).get("preferred_work_mode", "") == filters["work_mode"]]

    out = [c for c in out if c.get("behavioral", {}).get("notice_period_days", 0) <= filters["notice_max"]]

    lo, hi = filters["yoe_range"]
    out = [c for c in out if lo <= (c.get("profile", {}).get("years_of_experience", 0) or 0) <= hi]

    sort_key_map = {
        "Final Score": lambda c: c["final_score"],
        "Deep Skill":  lambda c: c.get("score_breakdown", {}).get("deep_skill", c.get("score_breakdown", {}).get("skill_match", 0)),
        "Behavioral":  lambda c: c.get("behavioral", {}).get("score", 0),
        "Experience":  lambda c: c.get("profile", {}).get("years_of_experience", 0),
    }
    key_fn = sort_key_map.get(filters["sort_by"], lambda c: c["final_score"])
    out.sort(key=key_fn, reverse=True)

    return out


# ─── Charts ──────────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,39,0.8)",
    font=dict(color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)


def chart_score_dist(results):
    scores = [c["final_score"] for c in results]
    fig = px.histogram(
        x=scores, nbins=30,
        labels={"x": "Final Score", "y": "Count"},
        title="Score Distribution",
        color_discrete_sequence=["#6c63ff"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**CHART_LAYOUT)
    return fig


def chart_skill_vs_behavioral(results):
    df = pd.DataFrame([{
        "deep_skill":  c.get("score_breakdown", {}).get("deep_skill",
                       c.get("score_breakdown", {}).get("skill_match", 0)),
        "behavioral":  c.get("behavioral", {}).get("score", 0),
        "final_score": c["final_score"],
        "title":       c.get("profile", {}).get("current_title", ""),
        "id":          c["id"],
    } for c in results])
    fig = px.scatter(
        df, x="deep_skill", y="behavioral",
        color="final_score", size="final_score",
        color_continuous_scale="Purp",
        hover_data=["id", "title"],
        labels={"deep_skill": "Deep Skill Score", "behavioral": "Behavioral Score"},
        title="Deep Skill vs Behavioral",
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


def chart_experience_dist(results):
    yoe = [c.get("profile", {}).get("years_of_experience", 0) or 0 for c in results]
    fig = px.histogram(
        x=yoe, nbins=20,
        labels={"x": "Years of Experience", "y": "Count"},
        title="Experience Distribution",
        color_discrete_sequence=["#a78bfa"],
    )
    fig.add_vline(x=5, line_dash="dash", line_color="#f59e0b",
                  annotation_text="JD min 5yrs", annotation_position="top right")
    fig.add_vline(x=9, line_dash="dash", line_color="#f59e0b",
                  annotation_text="JD max 9yrs", annotation_position="top left")
    fig.update_layout(**CHART_LAYOUT)
    return fig


def chart_location(results):
    countries = pd.Series([
        c.get("profile", {}).get("country", "Unknown") for c in results
    ]).value_counts().reset_index()
    countries.columns = ["country", "count"]
    fig = px.bar(
        countries.head(10), x="count", y="country", orientation="h",
        color="count", color_continuous_scale="Purp",
        title="Top Countries",
        labels={"count": "Candidates", "country": ""},
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
    return fig


def chart_top_skills(results):
    from collections import Counter
    skill_counter = Counter()
    for c in results:
        for sk in c.get("skills", []):
            name = sk.get("name", "")
            if name:
                skill_counter[name] += 1
    top = pd.DataFrame(skill_counter.most_common(15), columns=["skill", "count"])
    fig = px.bar(
        top, x="count", y="skill", orientation="h",
        color="count", color_continuous_scale="Purp",
        title="Most Common Skills (Top 100)",
        labels={"count": "Candidates", "skill": ""},
    )
    fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
    return fig


def chart_notice_period(results):
    notice = [c.get("behavioral", {}).get("notice_period_days", 90) for c in results]
    fig = px.histogram(
        x=notice, nbins=18,
        labels={"x": "Notice Period (days)", "y": "Count"},
        title="Notice Period Distribution",
        color_discrete_sequence=["#22c55e"],
    )
    fig.add_vline(x=30, line_dash="dash", line_color="#f59e0b",
                  annotation_text="JD buyout limit", annotation_position="top right")
    fig.update_layout(**CHART_LAYOUT)
    return fig


# ─── Candidate Detail Panel ───────────────────────────────────────────────────

def render_candidate_detail(c):
    p  = c.get("profile", {})
    b  = c.get("behavioral", {})
    sb = c.get("score_breakdown", {})
    comp = c.get("computed", {})

    st.markdown(f"### {p.get('current_title', 'Candidate')} — {c['id']}")
    st.markdown(f"*{p.get('headline', '')}*")
    st.markdown(f"**{p.get('current_company', '')}** &nbsp;·&nbsp; {p.get('location', '')}, {p.get('country', '')} &nbsp;·&nbsp; {p.get('years_of_experience', 0):.1f} yrs experience")

    if p.get("summary_snippet"):
        st.markdown(f"> {p['summary_snippet']}...")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Score",    f"{c['final_score']:.1%}")
    c2.metric("Deep Skill",     f"{sb.get('deep_skill', sb.get('skill_match', 0)):.1%}")
    c3.metric("Behavioral",     f"{b.get('score', 0):.1%}")
    c4.metric("Experience Fit", f"{sb.get('experience_fit', 0):.1%}")

    st.markdown('<div class="section-header">Behavioral Signals</div>', unsafe_allow_html=True)
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Response Rate",      f"{b.get('recruiter_response_rate', 0):.0%}")
    bc2.metric("Notice Period",       f"{b.get('notice_period_days', 90)} days")
    bc3.metric("Interview Completion",f"{b.get('interview_completion_rate', 0):.0%}")

    bd1, bd2, bd3 = st.columns(3)
    bd1.metric("GitHub Score", f"{b.get('github_activity_score', -1)}" if b.get('github_activity_score', -1) >= 0 else "Not linked")
    bd2.metric("Saved by Recruiters", f"{b.get('saved_by_recruiters_30d', 0)}")
    bd3.metric("Profile Completeness",f"{b.get('profile_completeness_score', 0):.0f}%")

    # Radar chart for score breakdown
    categories = ["Semantic", "Deep Skill", "Career\nDomain", "Behavioral", "Exp Fit"]
    values = [
        sb.get("rrf_norm", 0),
        sb.get("deep_skill", sb.get("skill_match", 0)),
        sb.get("career_domain", sb.get("skill_match", 0)),
        sb.get("behavioral", 0),
        sb.get("experience_fit", 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(108,99,255,0.2)",
        line=dict(color="#6c63ff", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(22,27,39,0.8)",
            radialaxis=dict(visible=True, range=[0, 1], color="#94a3b8", gridcolor="#2a3347"),
            angularaxis=dict(color="#94a3b8", gridcolor="#2a3347"),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Career history
    if c.get("career_history"):
        st.markdown('<div class="section-header">Career History</div>', unsafe_allow_html=True)
        for job in c["career_history"][:4]:
            end = job.get("end_date") or "Present"
            dur = job.get("duration_months", 0)
            st.markdown(f"**{job.get('title','')}** at {job.get('company','')}  \n"
                        f"*{job.get('start_date','')[:7]} → {str(end)[:7]}* &nbsp;({dur} months) &nbsp;|&nbsp; {job.get('industry','')}")
            desc = job.get("description", "")
            if desc:
                st.caption(desc[:300] + ("..." if len(desc) > 300 else ""))

    # Skills
    if c.get("skills"):
        st.markdown('<div class="section-header">Skills</div>', unsafe_allow_html=True)
        skill_html = ""
        for sk in sorted(c["skills"], key=lambda x: ["beginner","intermediate","advanced","expert"].index(x.get("proficiency","beginner")), reverse=True)[:20]:
            level = sk.get("proficiency", "beginner")
            cls = {"expert": "pill-green", "advanced": "pill-blue", "intermediate": "pill-yellow"}.get(level, "pill-gray")
            skill_html += f'<span class="pill {cls}">{sk["name"]}</span>'
        st.markdown(skill_html, unsafe_allow_html=True)

    # Education
    if c.get("education"):
        st.markdown('<div class="section-header">Education</div>', unsafe_allow_html=True)
        for edu in c["education"]:
            tier = edu.get("tier", "")
            tier_badge = {"tier_1": "🥇", "tier_2": "🥈", "tier_3": "🥉"}.get(tier, "")
            st.markdown(f"{tier_badge} **{edu.get('degree','')}** in {edu.get('field_of_study','')}  \n"
                        f"{edu.get('institution','')} &nbsp;({edu.get('start_year','')}–{edu.get('end_year','')})")

    # Skill assessments
    assessments = comp.get("skill_assessment_scores", {})
    if assessments:
        st.markdown('<div class="section-header">Platform Assessment Scores</div>', unsafe_allow_html=True)
        df_a = pd.DataFrame([(k, v) for k, v in assessments.items()], columns=["Skill", "Score"])
        fig_a = px.bar(df_a, x="Score", y="Skill", orientation="h",
                       color="Score", color_continuous_scale="Purp", range_x=[0, 100])
        fig_a.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=max(150, len(assessments)*40))
        st.plotly_chart(fig_a, use_container_width=True)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    # Load data
    results, jd = load_results()
    demo_mode = results is None

    if demo_mode:
        st.warning("⚠️ **Demo mode** — Run `python rank_candidates.py` first to load real results. Showing sample data.")
        sample = load_sample_candidates()
        if not sample:
            st.error("No data found. Please run `python ingest_candidates.py` then `python rank_candidates.py`.")
            return
        results = make_demo_results(sample)
        jd = {"must_have_keywords": ["embeddings", "FAISS", "Python", "retrieval", "ranking"],
              "preferred_titles": ["ML Engineer", "AI Engineer"],
              "min_years_experience": 5, "max_years_experience": 9}

    # Sidebar
    filters = render_sidebar(results, jd, demo_mode=demo_mode)
    filtered = apply_filters(results, filters)

    # ── Header ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
      <h1 style="font-size:1.8rem;font-weight:700;margin:0;color:#e2e8f0;">
        🎯 Intelligent Candidate Discovery
      </h1>
      <p style="color:#94a3b8;margin:0.25rem 0 0;font-size:0.9rem;">
        Senior AI Engineer &nbsp;·&nbsp; Redrob AI &nbsp;·&nbsp;
        {"<span style='color:#f59e0b;'>Demo mode</span>" if demo_mode else "<span style='color:#22c55e;'>Live results</span>"}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────
    total = len(filtered)
    open_count = sum(1 for c in filtered if c.get("behavioral", {}).get("open_to_work"))
    avg_score = sum(c["final_score"] for c in filtered) / max(len(filtered), 1)
    sub30_notice = sum(1 for c in filtered if c.get("behavioral", {}).get("notice_period_days", 999) <= 30)

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Candidates Shown</div>
        <div class="metric-value">{total}</div>
        <div class="metric-sub">of {len(results)} total ranked</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Final Score</div>
        <div class="metric-value">{avg_score:.2f}</div>
        <div class="metric-sub">higher = better fit</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Open to Work</div>
        <div class="metric-value" style="color:#22c55e;">{open_count}</div>
        <div class="metric-sub">actively looking</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Short Notice ≤30d</div>
        <div class="metric-value" style="color:#a78bfa;">{sub30_notice}</div>
        <div class="metric-sub">immediate availability</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────
    tab_list, tab_charts, tab_jd, tab_download = st.tabs(
        ["📋 Rankings", "📊 Analytics", "📄 Job Description", "⬇️ Download"]
    )

    # ── Tab: Rankings ─────────────────────────────────────────────
    with tab_list:
        if not filtered:
            st.markdown("""
            <div style="background:#1e2535;border:1px solid #2a3347;border-radius:12px;
                        padding:2rem;text-align:center;margin-top:1rem;">
              <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
              <div style="font-size:1rem;font-weight:500;color:#e2e8f0;margin-bottom:0.5rem;">
                No candidates match your filters
              </div>
              <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
                Try these fixes:<br>
                • <b>Clear the search box</b><br>
                • Set Minimum score to <b>0.00</b><br>
                • Set Max notice period to <b>180</b><br>
                • Uncheck Open to work &amp; Willing to relocate<br>
                • Set Years of experience to <b>0 – 20</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            selected_id = st.session_state.get("selected_id")

            for i, c in enumerate(filtered[:100]):
                rank = i + 1
                p   = c.get("profile", {})
                b   = c.get("behavioral", {})
                sb  = c.get("score_breakdown", {})
                score_pct = int(c["final_score"] * 100)
                bar_w = min(100, score_pct)
                rank_cls = "top3" if rank <= 3 else ""
                title_safe = p.get("current_title", "Unknown").replace("<", "&lt;")
                company = p.get("current_company", "")
                location = f"{p.get('location','')}, {p.get('country','')}".strip(", ")
                yoe = p.get("years_of_experience", 0)
                headline = (p.get("headline", "") or "")[:80]
                deep_skill_pct = int(sb.get("deep_skill", sb.get("skill_match", 0)) * 100)
                career_pct = int(sb.get("career_domain", 0) * 100)

                pills = []
                if b.get("open_to_work"):
                    pills.append('<span class="pill pill-green">Open to Work</span>')
                notice = b.get("notice_period_days", 90)
                nc = "pill-green" if notice <= 30 else "pill-yellow" if notice <= 60 else "pill-red"
                pills.append(f'<span class="pill {nc}">Notice {notice}d</span>')
                pills.append(activity_pill(b.get("last_active_date", "")))
                wm = b.get("preferred_work_mode", "")
                if wm:
                    pills.append(f'<span class="pill pill-gray">{wm}</span>')
                if b.get("willing_to_relocate"):
                    pills.append('<span class="pill pill-blue">Relocatable</span>')
                gh = b.get("github_activity_score", -1)
                if gh and gh > 0:
                    gc = "pill-green" if gh >= 50 else "pill-yellow" if gh >= 20 else "pill-gray"
                    pills.append(f'<span class="pill {gc}">GitHub {gh:.0f}</span>')
                pills_html = "".join(pills)
                headline_html = f'<div class="cand-meta" style="margin-top:0.2rem;color:#94a3b8;">{headline}</div>' if headline else ""
                company_html = f"<b>{company}</b> &nbsp;&middot;&nbsp; " if company else ""

                st.markdown(f"""
                <div class="cand-card">
                  <div style="display:flex;align-items:flex-start;gap:0.75rem;">
                    <span class="cand-rank {rank_cls}">#{rank}</span>
                    <div style="flex:1;min-width:0;">
                      <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
                        <div>
                          <div class="cand-title">{title_safe}
                            <span class="cand-id" style="margin-left:0.5rem;">{c['id']}</span>
                          </div>
                          <div class="cand-meta">{company_html}{yoe:.1f} yrs &nbsp;&middot;&nbsp; {location}</div>
                          {headline_html}
                        </div>
                        <div style="text-align:right;white-space:nowrap;">
                          <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;">{score_pct}</div>
                          <div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;">score</div>
                        </div>
                      </div>
                      <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.75rem;">
                        <div class="score-bar-wrap"><div class="score-bar" style="width:{bar_w}%;"></div></div>
                        <div style="font-size:0.72rem;color:#94a3b8;white-space:nowrap;min-width:90px;">
                          Skill {deep_skill_pct}% &nbsp; Career {career_pct}%
                        </div>
                      </div>
                      <div style="margin-top:0.6rem;">{pills_html}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"▸ Details — {c['id']}", expanded=(selected_id == c['id'])):
                    render_candidate_detail(c)

    # ── Tab: Analytics ────────────────────────────────────────────
    with tab_charts:
        if not filtered:
            st.info("No candidates match your current filters — relax them to see analytics.")
        else:
            st.markdown('<div class="section-header">Score Analysis</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.plotly_chart(chart_score_dist(filtered), use_container_width=True)
            with cc2:
                st.plotly_chart(chart_skill_vs_behavioral(filtered), use_container_width=True)

            st.markdown('<div class="section-header">Candidate Pool Breakdown</div>', unsafe_allow_html=True)
            cc3, cc4 = st.columns(2)
            with cc3:
                st.plotly_chart(chart_experience_dist(filtered), use_container_width=True)
            with cc4:
                st.plotly_chart(chart_notice_period(filtered), use_container_width=True)

            st.markdown('<div class="section-header">Skills & Geography</div>', unsafe_allow_html=True)
            cc5, cc6 = st.columns(2)
            with cc5:
                st.plotly_chart(chart_top_skills(filtered), use_container_width=True)
            with cc6:
                st.plotly_chart(chart_location(filtered), use_container_width=True)

            # Score breakdown heatmap for top 20
            st.markdown('<div class="section-header">Top 20 Score Breakdown</div>', unsafe_allow_html=True)
            top20 = filtered[:20]
            hm_data = pd.DataFrame([{
            "ID": c["id"],
            "Semantic": c.get("score_breakdown", {}).get("rrf_norm", 0),
            "Deep Skill": c.get("score_breakdown", {}).get("deep_skill", c.get("score_breakdown", {}).get("skill_match", 0)),
            "Career Dom.": c.get("score_breakdown", {}).get("career_domain", 0),
            "Behav.": c.get("behavioral", {}).get("score", 0),
            "Exp Fit": c.get("score_breakdown", {}).get("experience_fit", 0),
            } for c in top20]).set_index("ID")

            fig_hm = px.imshow(
            hm_data,
            color_continuous_scale="Purp",
            zmin=0, zmax=1,
            aspect="auto",
            title="Score Component Heatmap (Top 20)",
            )
            fig_hm.update_layout(**CHART_LAYOUT, height=500)
            st.plotly_chart(fig_hm, use_container_width=True)

    # ── Tab: JD ───────────────────────────────────────────────────
    with tab_jd:
        st.markdown("### Senior AI Engineer — Redrob AI")
        st.markdown("*Parsed requirements used for ranking:*")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Must-Have Keywords**")
            must_html = "".join(f'<span class="pill pill-green">{k}</span>' for k in jd.get("must_have_keywords", []))
            st.markdown(must_html or "*none parsed*", unsafe_allow_html=True)

            st.markdown("**Preferred Titles**")
            title_html = "".join(f'<span class="pill pill-blue">{t}</span>' for t in jd.get("preferred_titles", []))
            st.markdown(title_html or "*none parsed*", unsafe_allow_html=True)

        with col_b:
            st.markdown("**Nice-to-Have Keywords**")
            nice_html = "".join(f'<span class="pill pill-yellow">{k}</span>' for k in jd.get("nice_to_have_keywords", []))
            st.markdown(nice_html or "*none parsed*", unsafe_allow_html=True)

            st.markdown("**Disqualifier Signals**")
            dq_html = "".join(f'<span class="pill pill-red">{d}</span>' for d in jd.get("disqualifier_signals", []))
            st.markdown(dq_html or "*none parsed*", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Search Query (used for semantic search)**")
        st.info(jd.get("search_query", "No query generated."))

    # ── Tab: Download ─────────────────────────────────────────────
    with tab_download:
        st.markdown("### Export Submission")

        # Build CSV in memory
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, c in enumerate(results[:100], 1):
            writer.writerow([c["id"], rank, f"{c['final_score']:.4f}", c.get("reasoning", "")])
        csv_str = buf.getvalue()

        st.download_button(
            label="⬇️  Download submission.csv (top 100)",
            data=csv_str,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Export Filtered List")
        buf2 = io.StringIO()
        writer2 = csv.writer(buf2)
        writer2.writerow(["candidate_id", "rank", "final_score", "current_title",
                           "years_exp", "location", "behavioral_score",
                           "response_rate", "notice_days", "open_to_work", "reasoning"])
        for rank, c in enumerate(filtered[:100], 1):
            p = c.get("profile", {})
            b = c.get("behavioral", {})
            writer2.writerow([
                c["id"], rank, f"{c['final_score']:.4f}",
                p.get("current_title", ""), f"{p.get('years_of_experience',0):.1f}",
                f"{p.get('location','')}, {p.get('country','')}",
                f"{b.get('score',0):.3f}",
                f"{b.get('recruiter_response_rate',0):.2f}",
                b.get("notice_period_days", 90),
                b.get("open_to_work", False),
                c.get("reasoning", ""),
            ])
        csv2_str = buf2.getvalue()

        st.download_button(
            label="⬇️  Download filtered results (current view)",
            data=csv2_str,
            file_name="filtered_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("**Validation command:**")
        st.code("python validate_submission.py submission.csv", language="bash")


if __name__ == "__main__":
    main()
