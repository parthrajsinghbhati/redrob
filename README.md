# Redrob AI — Intelligent Candidate Discovery & Ranking

An AI-powered candidate ranking system that goes beyond keyword matching to understand who genuinely fits a role. Built for the Redrob Hackathon (India Runs Data & AI Challenge).

## Architecture

```
100K Candidates (JSONL)
    │
    ├── Stage 1:   Hard Filters          (structural disqualifiers)
    ├── Stage 1.5: Honeypot Detection    (concrete profile anomalies)
    ├── Stage 2:   Semantic Scoring      (hybrid: dense BGE cosine + lexical BM25)
    ├── Stage 3:   Rule-Based Scoring    (parallel — title, skills, career, education, location)
    ├── Stage 4:   Behavioral Modifier   (platform engagement multiplier, 0.5×–1.2×)
    ├── Stage 5:   Score Combination     (base = 0.375 semantic + 0.625 rule, × behavioral)
    ├── Stage 5.5: Cross-Encoder Re-rank (bge-reranker-base over the top-200 shortlist)
    └── Stage 6:   Top-100 + Reasoning   (rank-aware templates, real profile data)
```

## Key Design Decisions

1. **Hybrid retrieval (dense + BM25), dense half precomputed offline.** Candidate profiles and the JD are encoded with `BAAI/bge-base-en-v1.5` (768-dim) once, offline (GPU-friendly, allowed to exceed the 5-min window). The ranking step only *loads* the vectors and does a CPU dot-product — so it needs no GPU and no network — catching "plain-language" strong candidates who describe their work naturally without buzzwords. At rank time this dense score is blended with a lexical **BM25-Okapi** score (`rank-bm25`, pure-CPU) at 70% dense / 30% BM25, so exact-term matches (specific tools, frameworks) still surface alongside semantic matches.

2. **Cross-encoder re-ranking of a shortlist.** A bi-encoder is fast but coarse; a cross-encoder (`BAAI/bge-reranker-base`) is precise but expensive. We get both by re-ranking only the top-200 shortlist — precision where it matters (NDCG@10) while staying inside the CPU/5-min budget. It degrades gracefully (dense+rule order) if the weights aren't available locally.

3. **Anti-keyword-stuffing, evidence-gated.** Skills are scored with a *trust multiplier* (proficiency, endorsements, duration, assessment scores — not just presence). On top of that, a title-agnostic stuffer check separates *claims* from *evidence*: a profile that advertises AI (a packed skills list or a buzzword headline like "AI enthusiast | Building with LLMs") but shows no genuine ML work in its actual job titles or role descriptions is penalised. Crucially, the self-written summary counts as a claim, not evidence — so domain pivots (PMs, marketers, analysts) who buzzword-stuff are caught, while real practitioners (whose titles/role descriptions always carry ML signal) are never touched. ML/AI experience credit likewise requires real ML terms, so generic words like "production" or "pipeline" don't leak credit from manufacturing/ops roles.

4. **Career trajectory analysis.** Detects consulting-only careers, claim-vs-evidence mismatches, and job-hopping patterns. Genuine product/ML role experience is weighted heavily — including partial credit for real ML work done at consulting firms.

5. **Honeypot detection.** Flags only profiles with verifiable, concrete impossibilities (experience exceeding the career span, a role longer than the whole career, "expert" skills with zero usage, future dates) — high precision, so it never discards genuinely excellent candidates. Keyword-stuffer traps are handled separately as score penalties.

6. **Behavioral signals as a true multiplier.** The semantic and rule scores form a weighted base in `[0, 1]`; the behavioral composite then *scales* that base (floor 0.5×, ceiling 1.2×) rather than adding a fixed amount. Active, responsive candidates get a genuine boost; inactive ones a real discount — with no artificial additive floor. The composite reads 13 signals (recruiter response, recency, open-to-work, interview completion, notice period, GitHub, verification, saved-by-recruiters, offer-acceptance rate, application activity, and network strength).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Submission

### Step 1: Pre-compute embeddings (offline, once)

Runs on CPU (~15-30 min) or any GPU (a free Google Colab T4 does it in ~2-8 min):

```bash
python precompute_embeddings.py \
    --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
    --out-dir ./data
# On a GPU, add: --fp16 --batch-size 384
```

This writes `data/embeddings.npz` and `data/jd_embedding.npy`. (These artifacts are
not committed because of their size; regenerate them with the command above.)

The cross-encoder weights (`BAAI/bge-reranker-base`, ~1.1 GB) download automatically
on the first `rank.py` run and are then cached locally for offline re-runs.

### Step 2: Run the ranking pipeline (CPU, no network, < 5 min)

```bash
python rank.py \
    --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \
    --out ./submission.csv
# Add --no-rerank to skip the cross-encoder stage.
```

### Step 3: Validate the output

```bash
python India_runs_data_and_ai_challenge/validate_submission.py ./submission.csv
```

## Project Structure

```
redrob/
├── rank.py                     # Main pipeline (entry point)
├── precompute_embeddings.py    # Offline dense-embedding generation + loader
├── config.py                   # All weights, thresholds, skill lists, model names
├── utils.py                    # Streaming, date parsing, safe accessors, text builder
├── honeypot_detector.py        # Concrete-anomaly honeypot detection
├── candidate_analyzer.py       # Rule-based per-candidate scoring
├── scoring.py                  # Behavioral multiplier + score combination
├── reasoning.py                # Reasoning text generation (rank-aware templates)
├── requirements.txt            # Python dependencies
├── submission_metadata.yaml    # Portal metadata mirror
├── data/
│   ├── embeddings.npz          # Pre-computed candidate embeddings (regenerate)
│   └── jd_embedding.npy        # Pre-computed JD embedding (regenerate)
└── India_runs_data_and_ai_challenge/
    ├── candidates.jsonl         # 100K candidate pool
    └── ...                      # Other challenge files
```

## Compute Environment

- **Platform**: macOS (Apple Silicon)
- **RAM**: 16 GB
- **Python**: 3.11+
- **GPU**: Used only for *offline* embedding pre-computation (free Colab T4). Ranking is CPU-only.
- **Network**: Not used during ranking (embeddings precomputed; cross-encoder weights cached).

## Methodology (≤200 words)

Multi-stage funnel built to surface **genuine AI engineers** over keyword matchers.

**1 — Hard filters**  
Drop structurally unfit profiles: wrong experience band, pure non-tech careers without ML skills.

**2 — Honeypot detection**  
Remove concrete impossibilities only — experience exceeding career span, expert skills with zero usage, future dates. Keyword-stuffer traps are penalised later, not filtered here.

**3 — Hybrid retrieval**  
`bge-base-en-v1.5` dense cosine (offline precompute, CPU dot-product) blended **70 / 30** with BM25-Okapi.

**4 — Rule-based scoring** *(parallel)*  
Title/career (30) · trust-weighted skills (25) · experience (15) · education (10) · location (10) · penalties (−10). Evidence-gated penalty catches AI claims without ML work in actual roles.

**5 — Behavioral modifier**  
13 signals → multiplier **0.5×–1.2×** (response, recency, GitHub, offer-acceptance, network, etc.).

**6 — Score combination**  
`base = 0.375 × semantic + 0.625 × rule` → `final = behavioral × base`.

**7 — Cross-encoder re-rank**  
`bge-reranker-base` over the top-200 shortlist, blended **60 / 40** with the base score for top-of-funnel precision.

**Compute:** ranking is CPU-only, no network, ~1 min. Embedding pre-computation is offline (GPU-friendly).
