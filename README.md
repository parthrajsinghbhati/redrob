# Redrob AI — Intelligent Candidate Discovery & Ranking

An AI-powered candidate ranking system that goes beyond keyword matching to understand who genuinely fits a role. Built for the Redrob Hackathon (India Runs Data & AI Challenge).

## Architecture

```
100K Candidates (JSONL)
    │
    ├── Stage 1:   Hard Filters          (structural disqualifiers)
    ├── Stage 1.5: Honeypot Detection    (concrete profile anomalies)
    ├── Stage 2:   Semantic Scoring      (dense cosine vs JD — precomputed BGE embeddings)
    ├── Stage 3:   Rule-Based Scoring    (parallel — title, skills, career, education, location)
    ├── Stage 4:   Behavioral Modifier   (platform engagement multiplier)
    ├── Stage 5:   Score Combination     (0.30 semantic + 0.50 rule + 0.20 behavioral)
    ├── Stage 5.5: Cross-Encoder Re-rank (bge-reranker-base over the top-200 shortlist)
    └── Stage 6:   Top-100 + Reasoning   (rank-aware templates, real profile data)
```

## Key Design Decisions

1. **Dense retrieval, precomputed offline.** Candidate profiles and the JD are encoded with `BAAI/bge-base-en-v1.5` (768-dim) once, offline (GPU-friendly, allowed to exceed the 5-min window). The ranking step only *loads* the vectors and does a CPU dot-product — so it needs no GPU and no network. This catches "plain-language" strong candidates who describe their work naturally without buzzwords.

2. **Cross-encoder re-ranking of a shortlist.** A bi-encoder is fast but coarse; a cross-encoder (`BAAI/bge-reranker-base`) is precise but expensive. We get both by re-ranking only the top-200 shortlist — precision where it matters (NDCG@10) while staying inside the CPU/5-min budget. It degrades gracefully (dense+rule order) if the weights aren't available locally.

3. **Anti-keyword-stuffing.** Skills are scored with a *trust multiplier* based on proficiency, endorsements, duration, and assessment scores — not just presence/absence.

4. **Career trajectory analysis.** Detects consulting-only careers, title-description mismatches, and job-hopping patterns. Product-company experience is weighted heavily.

5. **Honeypot detection.** Flags only profiles with verifiable, concrete impossibilities (experience exceeding the career span, a role longer than the whole career, "expert" skills with zero usage, future dates) — high precision, so it never discards genuinely excellent candidates. Keyword-stuffer traps are handled separately as score penalties.

6. **Behavioral signals as a multiplier.** Active, responsive candidates get a boost; inactive ones a moderate discount (floor 0.5×, ceiling 1.2×).

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

Multi-stage funnel with hybrid scoring. Stage 1 eliminates structurally unfit candidates (wrong experience band, pure non-tech careers). Stage 1.5 removes honeypots via high-precision concrete-anomaly detection (claimed experience exceeding the career span, a single role longer than the whole career, multiple "expert" skills with zero months used, future dates, assessment contradictions) — calibrated to catch the seeded impossibilities without discarding good candidates. Stage 2 computes dense semantic similarity using `bge-base-en-v1.5` embeddings precomputed offline — at rank time this is a pure CPU dot-product, catching "plain-language" strong candidates. Stage 3 runs parallel rule-based scoring: title/career trajectory (30pts), skill relevance with trust weighting (25pts), experience depth with production-keyword analysis (15pts), education (10pts), location fit (10pts), and career-pattern penalties (−10pts for consulting-only, keyword stuffers, job hoppers). Stage 4 applies a behavioral multiplier (0.5–1.2×) over response rate, recency, GitHub activity, notice period, and verification. Stage 5 combines 30% semantic + 50% rule + 20% behavioral. Stage 5.5 re-ranks the top-200 shortlist with a `bge-reranker-base` cross-encoder (blended 0.6/0.4 with the base score) for top-of-funnel precision, within the CPU budget. The skill trust multiplier is the core anti-stuffing mechanism: it cross-references proficiency claims against endorsements, usage duration, and assessment scores. Ranking runtime: a few minutes on CPU; embedding pre-computation is offline.
```
