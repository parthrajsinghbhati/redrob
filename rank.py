"""
Redrob AI Candidate Ranking — Main Pipeline
=============================================
Ranking pipeline that produces the top-100 CSV.

Stages:
  1.   Hard Filters          – Eliminate structurally disqualified candidates
  1.5  Honeypot Detection    – Remove concretely impossible profiles
  2.   Semantic Scoring      – Hybrid retrieval: dense cosine (precomputed BGE
                               embeddings) blended with lexical BM25 (Okapi)
  3.   Rule-Based Scoring    – Parallel per-candidate analysis (joblib)
  4.   Behavioral Modifier   – Platform engagement multiplier
  5.   Final Combination     – Weighted merge (0.30 sem + 0.50 rule + 0.20 beh)
  5.5  Cross-Encoder Re-rank – Re-rank the top shortlist for top-of-funnel precision
  6.   Top-100 + Reasoning   – Rank, generate reasoning, write CSV

The dense embeddings are pre-computed offline (precompute_embeddings.py);
this script only LOADS them and does a CPU dot-product, so the ranking step
needs no GPU and no network.  The optional cross-encoder re-rank runs only
on a small shortlist and only if its weights are available locally.

Usage:
    python rank.py \\
        --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \\
        --out ./submission.csv
"""

import argparse
import csv
import logging
import time

import numpy as np
from joblib import Parallel, delayed

from config import (
    MIN_EXPERIENCE_YEARS,
    MAX_EXPERIENCE_YEARS,
    ML_ADJACENT_TITLES,
    NON_TECH_TITLES,
    CORE_AI_SKILLS,
    N_JOBS,
    JD_EMBEDDING_TEXT,
    BM25_WEIGHT,
    RERANKER_MODEL,
    RERANK_SHORTLIST,
    RERANK_MAX_LENGTH,
    RERANK_BATCH_SIZE,
    RERANK_WEIGHT,
    NORMALIZATION_TARGET_MIN,
    NORMALIZATION_TARGET_MAX,
)
from utils import stream_candidates, build_candidate_text
from honeypot_detector import is_honeypot
from candidate_analyzer import compute_rule_score
from scoring import (
    compute_behavioral_multiplier,
    compute_final_score,
)
from reasoning import generate_reasoning
from precompute_embeddings import load_embeddings

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Stage 1: Hard Filters
# ============================================================================

def passes_hard_filter(candidate: dict) -> bool:
    """Fast structural checks.  Returns True if the candidate survives."""
    profile = candidate.get("profile") or {}

    # --- Experience floor ---
    yoe = profile.get("years_of_experience")
    if yoe is None or yoe < MIN_EXPERIENCE_YEARS:
        return False

    # --- Experience ceiling + no ML titles ---
    if yoe > MAX_EXPERIENCE_YEARS:
        title = (profile.get("current_title") or "").lower().strip()
        if title not in ML_ADJACENT_TITLES:
            return False

    # --- Pure non-tech career (all roles are non-tech, no overlap) ---
    career = candidate.get("career_history") or []
    if career:
        titles = [(r.get("title") or "").lower().strip() for r in career]
        if all(t in NON_TECH_TITLES for t in titles if t):
            skills = candidate.get("skills") or []
            skill_names = {(s.get("name") or "").lower() for s in skills}
            core_hits = len(skill_names & CORE_AI_SKILLS)
            if core_hits < 2:
                return False

    return True


# ============================================================================
# Stage 2: Semantic similarity from precomputed dense embeddings
# ============================================================================

def build_semantic_index(data_dir: str = "data") -> dict[str, float]:
    """
    Load precomputed embeddings, compute cosine similarity to the JD, and
    min-max normalise to [0, 1] so the semantic signal has full spread.

    Returns a dict candidate_id -> normalised semantic score.
    """
    ids, emb, jd = load_embeddings(data_dir)
    if ids is None:
        raise FileNotFoundError(
            f"No embeddings found in '{data_dir}/'. Run:\n"
            "  python precompute_embeddings.py "
            "--candidates <candidates.jsonl> --out-dir data"
        )

    # Vectors are already L2-normalised, so cosine == dot product.
    sims = emb @ jd
    lo, hi = float(sims.min()), float(sims.max())
    if hi > lo:
        sims_norm = (sims - lo) / (hi - lo)
    else:
        sims_norm = np.full_like(sims, 0.5)

    logger.info(
        "  Loaded %d embeddings (dim=%d) — raw cosine [%.4f, %.4f]",
        emb.shape[0], emb.shape[1], lo, hi,
    )
    return dict(zip(ids.tolist(), sims_norm.tolist()))


# ============================================================================
# Stage 2b: Lexical BM25 (Okapi) — the lexical half of hybrid retrieval
# ============================================================================

def build_bm25_scores(survivors: list[dict], jd_text: str) -> list[float] | None:
    """
    Score survivors against the JD with BM25-Okapi (pure-CPU, no GPU/network).

    Returns a list of min-max normalised scores aligned with `survivors`, or
    None if `rank_bm25` is unavailable — in which case the caller keeps the
    dense-only semantic scores.
    """
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:  # noqa: BLE001
        logger.warning("rank_bm25 import failed (%s) — using dense-only semantic", e)
        return None

    tokenized_corpus = [build_candidate_text(c).lower().split() for c in survivors]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(jd_text.lower().split())

    lo, hi = float(scores.min()), float(scores.max())
    if hi > lo:
        return ((scores - lo) / (hi - lo)).tolist()
    return [0.5] * len(survivors)


# ============================================================================
# Stage 5.5: Cross-encoder re-rank (shortlist only, best-effort)
# ============================================================================

def rerank_shortlist(shortlist: list[dict], query_text: str) -> list[float] | None:
    """
    Score (JD, candidate) pairs with a cross-encoder for the shortlist.

    Returns a list of raw relevance scores aligned with `shortlist`, or None
    if the cross-encoder is unavailable (e.g. weights not cached, no network).
    The caller falls back to the dense+rule order when None is returned.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:  # noqa: BLE001
        logger.warning("CrossEncoder import failed (%s) — skipping re-rank", e)
        return None

    try:
        # Force CPU: the ranking step must be CPU-only per the compute rules.
        model = CrossEncoder(RERANKER_MODEL, max_length=RERANK_MAX_LENGTH, device="cpu")
        pairs = [[query_text, build_candidate_text(c)] for c in shortlist]
        scores = model.predict(
            pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False
        )
        return [float(s) for s in scores]
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Cross-encoder re-rank failed (%s) — falling back to dense+rule order", e
        )
        return None


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


# ============================================================================
# CSV Writer
# ============================================================================

def write_csv(ranked: list[dict], output_path: str):
    """Write the top-100 submission CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c in ranked:
            writer.writerow([
                c["candidate_id"],
                c["rank"],
                f"{c['out_score']:.4f}",
                c["reasoning"],
            ])
    logger.info("Wrote %d rows → %s", len(ranked), output_path)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Redrob AI challenge."
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument(
        "--data-dir", default="data", help="Dir with precomputed embeddings"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable the cross-encoder re-rank"
    )
    args = parser.parse_args()

    t_start = time.time()

    # ==================================================================
    # Stage 2 (pre-step): Load dense semantic index
    # ==================================================================
    logger.info("Stage 2 (pre-step): Loading dense embeddings ...")
    semantic_index = build_semantic_index(args.data_dir)

    # ==================================================================
    # Stage 1 + 1.5: Stream → Hard Filter → Honeypot Detection
    # ==================================================================
    logger.info("Stage 1+1.5: Streaming candidates, applying filters ...")
    survivors = []
    total = filtered_out = honeypots = 0

    for candidate in stream_candidates(args.candidates):
        total += 1
        if not passes_hard_filter(candidate):
            filtered_out += 1
            continue
        if is_honeypot(candidate):
            honeypots += 1
            continue
        survivors.append(candidate)

    logger.info(
        "  %d total → %d filtered → %d honeypots → %d survivors",
        total, filtered_out, honeypots, len(survivors),
    )

    # ==================================================================
    # Stage 2: Assign semantic scores
    # ==================================================================
    logger.info("Stage 2: Assigning semantic scores to survivors ...")
    for c in survivors:
        c["_dense"] = semantic_index.get(c["candidate_id"], 0.0)
        c["_semantic"] = c["_dense"]

    # ==================================================================
    # Stage 2b: Lexical BM25, blended into the semantic signal (hybrid)
    # ==================================================================
    logger.info("Stage 2b: BM25 lexical scoring (hybrid retrieval) ...")
    t_bm = time.time()
    bm25_scores = build_bm25_scores(survivors, JD_EMBEDDING_TEXT)
    if bm25_scores is not None:
        for c, bm in zip(survivors, bm25_scores):
            c["_bm25"] = bm
            c["_semantic"] = (1 - BM25_WEIGHT) * c["_dense"] + BM25_WEIGHT * bm
        logger.info(
            "  BM25 done in %.1fs — blended %.0f%% dense / %.0f%% BM25",
            time.time() - t_bm, (1 - BM25_WEIGHT) * 100, BM25_WEIGHT * 100,
        )

    # ==================================================================
    # Stage 3: Rule-Based Scoring  (parallel via joblib)
    # ==================================================================
    logger.info(
        "Stage 3: Rule-based scoring (%d candidates, n_jobs=%s) ...",
        len(survivors), N_JOBS,
    )
    t0 = time.time()
    rule_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(compute_rule_score)(c) for c in survivors
    )
    for c, scores in zip(survivors, rule_results):
        c["_rule_scores"] = scores
        c["_rule_total"] = scores["rule_total"]
    logger.info("  Rule-based scoring done in %.1fs", time.time() - t0)

    # ==================================================================
    # Stage 4: Behavioral Multiplier
    # ==================================================================
    logger.info("Stage 4: Behavioral signal scoring ...")
    for c in survivors:
        c["_behavioral"] = compute_behavioral_multiplier(c)

    # ==================================================================
    # Stage 5: Final Score Combination
    # ==================================================================
    logger.info("Stage 5: Combining scores ...")
    for c in survivors:
        c["final_score"] = compute_final_score(
            semantic_score=c["_semantic"],
            rule_score=c["_rule_total"],
            behavioral_multiplier=c["_behavioral"],
        )

    # Sort by base score; take the shortlist for re-ranking.
    survivors.sort(key=lambda c: (-c["final_score"], c["candidate_id"]))
    shortlist = survivors[:RERANK_SHORTLIST]

    # ==================================================================
    # Stage 5.5: Cross-encoder re-rank (shortlist only)
    # ==================================================================
    ce_scores = None
    if not args.no_rerank and shortlist:
        logger.info(
            "Stage 5.5: Cross-encoder re-rank of top %d ...", len(shortlist)
        )
        t0 = time.time()
        ce_scores = rerank_shortlist(shortlist, JD_EMBEDDING_TEXT)
        if ce_scores is not None:
            logger.info("  Re-rank done in %.1fs", time.time() - t0)

    if ce_scores is not None:
        ce_norm = _minmax(ce_scores)
        base_norm = _minmax([c["final_score"] for c in shortlist])
        for c, ce_n, base_n in zip(shortlist, ce_norm, base_norm):
            c["_ce"] = ce_n
            c["combined"] = RERANK_WEIGHT * ce_n + (1 - RERANK_WEIGHT) * base_n
    else:
        for c in shortlist:
            c["combined"] = c["final_score"]

    # ==================================================================
    # Stage 6: Rank, Reason, Write
    # ==================================================================
    logger.info("Stage 6: Ranking and generating reasoning ...")

    # Sort by combined score DESC, then candidate_id ASC (tie-break rule).
    shortlist.sort(key=lambda c: (-c["combined"], c["candidate_id"]))
    top_100 = shortlist[:100]

    # Map combined scores onto a presentation range, then enforce strictly
    # decreasing 4-decimal output so the validator's tie-break rule can never
    # be violated by float rounding collisions.
    combined_vals = [c["combined"] for c in top_100]
    lo, hi = min(combined_vals), max(combined_vals)
    span = NORMALIZATION_TARGET_MAX - NORMALIZATION_TARGET_MIN
    prev = None
    for i, c in enumerate(top_100):
        c["rank"] = i + 1
        if hi > lo:
            s = NORMALIZATION_TARGET_MIN + (c["combined"] - lo) / (hi - lo) * span
        else:
            s = NORMALIZATION_TARGET_MAX
        s = round(s, 4)
        if prev is not None and s >= prev:
            s = round(prev - 0.0001, 4)
        c["out_score"] = s
        prev = s
        c["reasoning"] = generate_reasoning(
            candidate=c, rank=c["rank"], scores=c.get("_rule_scores", {})
        )

    write_csv(top_100, args.out)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs  (%.1f min)", elapsed, elapsed / 60)

    print("\n--- Top 5 Candidates ---")
    for c in top_100[:5]:
        p = c.get("profile", {})
        ce = c.get("_ce")
        ce_str = f"ce={ce:.3f}" if ce is not None else "ce=--"
        print(
            f"  #{c['rank']:>3}  {c['candidate_id']}  "
            f"{p.get('current_title', '?'):30s}  "
            f"score={c['out_score']:.4f}  "
            f"sem={c['_semantic']:.3f}  rule={c['_rule_total']:.1f}  "
            f"beh={c['_behavioral']:.3f}  {ce_str}"
        )


if __name__ == "__main__":
    main()
