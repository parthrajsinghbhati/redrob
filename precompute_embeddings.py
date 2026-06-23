"""
Redrob AI Candidate Ranking — Pre-compute Dense Embeddings
===========================================================
Encodes every candidate profile into a dense vector with a BGE
sentence-transformer and stores the matrix on disk, alongside a single
embedding for the job description.

This is the ONLY step that benefits from a GPU.  It is run once, offline,
and is allowed to exceed the 5-minute ranking budget (per submission_spec
Section 3).  The ranking step (rank.py) only *loads* these files and does
a CPU dot-product — no model, no network.

Outputs (written to --out-dir, default ./data):
    embeddings.npz   -> {candidate_ids: (N,), embeddings: (N, D) float32}
    jd_embedding.npy -> (D,) float32

The candidate matrix and the JD vector are L2-normalised, so cosine
similarity is a plain dot-product at rank time.

Usage (CPU or GPU; auto-detected):
    python precompute_embeddings.py \\
        --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl \\
        --out-dir ./data

On a free Google Colab T4 GPU this encodes 100K candidates in a couple of
minutes with --model BAAI/bge-base-en-v1.5 --fp16.
"""

import argparse
import os
import time

import numpy as np

from config import EMBEDDING_MODEL, QUERY_PREFIX, JD_EMBEDDING_TEXT
from utils import stream_candidates, build_candidate_text


def precompute(
    candidates_path: str,
    out_dir: str = "data",
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 256,
    fp16: bool = False,
    max_seq_length: int | None = 256,
):
    """Encode all candidates + the JD and persist the vectors."""
    from sentence_transformers import SentenceTransformer
    import torch

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  model: {model_name}")

    # --- 1. Read candidate texts (streaming, never loads file fully) ---
    t0 = time.time()
    candidate_ids: list[str] = []
    candidate_texts: list[str] = []
    for cand in stream_candidates(candidates_path):
        candidate_ids.append(cand["candidate_id"])
        candidate_texts.append(build_candidate_text(cand))
    print(f"Loaded {len(candidate_ids)} candidates in {time.time() - t0:.1f}s")

    # --- 2. Encode ---
    model = SentenceTransformer(model_name, device=device)
    if fp16 and device == "cuda":
        model.half()
    if max_seq_length:
        model.max_seq_length = max_seq_length

    t0 = time.time()
    embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"Encoded {embeddings.shape} in {time.time() - t0:.1f}s")

    # Documents get no prefix; the query (JD) gets the BGE retrieval prefix.
    jd_embedding = model.encode(
        [QUERY_PREFIX + JD_EMBEDDING_TEXT],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    # --- 3. Persist ---
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        candidate_ids=np.array(candidate_ids),
        embeddings=embeddings,
    )
    np.save(os.path.join(out_dir, "jd_embedding.npy"), jd_embedding)

    sims = embeddings @ jd_embedding
    print(
        f"Saved -> {out_dir}/embeddings.npz + jd_embedding.npy  "
        f"(sim min/max/mean = {sims.min():.4f}/{sims.max():.4f}/{sims.mean():.4f})"
    )
    return candidate_ids, embeddings, jd_embedding


def load_embeddings(data_dir: str = "data"):
    """
    Load pre-computed candidate embeddings + JD embedding.

    Returns
    -------
    candidate_ids : np.ndarray[str] | None
    embeddings    : np.ndarray (N, D) float32 | None  (L2-normalised)
    jd_embedding  : np.ndarray (D,) float32 | None    (L2-normalised)
    """
    emb_path = os.path.join(data_dir, "embeddings.npz")
    jd_path = os.path.join(data_dir, "jd_embedding.npy")
    if not (os.path.exists(emb_path) and os.path.exists(jd_path)):
        return None, None, None

    data = np.load(emb_path, allow_pickle=True)
    candidate_ids = data["candidate_ids"]
    embeddings = data["embeddings"].astype(np.float32)
    jd_embedding = np.load(jd_path).astype(np.float32)
    return candidate_ids, embeddings, jd_embedding


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute dense embeddings for candidate ranking."
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out-dir", default="data", help="Output dir (default: ./data)")
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="sentence-transformers model")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", help="Half precision (GPU only)")
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    precompute(
        candidates_path=args.candidates,
        out_dir=args.out_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        fp16=args.fp16,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
