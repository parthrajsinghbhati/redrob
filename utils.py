"""
Redrob AI Candidate Ranking — Shared Utilities
================================================
Streaming JSONL reader, safe field accessors, date parsing.
"""

import json
import math
from datetime import datetime
from typing import Any, Generator

from config import DEFAULTS, DATASET_REFERENCE_DATE


# ============================================================================
# JSONL Streaming
# ============================================================================

def stream_candidates(filepath: str) -> Generator[dict, None, None]:
    """
    Lazily iterate over a JSONL file one candidate at a time.
    Never loads the full file into memory.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ============================================================================
# Safe Field Access  (always returns a usable value)
# ============================================================================

def safe_get(data: dict, *keys, default=None):
    """
    Drill into nested dicts safely.

        safe_get(candidate, "profile", "years_of_experience", default=0.0)

    Returns the configured default from DEFAULTS when the field is missing
    or None, unless an explicit *default* is supplied.
    """
    current = data
    final_key = keys[-1] if keys else None
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = None
        if current is None:
            break

    if current is not None:
        return current

    # Fall back to explicit default, then DEFAULTS table
    if default is not None:
        return default
    if final_key and final_key in DEFAULTS:
        return DEFAULTS[final_key]
    return None


def safe_get_signal(candidate: dict, signal_name: str):
    """Shorthand for accessing redrob_signals fields with defaults."""
    signals = candidate.get("redrob_signals") or {}
    val = signals.get(signal_name)
    if val is not None:
        return val
    return DEFAULTS.get(signal_name)


# ============================================================================
# Date Utilities
# ============================================================================

# Pinned reference "now" (see config.DATASET_REFERENCE_DATE) — keeps date-based
# checks reproducible regardless of when the pipeline runs.
REFERENCE_DATE = datetime.strptime(DATASET_REFERENCE_DATE, "%Y-%m-%d")


def parse_date(date_str: str | None) -> datetime | None:
    """Parse an ISO-format date string. Returns None on failure."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def days_since(date_str: str | None) -> int | None:
    """Days between a date string and today.  Positive = past."""
    dt = parse_date(date_str)
    if dt is None:
        return None
    return (REFERENCE_DATE - dt).days


def months_between(start_str: str | None, end_str: str | None) -> float:
    """Approximate months between two date strings."""
    s = parse_date(start_str)
    e = parse_date(end_str) if end_str else REFERENCE_DATE
    if s is None or e is None:
        return 0.0
    delta_days = (e - s).days
    return max(0.0, delta_days / 30.44)


# ============================================================================
# Numeric helpers
# ============================================================================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def log_scale(value: float, cap: float = 50.0) -> float:
    """Logarithmic normalisation: 0→0, cap→1."""
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(cap))


def normalize(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] → [0, 1], clamped."""
    if hi <= lo:
        return 0.5
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


# ============================================================================
# Text helpers
# ============================================================================

def build_candidate_text(candidate: dict) -> str:
    """
    Build a single text blob from a candidate dict for embedding.
    Covers headline, summary, and all career descriptions.
    Truncated to ~2000 chars to keep encoding fast.
    """
    parts = []

    profile = candidate.get("profile") or {}
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""
    parts.append(headline)
    parts.append(summary)

    for role in (candidate.get("career_history") or []):
        desc = role.get("description") or ""
        title = role.get("title") or ""
        company = role.get("company") or ""
        parts.append(f"{title} at {company}: {desc}")

    text = " ".join(p for p in parts if p)
    return text[:2000]  # Truncate to keep encoding fast
