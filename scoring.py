"""
Redrob AI Candidate Ranking — Scoring Engine
==============================================
Combines semantic, rule-based, and behavioral scores into a single
final score, and computes the behavioral-signal multiplier.
"""

import math

from config import (
    SEMANTIC_WEIGHT,
    RULE_BASED_WEIGHT,
    BEHAVIORAL_FLOOR,
    BEHAVIORAL_RANGE,
    BEHAVIORAL_SIGNAL_WEIGHTS,
    RULE_BASED_TOTAL_MAX,
    IDEAL_NOTICE_DAYS,
    MAX_ACCEPTABLE_NOTICE_DAYS,
)
from utils import (
    days_since,
    clamp,
)


# ============================================================================
# Behavioral Multiplier   (0.5 → 1.2)
# ============================================================================

def compute_behavioral_multiplier(candidate: dict) -> float:
    """
    Compute the behavioral-signal multiplier for a candidate.

    Each of the 13 signals is normalised to [0, 1], weighted, summed,
    then mapped into the [BEHAVIORAL_FLOOR, BEHAVIORAL_CEILING] range.
    """
    signals = candidate.get("redrob_signals") or {}
    weights = BEHAVIORAL_SIGNAL_WEIGHTS

    composite = 0.0

    # 1. Recruiter response rate (0-1, higher is better)
    rr = _s(signals, "recruiter_response_rate", 0.0)
    composite += weights["recruiter_response_rate"] * clamp(rr, 0, 1)

    # 2. Average response time (lower is better; 0h=perfect, 168h+=bad)
    art = _s(signals, "avg_response_time_hours", 168.0)
    art_norm = 1.0 - clamp(art / 168.0, 0, 1)
    composite += weights["avg_response_time"] * art_norm

    # 3. Last active recency (active within 90 days = good)
    last_active = signals.get("last_active_date")
    d = days_since(last_active)
    if d is not None:
        recency = 1.0 - clamp(d / 365.0, 0, 1)  # Within a year
    else:
        recency = 0.3
    composite += weights["last_active_recency"] * recency

    # 4. Open to work (binary)
    otw = 1.0 if _s(signals, "open_to_work_flag", False) else 0.3
    composite += weights["open_to_work"] * otw

    # 5. Profile completeness (0-100 → 0-1)
    pc = _s(signals, "profile_completeness_score", 50.0)
    composite += weights["profile_completeness"] * clamp(pc / 100, 0, 1)

    # 6. Interview completion rate (0-1)
    icr = _s(signals, "interview_completion_rate", 0.5)
    composite += weights["interview_completion_rate"] * clamp(icr, 0, 1)

    # 7. Notice period (lower is better; 0-30 = perfect, 90+ = bad)
    np_days = _s(signals, "notice_period_days", 90)
    if np_days <= IDEAL_NOTICE_DAYS:
        np_norm = 1.0
    elif np_days <= MAX_ACCEPTABLE_NOTICE_DAYS:
        np_norm = 0.6
    else:
        np_norm = 0.2
    composite += weights["notice_period"] * np_norm

    # 8. GitHub activity (-1 = no GitHub, 0-100 = score)
    gh = _s(signals, "github_activity_score", -1)
    if gh < 0:
        gh_norm = 0.3  # No GitHub is neutral, not punishing
    else:
        gh_norm = clamp(gh / 100, 0, 1)
    composite += weights["github_activity"] * gh_norm

    # 9. Verification (email + phone + LinkedIn)
    v_email = 1 if _s(signals, "verified_email", False) else 0
    v_phone = 1 if _s(signals, "verified_phone", False) else 0
    v_li = 1 if _s(signals, "linkedin_connected", False) else 0
    v_norm = (v_email + v_phone + v_li) / 3.0
    composite += weights["verification"] * v_norm

    # 10. Saved by recruiters (social proof, log-scaled)
    saved = _s(signals, "saved_by_recruiters_30d", 0)
    saved_norm = min(1.0, math.log1p(saved) / math.log1p(20))
    composite += weights["saved_by_recruiters"] * saved_norm

    # 11. Offer acceptance rate (0-1, higher = more serious / closeable candidate)
    oar = _s(signals, "offer_acceptance_rate", 0.5)
    composite += weights.get("offer_acceptance_rate", 0) * clamp(oar, 0, 1)

    # 12. Applications activity (submitted + recruiter-search appearances ⇒ actively looking)
    apps = _s(signals, "applications_submitted_30d", 0)
    searches = _s(signals, "search_appearance_30d", 0)
    activity_norm = min(
        1.0,
        (math.log1p(apps) / math.log1p(10) + math.log1p(searches) / math.log1p(100)) / 2,
    )
    composite += weights.get("applications_activity", 0) * activity_norm

    # 13. Network strength (log-scaled professional connection count)
    connections = _s(signals, "connection_count", 0)
    network_norm = min(1.0, math.log1p(connections) / math.log1p(500))
    composite += weights.get("network_strength", 0) * network_norm

    # Map weighted sum [0, 1] → [BEHAVIORAL_FLOOR, BEHAVIORAL_CEILING]
    return BEHAVIORAL_FLOOR + BEHAVIORAL_RANGE * clamp(composite, 0, 1)


def _s(signals: dict, key: str, default):
    """Get a signal value with a default if None or missing."""
    val = signals.get(key)
    return val if val is not None else default


# ============================================================================
# Final Score Combination
# ============================================================================

def compute_final_score(
    semantic_score: float,
    rule_score: float,
    behavioral_multiplier: float,
) -> float:
    """
    Combine the three scoring dimensions into one final score.

    The behavioral value is a TRUE multiplier (Issue 4 / Option A): the
    semantic and rule-based scores form a weighted base in [0, 1], and the
    behavioral multiplier (0.5–1.2) scales that base up or down. An inactive
    candidate therefore loses points (0.5× base) rather than receiving a free
    additive floor, while a highly-engaged one is boosted (up to 1.2× base).

    - semantic_score:         [0, 1]  cosine similarity
    - rule_score:             [0, 100]  raw rule-based points
    - behavioral_multiplier:  [0.5, 1.2]
    """
    sem_norm = clamp(semantic_score, 0, 1)
    rule_norm = clamp(rule_score / RULE_BASED_TOTAL_MAX, 0, 1)

    weight_sum = SEMANTIC_WEIGHT + RULE_BASED_WEIGHT
    base = (
        (SEMANTIC_WEIGHT / weight_sum) * sem_norm
        + (RULE_BASED_WEIGHT / weight_sum) * rule_norm
    )

    return clamp(behavioral_multiplier * base, 0, 1)
