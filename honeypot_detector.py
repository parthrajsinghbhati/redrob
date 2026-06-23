"""
Redrob AI Candidate Ranking — Honeypot Detector
=================================================
Identifies candidates with *concretely impossible* profiles (the ~80 honeypots
the challenge seeds with subtly impossible data).

Design principle: HIGH PRECISION over recall. We only drop a candidate when the
profile is structurally impossible (verifiable data contradiction), never on
"looks too good." A genuinely excellent candidate has great signals — that's not
a honeypot. Keyword-stuffers and other "trap" profiles are NOT dropped here;
they are handled as score penalties in candidate_analyzer.py and sink naturally.

These checks were calibrated against the full 100K pool: each fires on a small
cluster (tens of candidates), matching the spec's examples ("expert proficiency
in N skills with 0 years used", "more claimed experience than the career spans").

Honeypot checks run at Stage 1.5 (immediately after hard filters, before any
expensive scoring) so we never waste compute on profiles we'll discard.
"""

from datetime import timedelta

from config import (
    EXP_SPAN_TOLERANCE_YEARS,
    EXPERT_ZERO_DURATION_MIN,
    ROLE_TENURE_SLACK_MONTHS,
    FUTURE_DATE_GRACE_DAYS,
)
from utils import (
    parse_date,
    safe_get_signal,
    REFERENCE_DATE,
)


# ============================================================================
# Public API
# ============================================================================

def is_honeypot(candidate: dict) -> bool:
    """
    Return True if the candidate has concrete data anomalies that mark it as a
    honeypot (subtly impossible profile). High precision: any single fired check
    is a verifiable impossibility.
    """
    return (
        _check_experience_vs_career_span(candidate)
        or _check_role_tenure_vs_experience(candidate)
        or _check_expert_zero_duration(candidate)
        or _check_future_dates(candidate)
        or _check_assessment_vs_proficiency(candidate)
    )


# ============================================================================
# Individual checks
# ============================================================================

def _check_experience_vs_career_span(candidate: dict) -> bool:
    """
    Flag if claimed years_of_experience drastically exceeds the actual span
    covered by career_history entries.

    Tolerance accounts for unlisted early roles (default 3 years).
    """
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience")
    career = candidate.get("career_history") or []

    if yoe is None or not career:
        return False

    start_dates = [parse_date(r.get("start_date")) for r in career]
    start_dates = [d for d in start_dates if d]
    if not start_dates:
        return False

    actual_span_years = (REFERENCE_DATE - min(start_dates)).days / 365.25
    return yoe > actual_span_years + EXP_SPAN_TOLERANCE_YEARS


def _check_role_tenure_vs_experience(candidate: dict) -> bool:
    """
    Flag if any single role's duration exceeds the candidate's TOTAL claimed
    experience (you cannot work one job longer than your entire career).

    Mirrors the spec's "8 years at a company founded 3 years ago" family of
    impossibilities, expressed via the data we have (tenure vs total years).
    """
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience")
    career = candidate.get("career_history") or []

    if not yoe or yoe <= 0 or not career:
        return False

    max_allowed_months = yoe * 12 + ROLE_TENURE_SLACK_MONTHS
    return any(
        (role.get("duration_months") or 0) > max_allowed_months
        for role in career
    )


def _check_expert_zero_duration(candidate: dict) -> bool:
    """
    Flag profiles claiming "expert" proficiency in multiple skills with ZERO
    months of usage — the spec's canonical honeypot ("expert proficiency in 10
    skills with 0 years used").
    """
    skills = candidate.get("skills") or []
    expert_zero = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert" and not s.get("duration_months")
    )
    return expert_zero >= EXPERT_ZERO_DURATION_MIN


def _check_future_dates(candidate: dict) -> bool:
    """
    Flag if key dates are in the future (beyond a grace window).
    Checks: signup_date, last_active_date, career start/end dates.
    """
    grace = REFERENCE_DATE + timedelta(days=FUTURE_DATE_GRACE_DAYS)

    for field in ("signup_date", "last_active_date"):
        dt = parse_date(safe_get_signal(candidate, field))
        if dt and dt > grace:
            return True

    for role in (candidate.get("career_history") or []):
        for field in ("start_date", "end_date"):
            dt = parse_date(role.get(field))
            # end_date=null is fine (current role); an actual future date isn't.
            if dt and dt > grace:
                return True

    return False


def _check_assessment_vs_proficiency(candidate: dict) -> bool:
    """
    Flag if assessment scores contradict stated proficiency:
    - beginner proficiency with assessment > 85 → impossible
    - expert proficiency with assessment < 15 → impossible
    Requires 2+ contradictions (one could be noise).
    """
    signals = candidate.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores") or {}
    skills = candidate.get("skills") or []

    if not assessments or not skills:
        return False

    skill_map = {s["name"]: s for s in skills if "name" in s}
    contradictions = 0

    for skill_name, score in assessments.items():
        if skill_name not in skill_map:
            continue
        proficiency = skill_map[skill_name].get("proficiency", "")
        if proficiency == "beginner" and score > 85:
            contradictions += 1
        elif proficiency == "expert" and score < 15:
            contradictions += 1

    return contradictions >= 2
