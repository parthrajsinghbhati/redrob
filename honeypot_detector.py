"""
Redrob AI Candidate Ranking — Honeypot Detector
=================================================
Identifies candidates with *concretely impossible* profiles.

Only flags on verifiable data anomalies — never on "profile looks too good."
A genuinely excellent candidate WILL have great signals; that's not a honeypot.

Honeypot checks run at Stage 1.5 (immediately after hard filters, before any
expensive scoring) so we never waste compute on profiles we'll discard.
"""

from datetime import timedelta

from utils import (
    parse_date,
    safe_get,
    safe_get_signal,
    REFERENCE_DATE,
)


# ============================================================================
# Public API
# ============================================================================

def is_honeypot(candidate: dict) -> bool:
    """
    Return True if the candidate has concrete data anomalies that mark
    it as a honeypot (subtly impossible profile).
    """
    flags = []
    flags.append(_check_experience_vs_career_span(candidate))
    flags.append(_check_future_dates(candidate))
    flags.append(_check_assessment_vs_proficiency(candidate))
    flags.append(_check_skill_duration_vs_career(candidate))
    flags.append(_check_career_domain_skill_disconnect(candidate))
    return any(flags)


# ============================================================================
# Individual checks
# ============================================================================

def _check_experience_vs_career_span(candidate: dict) -> bool:
    """
    Flag if claimed years_of_experience drastically exceeds the actual
    span covered by career_history entries.

    Tolerance: 3 years (accounts for unlisted early roles).
    """
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience")
    career = candidate.get("career_history") or []

    if yoe is None or not career:
        return False

    start_dates = []
    for role in career:
        sd = parse_date(role.get("start_date"))
        if sd:
            start_dates.append(sd)

    if not start_dates:
        return False

    earliest = min(start_dates)
    actual_span_years = (REFERENCE_DATE - earliest).days / 365.25

    # Impossible: claims 15 years but career only spans 3
    if yoe > actual_span_years + 3.0:
        return True

    return False


def _check_future_dates(candidate: dict) -> bool:
    """
    Flag if key dates are in the future (beyond a 30-day grace window).
    Checks: signup_date, last_active_date, career start/end dates.
    """
    grace = REFERENCE_DATE + timedelta(days=30)

    # Check signal dates
    for field in ("signup_date", "last_active_date"):
        dt = parse_date(safe_get_signal(candidate, field))
        if dt and dt > grace:
            return True

    # Check career dates
    for role in (candidate.get("career_history") or []):
        for field in ("start_date", "end_date"):
            dt = parse_date(role.get(field))
            # end_date=null is fine (current role), but an actual future date isn't
            if dt and dt > grace:
                return True

    return False


def _check_assessment_vs_proficiency(candidate: dict) -> bool:
    """
    Flag if assessment scores contradict stated proficiency.

    - Beginner proficiency with assessment > 85 → impossible
    - Expert proficiency with assessment < 15 → impossible
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

    # One contradiction could be noise; two+ is suspicious
    return contradictions >= 2


def _check_skill_duration_vs_career(candidate: dict) -> bool:
    """
    Flag if the sum of skill durations vastly exceeds total career span.
    A person with 3 years of career can't have 10 skills each with 60 months
    of usage.
    """
    profile = candidate.get("profile") or {}
    yoe = profile.get("years_of_experience", 0)
    skills = candidate.get("skills") or []

    if not skills or yoe <= 0:
        return False

    career_months = yoe * 12
    total_skill_months = sum(
        s.get("duration_months", 0)
        for s in skills
        if s.get("duration_months")
    )

    # Skills can overlap (you use Python AND PyTorch at the same time),
    # so allow generous headroom: 3× career span is fine;  8× is not.
    if total_skill_months > career_months * 8 and len(skills) >= 5:
        return True

    return False


def _check_career_domain_skill_disconnect(candidate: dict) -> bool:
    """
    Flag if ALL career roles are in a completely unrelated domain AND
    ALL skills are in a different domain, with no transition narrative
    in any role description.

    This is stricter than keyword-stuffer detection (which is handled
    in candidate_analyzer as a score penalty).  Here we look for profiles
    that are *structurally incoherent* — e.g., every role is "Accountant"
    at finance firms, but skills are 100% ML/AI, and no description
    mentions learning, transitioning, or side projects.
    """
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    if not career or not skills:
        return False

    # Check if ALL titles are non-tech
    non_tech_titles = {
        "accountant", "hr manager", "sales executive",
        "marketing manager", "operations manager",
        "customer support", "graphic designer",
        "civil engineer", "mechanical engineer",
    }

    titles = [
        (role.get("title") or "").lower().strip()
        for role in career
    ]
    all_non_tech = all(t in non_tech_titles for t in titles if t)

    if not all_non_tech:
        return False

    # Check if skills are overwhelmingly AI/ML
    ai_keywords = {
        "pytorch", "tensorflow", "nlp", "machine learning",
        "deep learning", "faiss", "embeddings", "bert",
        "transformers", "rag", "fine-tuning llms", "lora",
        "neural networks", "reinforcement learning",
    }
    skill_names_lower = {
        (s.get("name") or "").lower() for s in skills
    }
    ai_skill_count = len(skill_names_lower & ai_keywords)

    if ai_skill_count < 4:
        return False

    # Check descriptions for transition language
    transition_words = {
        "transition", "career change", "self-taught", "bootcamp",
        "side project", "personal project", "kaggle", "coursera",
        "learning", "hobby", "upskill",
    }
    all_descs = " ".join(
        (role.get("description") or "").lower() for role in career
    )
    has_transition = any(w in all_descs for w in transition_words)

    if has_transition:
        return False

    # ALL titles are non-tech, 4+ core AI skills, no transition language
    # → structurally incoherent, likely honeypot
    return True
