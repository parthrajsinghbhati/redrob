"""
Redrob AI Candidate Ranking — Candidate Analyzer
==================================================
Per-candidate scoring functions for the rule-based stage.

Sub-scores:
  A. Title & Career Trajectory   (0 → 30 pts)
  B. Skill Relevance with Trust  (0 → 25 pts)
  C. Experience Depth            (0 → 15 pts)
  D. Education & Certifications  (0 → 10 pts)
  E. Location & Logistics Fit    (0 → 10 pts)
  F. Career Pattern Penalties    (0 → -10 pts)
"""

import math

from config import (
    TITLE_RELEVANCE,
    CORE_AI_SKILLS,
    NICE_TO_HAVE_SKILLS,
    RED_FLAG_SKILLS,
    CONSULTING_FIRMS,
    PREFERRED_LOCATIONS,
    ACCEPTABLE_INDIAN_CITIES,
    INDIA_COUNTRY,
    IDEAL_EXPERIENCE_MIN,
    IDEAL_EXPERIENCE_MAX,
    IDEAL_NOTICE_DAYS,
    MAX_ACCEPTABLE_NOTICE_DAYS,
    TITLE_CAREER_MAX_PTS,
    SKILL_RELEVANCE_MAX_PTS,
    EXPERIENCE_DEPTH_MAX_PTS,
    EDUCATION_MAX_PTS,
    LOCATION_MAX_PTS,
    CAREER_PATTERN_PENALTY_MAX,
    NON_TECH_TITLES,
)
from utils import safe_get, clamp, log_scale


# ============================================================================
# Master Scorer  (entry point for parallel execution)
# ============================================================================

def compute_rule_score(candidate: dict) -> dict:
    """
    Compute all rule-based sub-scores for a single candidate.
    Returns a dict of sub-scores plus the combined total.
    Safe for joblib.Parallel (no shared state).
    """
    title_pts = _score_title_career(candidate)
    skill_pts = _score_skills(candidate)
    exp_pts = _score_experience_depth(candidate)
    edu_pts = _score_education(candidate)
    loc_pts = _score_location(candidate)
    penalty = _score_career_penalties(candidate)

    total = title_pts + skill_pts + exp_pts + edu_pts + loc_pts + penalty
    total = clamp(total, 0.0, 100.0)

    return {
        "title_career": title_pts,
        "skill_relevance": skill_pts,
        "experience_depth": exp_pts,
        "education": edu_pts,
        "location": loc_pts,
        "career_penalty": penalty,
        "rule_total": total,
    }


# ============================================================================
# A. Title & Career Trajectory  (0 → 30 pts)
# ============================================================================

def _score_title_career(candidate: dict) -> float:
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []

    pts = 0.0

    # --- Current/most-recent title relevance (0-15 pts) ---
    current_title = (profile.get("current_title") or "").lower().strip()
    pts += _lookup_title_score(current_title)

    # --- Product-company experience (0-10 pts) ---
    has_product_exp = False
    product_months = 0
    for role in career:
        company = (role.get("company") or "").lower().strip()
        if company not in CONSULTING_FIRMS:
            has_product_exp = True
            product_months += role.get("duration_months", 0)

    if has_product_exp:
        # Scale: 12 months product exp = 3 pts, 48+ months = 10 pts
        pts += clamp(product_months / 48 * 10, 0, 10)

    # --- Technical career progression (0-5 pts) ---
    tech_role_count = 0
    for role in career:
        title = (role.get("title") or "").lower().strip()
        if _lookup_title_score(title) >= 5:
            tech_role_count += 1

    if tech_role_count >= 3:
        pts += 5
    elif tech_role_count == 2:
        pts += 3
    elif tech_role_count == 1:
        pts += 1

    return clamp(pts, 0, TITLE_CAREER_MAX_PTS)


def _lookup_title_score(title: str) -> float:
    """Fuzzy title matching against the relevance table."""
    title = title.lower().strip()
    if title in TITLE_RELEVANCE:
        return TITLE_RELEVANCE[title]

    # Partial match: check if any key is contained in the title
    best = 0.0
    for pattern, score in TITLE_RELEVANCE.items():
        if pattern in title or title in pattern:
            best = max(best, score)
    return best


# ============================================================================
# B. Skill Relevance with Trust Weighting  (0 → 25 pts)
# ============================================================================

def _score_skills(candidate: dict) -> float:
    skills = candidate.get("skills") or []
    signals = candidate.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores") or {}

    if not skills:
        return 0.0

    core_score = 0.0
    nice_score = 0.0
    red_flags = 0

    for skill in skills:
        name = (skill.get("name") or "").lower().strip()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)
        assessment = assessments.get(skill.get("name"))

        trust = _skill_trust(proficiency, endorsements, duration, assessment)

        if name in CORE_AI_SKILLS:
            core_score += trust * 3.0          # Core skills worth more
        elif name in NICE_TO_HAVE_SKILLS:
            nice_score += trust * 1.0
        elif name in RED_FLAG_SKILLS:
            red_flags += 1

    # Cap contributions
    core_contribution = min(core_score, 18.0)    # Max 18 from core
    nice_contribution = min(nice_score, 7.0)     # Max 7 from nice-to-have

    # Red flags reduce score (diminishing)
    red_penalty = min(red_flags * 1.0, 5.0)

    total = core_contribution + nice_contribution - red_penalty
    return clamp(total, 0, SKILL_RELEVANCE_MAX_PTS)


def _skill_trust(
    proficiency: str,
    endorsements: int,
    duration_months: int,
    assessment_score: float | None,
) -> float:
    """
    Trust multiplier for a single skill: [0.0 → 1.0].

    A skill with expert proficiency, many endorsements, long duration,
    and a high assessment is trusted.  A skill listed as "advanced" with
    0 endorsements and 2 months duration is likely stuffed.
    """
    prof_map = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
    base = prof_map.get(proficiency, 0.3)

    endorse_factor = log_scale(endorsements, cap=50.0)
    duration_factor = min(1.0, duration_months / 48.0) if duration_months else 0.0

    if assessment_score is not None and assessment_score >= 0:
        assess_factor = assessment_score / 100.0
    else:
        assess_factor = 0.5   # Neutral if no assessment taken

    return (
        0.30 * base
        + 0.25 * endorse_factor
        + 0.25 * duration_factor
        + 0.20 * assess_factor
    )


# ============================================================================
# C. Experience Depth  (0 → 15 pts)
# ============================================================================

def _score_experience_depth(candidate: dict) -> float:
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []

    yoe = profile.get("years_of_experience", 0)
    if yoe is None:
        yoe = 0

    pts = 0.0

    # --- Years-in-band bonus (0-5 pts) ---
    if IDEAL_EXPERIENCE_MIN <= yoe <= IDEAL_EXPERIENCE_MAX:
        pts += 5.0
    elif 3.0 <= yoe < IDEAL_EXPERIENCE_MIN:
        pts += 3.0
    elif IDEAL_EXPERIENCE_MAX < yoe <= 15:
        pts += 3.0
    else:
        pts += 1.0

    # --- Production signals in career descriptions (0-7 pts) ---
    production_keywords = {
        "production", "deployed", "shipped", "users", "scale",
        "ranking", "retrieval", "recommendation", "search",
        "pipeline", "real-time", "latency", "throughput",
        "a/b test", "metrics", "monitoring", "serving",
        "million", "inference", "api",
    }
    research_only_keywords = {
        "academic", "thesis", "publication", "paper",
        "theoretical", "coursework",
    }

    prod_hits = 0
    research_only_hits = 0

    for role in career:
        desc = (role.get("description") or "").lower()
        for kw in production_keywords:
            if kw in desc:
                prod_hits += 1
        for kw in research_only_keywords:
            if kw in desc:
                research_only_hits += 1

    pts += min(prod_hits * 0.7, 7.0)

    # Penalty for research-only language with no production
    if research_only_hits > 0 and prod_hits == 0:
        pts -= 2.0

    # --- ML/AI specific role time (0-3 pts) ---
    ml_months = 0
    for role in career:
        title = (role.get("title") or "").lower()
        if any(kw in title for kw in ("ml", "ai", "machine learning",
                                       "data scien", "nlp", "research")):
            ml_months += role.get("duration_months", 0)

    pts += min(ml_months / 48 * 3, 3.0)

    return clamp(pts, 0, EXPERIENCE_DEPTH_MAX_PTS)


# ============================================================================
# D. Education & Certifications  (0 → 10 pts)
# ============================================================================

def _score_education(candidate: dict) -> float:
    education = candidate.get("education") or []
    certifications = candidate.get("certifications") or []

    pts = 0.0

    # --- Relevant field of study (0-5 pts) ---
    relevant_fields = {
        "computer science", "machine learning", "artificial intelligence",
        "data science", "statistics", "mathematics", "math",
        "information technology", "software engineering",
        "electrical engineering", "electronics",
    }
    for edu in education:
        field = (edu.get("field_of_study") or "").lower()
        if any(rf in field for rf in relevant_fields):
            pts += 3.0
            break  # Only count once

    # --- Institution tier (0-2 pts) ---
    best_tier = 5
    for edu in education:
        tier_str = edu.get("tier", "unknown")
        tier_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
        tier = tier_map.get(tier_str, 5)
        best_tier = min(best_tier, tier)

    if best_tier <= 1:
        pts += 2.0
    elif best_tier == 2:
        pts += 1.5
    elif best_tier == 3:
        pts += 0.5

    # --- Advanced degree (0-1 pt) ---
    for edu in education:
        degree = (edu.get("degree") or "").lower()
        if any(d in degree for d in ("m.tech", "m.sc", "m.e.", "ms", "mba",
                                      "ph.d", "phd")):
            pts += 1.0
            break

    # --- Relevant certifications (0-2 pts) ---
    ml_cert_keywords = {
        "machine learning", "ml", "ai", "deep learning",
        "data science", "nlp", "tensorflow", "pytorch",
        "aws machine learning", "gcp machine learning",
        "professional data engineer",
    }
    cert_pts = 0
    for cert in certifications:
        cert_name = (cert.get("name") or "").lower()
        if any(kw in cert_name for kw in ml_cert_keywords):
            cert_pts += 1.0
    pts += min(cert_pts, 2.0)

    return clamp(pts, 0, EDUCATION_MAX_PTS)


# ============================================================================
# E. Location & Logistics Fit  (0 → 10 pts)
# ============================================================================

def _score_location(candidate: dict) -> float:
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}

    country = (profile.get("country") or "").lower().strip()
    location = (profile.get("location") or "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    if willing_to_relocate is None:
        willing_to_relocate = False
    work_mode = (signals.get("preferred_work_mode") or "").lower()
    notice_days = signals.get("notice_period_days")
    if notice_days is None:
        notice_days = 90

    pts = 0.0

    # --- Location match (0-6 pts) ---
    loc_tokens = set(location.replace(",", " ").split())
    is_preferred = bool(loc_tokens & PREFERRED_LOCATIONS)
    is_acceptable = bool(loc_tokens & ACCEPTABLE_INDIAN_CITIES)
    is_india = (country == INDIA_COUNTRY)

    if is_preferred:
        pts += 6.0
    elif is_acceptable:
        pts += 4.5
    elif is_india:
        pts += 3.0
    elif willing_to_relocate:
        pts += 1.5
    # else: 0

    # --- Work mode compatibility (0-2 pts) ---
    # JD says "hybrid — flexible cadence"
    if work_mode in ("hybrid", "flexible"):
        pts += 2.0
    elif work_mode == "remote":
        pts += 1.0
    elif work_mode == "onsite":
        pts += 1.5

    # --- Notice period (0-2 pts) ---
    if notice_days <= IDEAL_NOTICE_DAYS:
        pts += 2.0
    elif notice_days <= MAX_ACCEPTABLE_NOTICE_DAYS:
        pts += 1.0
    # else: 0

    return clamp(pts, 0, LOCATION_MAX_PTS)


# ============================================================================
# F. Career Pattern Penalties  (0 → -10 pts)
# ============================================================================

def _score_career_penalties(candidate: dict) -> float:
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    penalty = 0.0

    # --- Consulting-only career (−8 pts) ---
    if _is_consulting_only(career):
        penalty -= 8.0

    # --- Title-description mismatch / keyword stuffer (−5 pts) ---
    if _is_keyword_stuffer(profile, career, skills):
        penalty -= 5.0

    # --- Job-hopping pattern (−3 pts) ---
    if _is_job_hopper(career):
        penalty -= 3.0

    return max(penalty, CAREER_PATTERN_PENALTY_MAX)


def _is_consulting_only(career: list) -> bool:
    """True if EVERY role in career history is at a known consulting firm."""
    if not career:
        return False
    return all(
        (role.get("company") or "").lower().strip() in CONSULTING_FIRMS
        for role in career
    )


def _is_keyword_stuffer(profile: dict, career: list, skills: list) -> bool:
    """
    Detect profiles where the title is clearly non-technical but the skills
    list is packed with AI/ML keywords — the exact trap the JD warns about.
    """
    title = (profile.get("current_title") or "").lower().strip()

    # Only applies to clearly non-tech titles
    if title not in NON_TECH_TITLES:
        return False

    # Count core AI skills
    skill_names = {(s.get("name") or "").lower() for s in skills}
    core_matches = len(skill_names & CORE_AI_SKILLS)

    # Check career descriptions for any AI/ML work
    ai_desc_keywords = {
        "machine learning", "deep learning", "nlp", "embedding",
        "model", "neural", "training", "inference", "ranking",
        "retrieval", "recommendation", "pytorch", "tensorflow",
    }
    all_descs = " ".join(
        (r.get("description") or "").lower() for r in career
    )
    has_ai_in_desc = any(kw in all_descs for kw in ai_desc_keywords)

    # Non-tech title + lots of AI skills + no AI in descriptions = stuffer
    if core_matches >= 3 and not has_ai_in_desc:
        return True

    return False


def _is_job_hopper(career: list) -> bool:
    """
    Detect title-chasing pattern: 3+ short stints (< 18 months each)
    with different companies.
    """
    if len(career) < 3:
        return False

    short_stints = 0
    companies = set()
    for role in career:
        dur = role.get("duration_months", 0)
        company = (role.get("company") or "").lower()
        if dur and dur < 18 and company not in companies:
            short_stints += 1
        companies.add(company)

    return short_stints >= 3
