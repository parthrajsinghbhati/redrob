"""
Redrob AI Candidate Ranking — Reasoning Generator
====================================================
Produces 1-2 sentence justifications for each ranked candidate.

Requirements:
  • Reference specific facts from the candidate's profile
  • Connect to specific JD requirements
  • Acknowledge gaps honestly
  • Use 5-6 different sentence structures (not templated)
  • Tone must match the rank position
"""

import random

from config import (
    CORE_AI_SKILLS,
    NICE_TO_HAVE_SKILLS,
    CONSULTING_FIRMS,
    PREFERRED_LOCATIONS,
    ACCEPTABLE_INDIAN_CITIES,
)
from utils import safe_get_signal


# ============================================================================
# Public API
# ============================================================================

def generate_reasoning(candidate: dict, rank: int, scores: dict) -> str:
    """
    Generate a 1-2 sentence reasoning string for a ranked candidate.

    Parameters
    ----------
    candidate : dict – full candidate record
    rank : int – rank position (1 = best)
    scores : dict – sub-score breakdown (semantic, rule sub-scores, behavioral)
    """
    profile = candidate.get("profile") or {}
    career = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    signals = candidate.get("redrob_signals") or {}

    # --- Extract concrete facts ---
    name = profile.get("anonymized_name", "Candidate")
    title = profile.get("current_title", "Professional")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")

    core_skills = _get_matching_skills(skills, CORE_AI_SKILLS)
    nice_skills = _get_matching_skills(skills, NICE_TO_HAVE_SKILLS)
    all_relevant = core_skills + nice_skills

    rr = signals.get("recruiter_response_rate", 0)
    gh = signals.get("github_activity_score", -1)
    notice = signals.get("notice_period_days", 90)
    otw = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "")

    # Product company experience
    product_companies = [
        r.get("company") for r in career
        if (r.get("company") or "").lower().strip() not in CONSULTING_FIRMS
    ]

    # --- Build strengths and gaps ---
    strengths = []
    gaps = []

    # Title fit (avoid repeating the title verbatim — templates already name it)
    if scores.get("title_career", 0) >= 15:
        strengths.append("title aligns closely with the role")
    elif scores.get("title_career", 0) >= 8:
        strengths.append(f"relevant technical background as {title}")
    else:
        gaps.append(f"current role as {title} is not directly AI/ML")

    # Experience
    if 5 <= yoe <= 9:
        strengths.append(f"{yoe:.1f} years experience in the ideal range")
    elif 3 <= yoe < 5:
        strengths.append(f"{yoe:.1f} years experience (slightly below preferred range)")
    elif yoe > 9:
        strengths.append(f"{yoe:.1f} years of deep experience")

    # Skills
    if len(core_skills) >= 3:
        top_3 = ", ".join(core_skills[:3])
        strengths.append(f"core skills include {top_3}")
    elif len(all_relevant) >= 2:
        top_2 = ", ".join(all_relevant[:2])
        strengths.append(f"relevant skills: {top_2}")
    else:
        gaps.append("limited overlap with required AI/ML core skills")

    # Product company experience
    if product_companies:
        strengths.append(f"product-company experience at {product_companies[0]}")

    # Behavioral signals
    if rr >= 0.6:
        strengths.append(f"high recruiter response rate ({rr:.0%})")
    elif rr < 0.2:
        gaps.append(f"low recruiter response rate ({rr:.0%})")

    if gh >= 40:
        strengths.append(f"active GitHub contributor (score: {gh:.0f})")

    if notice <= 30:
        strengths.append(f"{notice}-day notice period")
    elif notice > 90:
        gaps.append(f"long notice period ({notice} days)")

    # Location
    loc_lower = (location or "").lower()
    if any(p in loc_lower for p in PREFERRED_LOCATIONS):
        strengths.append(f"based in {location}")
    elif country and country.lower() != "india":
        gaps.append(f"located outside India ({country})")

    # --- Select template based on rank tier ---
    return _format_reasoning(rank, strengths, gaps, title, company, yoe)


# ============================================================================
# Template Selection  (6 distinct structures)
# ============================================================================

def _format_reasoning(
    rank: int,
    strengths: list[str],
    gaps: list[str],
    title: str,
    company: str,
    yoe: float,
) -> str:
    """Choose one of 6 templates and fill with actual data."""

    top_strengths = strengths[:3] if strengths else ["relevant profile"]
    top_gaps = gaps[:2] if gaps else []

    s_text = "; ".join(top_strengths)
    g_text = "; ".join(top_gaps) if top_gaps else ""

    # ---- Template 1: Lead with role, then strengths, then gaps ----
    if rank <= 15 and len(strengths) >= 2:
        base = f"{title} at {company} ({yoe:.1f} yrs) — {s_text}."
        if g_text:
            base += f" Minor concern: {g_text}."
        return base

    # ---- Template 2: Strengths-first, acknowledging gaps ----
    if rank <= 30:
        base = f"Strong fit: {s_text}."
        if g_text:
            base += f" Gap: {g_text}."
        return base

    # ---- Template 3: Balanced assessment ----
    if rank <= 50:
        if g_text:
            return (
                f"{title} with {yoe:.1f} years: {top_strengths[0]}. "
                f"However, {g_text}."
            )
        return f"{title} with {yoe:.1f} years — {s_text}."

    # ---- Template 4: Gap-forward with positives ----
    if rank <= 70:
        if g_text:
            return (
                f"Partial fit: {g_text}. "
                f"On the positive side, {top_strengths[0] if top_strengths else 'some relevant background'}."
            )
        return f"Moderate match as {title} at {company}: {s_text}."

    # ---- Template 5: Concise ranked assessment ----
    if rank <= 90:
        core = top_strengths[0] if top_strengths else "limited direct relevance"
        concern = top_gaps[0] if top_gaps else "no major red flags"
        return f"{title} ({yoe:.1f} yrs): {core}; {concern}."

    # ---- Template 6: Brief lower-rank summary ----
    core = top_strengths[0] if top_strengths else "some transferable experience"
    if g_text:
        return f"{title} ({yoe:.1f} yrs) ranked for {core}, though {g_text}."
    return f"{title} ({yoe:.1f} yrs) included for {core}."


# ============================================================================
# Helpers
# ============================================================================

def _get_matching_skills(
    skills: list[dict],
    reference_set: set[str],
) -> list[str]:
    """Return skill names (original case) that match the reference set."""
    matches = []
    for s in skills:
        name = s.get("name", "")
        if name.lower().strip() in reference_set:
            matches.append(name)
    return matches
