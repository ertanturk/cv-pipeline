"""Resume tailoring module with batched Google XYZ project framing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from src.llm import GeminiEngine
from src.models import (
    BatchTailoredProjectsResult,
    CertificationItem,
    EducationItem,
    ExperienceItem,
    LeadershipItem,
    MarketAnalysisResult,
    ProjectItem,
    ResumeData,
    RoleMatchResult,
    SpokenLanguageItem,
    TechnicalSkills,
)
from src.utils.exit_codes import ExitCode
from src.utils.logger import SessionLogger, log


def _build_market_weights(market_analysis: MarketAnalysisResult) -> dict[str, float]:
    """Maps lowercase canonical skill and tool names to their calculated market share."""
    weights: dict[str, float] = {}
    for skill in market_analysis.top_skills:
        clean_name = skill.name.lower().strip()
        if clean_name:
            weights[clean_name] = max(weights.get(clean_name, 0.0), skill.share_percentage)
    return weights


def _reorder_skill_list(skills: list[str], market_weights: dict[str, float]) -> list[str]:
    """Sorts candidate skills descending by market share without altering contents."""
    return sorted(
        skills,
        key=lambda skill: market_weights.get(skill.lower().strip(), 0.0),
        reverse=True,
    )


def _score_project_alignment(
    project_dict: dict[str, Any], market_weights: dict[str, float]
) -> float:
    """Calculates project alignment score based on candidate tech stack market overlap."""
    stack: list[str] = project_dict.get("stack", [])
    if not stack:
        return 0.0
    return sum(market_weights.get(tech.lower().strip(), 0.0) for tech in stack)


def run_tailor(
    profile_input: str | Path | dict[str, Any],
    market_analysis: MarketAnalysisResult,
    target_role: str | RoleMatchResult,
    engine: GeminiEngine | None = None,
    language: str = "en",
) -> ResumeData:
    """Reconciles candidate profile with market signals using batched accomplishment generation."""
    if isinstance(profile_input, Path):
        if not profile_input.exists():
            SessionLogger.fatal(
                f"Profile not found at: {profile_input}", code=ExitCode.CONFIG_ERROR
            )
        raw_dict = yaml.safe_load(profile_input.read_text(encoding="utf-8"))
    elif isinstance(profile_input, str):
        raw_dict = yaml.safe_load(profile_input)
    elif isinstance(profile_input, dict):
        raw_dict = profile_input
    else:
        SessionLogger.fatal(
            f"Unsupported profile type: {type(profile_input)}", code=ExitCode.CONFIG_ERROR
        )

    if not isinstance(raw_dict, dict):
        SessionLogger.fatal("Profile failed to parse into dictionary.", code=ExitCode.CONFIG_ERROR)

    if isinstance(target_role, RoleMatchResult):
        if not target_role.primary_role or not target_role.primary_role.title:
            SessionLogger.fatal(
                "RoleMatchResult lacks a valid primary role title.", code=ExitCode.CONFIG_ERROR
            )
        resolved_role = target_role.primary_role.title
    else:
        resolved_role = str(target_role).strip()

    log.info(
        f"Initiating Stage 4: Resume Tailoring for '{resolved_role}' "
        f"({len(market_analysis.top_skills)} market signals evaluated)"
    )

    active_engine: GeminiEngine = engine or GeminiEngine()
    market_weights = _build_market_weights(market_analysis)
    top_keywords = market_analysis.recommended_keywords or [
        s.name for s in market_analysis.top_skills[:10]
    ]

    # Prioritize Technical Skills (Profile Truth ∩ Market Demand)
    raw_skills = raw_dict.get("skills", {})
    tailored_skills = TechnicalSkills(
        languages=_reorder_skill_list(raw_skills.get("languages", []), market_weights),
        frameworks=_reorder_skill_list(raw_skills.get("frameworks", []), market_weights),
        tools_and_systems=_reorder_skill_list(
            raw_skills.get("tools_and_systems", []), market_weights
        ),
        databases=_reorder_skill_list(raw_skills.get("databases", []), market_weights),
    )

    # Rank Projects by Market Saturation
    raw_projects: list[dict[str, Any]] = raw_dict.get("projects", [])
    sorted_projects = sorted(
        raw_projects,
        key=lambda p: _score_project_alignment(p, market_weights),
        reverse=True,
    )

    # Prepare Batch Payload for Projects
    projects_for_batch: list[dict[str, Any]] = []
    project_indices_to_batch: list[int] = []

    for idx, proj in enumerate(sorted_projects):
        name = proj.get("name", "")
        desc = proj.get("description", "")
        highlights = proj.get("highlights", [])
        words = len(desc.split()) + sum(len(h.split()) for h in highlights)

        if words >= 15:
            prioritized_stack = _reorder_skill_list(proj.get("stack", []), market_weights)
            projects_for_batch.append(
                {
                    "name": name,
                    "stack": prioritized_stack,
                    "description": desc,
                    "highlights": highlights,
                }
            )
            project_indices_to_batch.append(idx)

    # Execute Batched Tailoring
    tailored_bullets_by_name: dict[str, list[str]] = {}
    if projects_for_batch:
        log.info(f"Tailoring {len(projects_for_batch)} projects in a single Google XYZ prompt...")
        try:
            if hasattr(active_engine, "tailor_projects_batch"):
                batch_res: BatchTailoredProjectsResult = active_engine.tailor_projects_batch(
                    projects_data=projects_for_batch,
                    top_skills=top_keywords,
                    target_role=resolved_role,
                    language=language,
                )
                for item in batch_res.projects:
                    bullets = [b.text.strip() for b in item.bullets if b.text and b.text.strip()]
                    if bullets:
                        tailored_bullets_by_name[item.project_name.lower().strip()] = bullets
        except Exception as exc:
            log.warning(f"Batch project tailoring failed, falling back to raw highlights: {exc}")

    # Assemble Tailored Project Items
    tailored_projects: list[ProjectItem] = []
    for idx, proj in enumerate(sorted_projects):
        name = proj.get("name", "")
        clean_name = name.lower().strip()
        stack = _reorder_skill_list(proj.get("stack", []), market_weights)
        link = proj.get("link", "")
        raw_highlights = proj.get("highlights", [])

        # Fetch batch-generated bullets or fallback to raw
        final_bullets = tailored_bullets_by_name.get(clean_name)
        if not final_bullets and idx in project_indices_to_batch:
            # Positional match fallback if model slightly altered project name
            pos = project_indices_to_batch.index(idx)
            if pos < len(tailored_bullets_by_name):
                final_bullets = list(tailored_bullets_by_name.values())[pos]

        tailored_projects.append(
            ProjectItem(
                name=name,
                stack=stack,
                link=link,
                bullets=final_bullets or raw_highlights,
            )
        )

    # Parse Static Sections
    personal = raw_dict.get("personal_info", {})

    def _parse_items(model: type[BaseModel], raw: list[Any]) -> list[Any]:
        items = []
        for entry in raw:
            try:
                items.append(model.model_validate(entry))
            except Exception:  # noqa: S112
                continue
        return items

    resume_data = ResumeData(
        name=personal.get("name", ""),
        title=resolved_role,
        location=personal.get("location", ""),
        email=personal.get("email", ""),
        phone=personal.get("phone", ""),
        linkedin_url=personal.get("linkedin_url", ""),
        github_url=personal.get("github_url", ""),
        portfolio_url=personal.get("portfolio_url") or "",
        photo_path=personal.get("photo_path") or "",
        summary=personal.get("summary") or "",
        education=_parse_items(EducationItem, raw_dict.get("education", [])),
        skills=tailored_skills,
        languages=_parse_items(SpokenLanguageItem, raw_dict.get("languages", [])),
        projects=tailored_projects,
        experience=_parse_items(ExperienceItem, raw_dict.get("experience", [])),
        leadership=_parse_items(LeadershipItem, raw_dict.get("leadership", [])),
        certifications=_parse_items(CertificationItem, raw_dict.get("certifications", [])),
    )

    log.info(
        f"Generated ResumeData with {len(tailored_projects)} tailored projects "
        f"and prioritized technical skills."
    )
    return resume_data
