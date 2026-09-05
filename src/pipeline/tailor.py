"""Resume tailoring module with batched project framing and profile localization."""

from __future__ import annotations

from collections import defaultdict, deque
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
    TailoredNonProjectContent,
    TechnicalSkills,
)
from src.utils.exit_codes import ExitCode
from src.utils.logger import SessionLogger, log


def _build_market_weights(market_analysis: MarketAnalysisResult) -> dict[str, float]:
    weights: dict[str, float] = {}
    for skill in market_analysis.top_skills:
        clean_name = skill.name.lower().strip()
        if clean_name:
            weights[clean_name] = max(weights.get(clean_name, 0.0), skill.share_percentage)
    return weights


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _reorder_skill_list(skills: list[Any], market_weights: dict[str, float]) -> list[str]:
    cleaned = [str(s).strip() for s in (skills or []) if s is not None and str(s).strip()]
    return sorted(
        cleaned,
        key=lambda skill: market_weights.get(skill.lower(), 0.0),
        reverse=True,
    )


def resolve_profile_photo(
    explicit_path: str | None,
    project_root: Path | None = None,
    profile_dir: Path | None = None,
) -> str | None:
    """Locates candidate profile picture from explicit path or standard assets directory.

    Args:
        explicit_path: Path specified in profile personal_info, if any.
        project_root: Root directory to search for default image assets.
        profile_dir: Directory containing candidate profile, used to resolve relative paths.

    Returns:
        Resolved absolute string path to the image, or None if not found.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return str(p.resolve())
        if profile_dir and (profile_dir / explicit_path).is_file():
            return str((profile_dir / explicit_path).resolve())
        return None

    search_roots = [project_root] if project_root is not None else [Path("."), PROJECT_ROOT]
    for root in search_roots:
        assets_dir = root / "assets"
        if assets_dir.is_dir():
            for filename in (
                "profile.jpg",
                "profile.png",
                "photo.jpg",
                "photo.png",
                "avatar.jpg",
                "avatar.png",
            ):
                candidate = assets_dir / filename
                if candidate.is_file():
                    return str(candidate.resolve())
    return None


MAX_PROJECTS_DEFAULT = 3
MAX_EXPERIENCE = 3
MAX_LEADERSHIP = 2
MAX_CERTIFICATIONS = 3


def compute_dynamic_project_limit(
    candidate_tier: str,
    num_experience: int,
    num_leadership: int = 0,
    num_certs: int = 0,
    available_projects: int = 0,
) -> int:
    """Dynamically calculates maximum project allocation based on page budget and available profile sections."""
    if num_experience == 0:
        # When candidate has no professional experience, projects are the primary demonstration of competency.
        # If leadership or certifications are present, allocate 4 projects; otherwise up to 5 projects.
        target = 4 if (num_leadership > 0 or num_certs > 0) else 5
    elif num_experience == 1:
        target = 4
    else:
        # 2 or more experience entries require significant vertical space.
        target = 3

    if available_projects > 0:
        target = min(target, available_projects)

    return max(1, target)


def _score_project_alignment(
    project_dict: dict[str, Any],
    market_weights: dict[str, float],
    target_role: str = "",
) -> float:
    """Scores project suitability for the target role, live market demand, and technical depth."""
    score = 0.0
    raw_stack = project_dict.get("stack") or []
    stack = [str(tech).strip() for tech in raw_stack if tech is not None and str(tech).strip()]
    if stack:
        score += sum(market_weights.get(tech.lower(), 0.0) for tech in stack)

    if target_role:
        role_tokens = {
            t.lower().strip()
            for t in target_role.replace("/", " ").replace("-", " ").split()
            if len(t.strip()) > 2
        }
        name = str(project_dict.get("name") or "").lower()
        desc = str(project_dict.get("description") or "").lower()
        stack_str = " ".join(stack).lower()
        search_text = f"{name} {desc} {stack_str}"

        for token in role_tokens:
            if token in search_text:
                score += 15.0

    raw_highlights = project_dict.get("highlights") or []
    highlights = [str(h) for h in raw_highlights if h is not None]
    score += min(len(highlights) * 2.0, 10.0)
    for h in highlights:
        if any(c.isdigit() for c in h):
            score += 3.0

    return score


def _normalize_lookup_key(value: Any) -> str:
    return str(value or "").lower().strip()


def _parse_team_size(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe_nonempty(items: list[Any] | None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if item is None:
            continue
        clean = str(item).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _build_project_bullets(
    description: str,
    llm_bullets: list[str] | None,
    raw_highlights: list[str],
) -> list[str]:
    """Enforce project bullet contract: #1 from description, #2-#3 from highlights."""
    clean_description = description.strip()
    clean_highlights = _dedupe_nonempty(raw_highlights)
    clean_llm = _dedupe_nonempty(llm_bullets or [])

    # Bullet #1 is mandatory and must be based on project description.
    if clean_llm:
        first_bullet = clean_llm[0]
    elif clean_description:
        first_bullet = clean_description
    elif clean_highlights:
        first_bullet = clean_highlights[0]
    else:
        return []

    remaining_llm = [b for b in clean_llm[1:] if b.casefold() != first_bullet.casefold()]
    highlight_candidates = [b for b in clean_highlights if b.casefold() != first_bullet.casefold()]

    second_third: list[str] = []
    for bullet in remaining_llm:
        if len(second_third) >= 2:
            break
        second_third.append(bullet)
    for bullet in highlight_candidates:
        if len(second_third) >= 2:
            break
        if bullet.casefold() not in {b.casefold() for b in second_third}:
            second_third.append(bullet)

    return [first_bullet, *second_third]


def run_tailor(
    profile_input: str | Path | dict[str, Any],
    market_analysis: MarketAnalysisResult,
    target_role: str | RoleMatchResult,
    engine: GeminiEngine | None = None,
    language: str = "en",
) -> ResumeData:
    """Reconciles candidate profile achievements with market competency weights and formats sections.

    Args:
        profile_input: File path, YAML string, or dictionary representing the candidate profile.
        market_analysis: Aggregated market intelligence result from Stage 3.
        target_role: Target job title string or RoleMatchResult from Stage 1.
        engine: Optional pre-configured GeminiEngine instance.
        language: Target output language code ('en' or 'tr').

    Returns:
        ResumeData validated schema tailored and ready for compilation.
    """
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

    raw_skills = raw_dict.get("skills") or {}
    tailored_skills = TechnicalSkills(
        languages=_reorder_skill_list(raw_skills.get("languages") or [], market_weights),
        frameworks=_reorder_skill_list(raw_skills.get("frameworks") or [], market_weights),
        tools_and_systems=_reorder_skill_list(
            raw_skills.get("tools_and_systems") or [], market_weights
        ),
        databases=_reorder_skill_list(raw_skills.get("databases") or [], market_weights),
    )

    personal = raw_dict.get("personal_info") or {}
    raw_exp = raw_dict.get("experience") or []
    if len(raw_exp) > MAX_EXPERIENCE:
        log.info(
            f"Candidate profile includes {len(raw_exp)} experience entries; selecting top "
            f"{MAX_EXPERIENCE} most recent/relevant entries to guarantee 1-page fit."
        )
        selected_exp = raw_exp[:MAX_EXPERIENCE]
    else:
        selected_exp = raw_exp

    raw_lead = raw_dict.get("leadership") or []
    if len(raw_lead) > MAX_LEADERSHIP:
        log.info(
            f"Candidate profile includes {len(raw_lead)} leadership entries; selecting top "
            f"{MAX_LEADERSHIP} entries to guarantee 1-page fit."
        )
        selected_lead = raw_lead[:MAX_LEADERSHIP]
    else:
        selected_lead = raw_lead

    raw_certs = raw_dict.get("certifications") or []
    if len(raw_certs) > MAX_CERTIFICATIONS:
        log.info(
            f"Candidate profile includes {len(raw_certs)} certifications; selecting top "
            f"{MAX_CERTIFICATIONS} entries to guarantee 1-page fit."
        )
        selected_certs = raw_certs[:MAX_CERTIFICATIONS]
    else:
        selected_certs = raw_certs

    if isinstance(target_role, RoleMatchResult):
        candidate_tier = target_role.candidate_tier
    else:
        profile_tier = _normalize_lookup_key(raw_dict.get("candidate_tier"))
        candidate_tier = profile_tier if profile_tier in {"student", "junior"} else "student"

    raw_projects: list[dict[str, Any]] = raw_dict.get("projects") or []
    sorted_projects = sorted(
        raw_projects,
        key=lambda p: _score_project_alignment(p, market_weights, resolved_role),
        reverse=True,
    )

    dynamic_max_projects = compute_dynamic_project_limit(
        candidate_tier=candidate_tier,
        num_experience=len(selected_exp),
        num_leadership=len(selected_lead),
        num_certs=len(selected_certs),
        available_projects=len(sorted_projects),
    )

    if len(sorted_projects) > dynamic_max_projects:
        log.info(
            f"Candidate profile includes {len(sorted_projects)} projects; dynamic allocation selects "
            f"top {dynamic_max_projects} projects for '{resolved_role}' "
            f"(experience count: {len(selected_exp)}, tier: {candidate_tier}) to guarantee optimal 1-page fit."
        )
        selected_projects = sorted_projects[:dynamic_max_projects]
    else:
        selected_projects = sorted_projects

    projects_for_batch: list[dict[str, Any]] = []
    project_indices_to_batch: list[int] = []
    batch_position_by_project_index: dict[int, int] = {}

    for idx, proj in enumerate(selected_projects):
        name = str(proj.get("name") or "")
        desc = str(proj.get("description") or "")
        highlights = [str(h) for h in (proj.get("highlights") or []) if h is not None]
        words = len(desc.split()) + sum(len(h.split()) for h in highlights)

        if words >= 15:
            prioritized_stack = _reorder_skill_list(proj.get("stack") or [], market_weights)
            batch_position = len(projects_for_batch)
            projects_for_batch.append(
                {
                    "name": name,
                    "stack": prioritized_stack,
                    "team_size": _parse_team_size(proj.get("team_size")),
                    "description": desc,
                    "highlights": highlights,
                }
            )
            project_indices_to_batch.append(idx)
            batch_position_by_project_index[idx] = batch_position

    tailored_bullets_by_name: dict[str, deque[list[str]]] = defaultdict(deque)
    tailored_bullets_by_batch_position: dict[int, list[str]] = {}
    if projects_for_batch:
        log.info(f"Tailoring {len(projects_for_batch)} projects in a single Gemini batch prompt...")
        try:
            batch_res: BatchTailoredProjectsResult = active_engine.tailor_projects_batch(
                projects_data=projects_for_batch,
                top_skills=top_keywords,
                target_role=resolved_role,
                language=language,
                max_projects=dynamic_max_projects,
            )
            for batch_pos, item in enumerate(batch_res.projects):
                bullets = [b.text.strip() for b in item.bullets if b.text and b.text.strip()]
                if bullets:
                    tailored_bullets_by_batch_position[batch_pos] = bullets
                    clean_name = _normalize_lookup_key(item.project_name)
                    if clean_name:
                        tailored_bullets_by_name[clean_name].append(bullets)
                    log.debug(
                        f"Project '{item.project_name}' tailored: {len(bullets)} bullets generated"
                    )
        except Exception as exc:
            log.warning(f"Batch project tailoring failed, falling back to raw highlights: {exc}")
    tailored_projects: list[ProjectItem] = []
    for idx, proj in enumerate(selected_projects):
        name = proj.get("name", "")
        clean_name = _normalize_lookup_key(name)
        stack = _reorder_skill_list(proj.get("stack", []), market_weights)
        link = proj.get("link", "")
        raw_highlights = proj.get("highlights", [])

        final_bullets: list[str] | None = None
        if clean_name and tailored_bullets_by_name.get(clean_name):
            final_bullets = tailored_bullets_by_name[clean_name].popleft()
        elif idx in batch_position_by_project_index:
            batch_pos = batch_position_by_project_index[idx]
            final_bullets = tailored_bullets_by_batch_position.get(batch_pos)

        project_description = str(proj.get("description", "") or "").strip()
        resolved_project_bullets = _build_project_bullets(
            description=project_description,
            llm_bullets=final_bullets,
            raw_highlights=raw_highlights,
        )

        tailored_projects.append(
            ProjectItem(
                name=name,
                stack=stack,
                team_size=_parse_team_size(proj.get("team_size")),
                link=link,
                description=project_description,
                bullets=resolved_project_bullets,
            )
        )
    # Tailor summary, experience, and leadership sections
    final_summary = personal.get("summary") or ""
    sec_res: TailoredNonProjectContent | None = None
    exp_bullets_map: dict[str, deque[list[str]]] = defaultdict(deque)
    lead_bullets_map: dict[str, deque[list[str]]] = defaultdict(deque)

    try:
        log.info(f"Tailoring summary, experience, and leadership into {language.upper()}...")
        sec_res = active_engine.tailor_profile_sections(
            summary=final_summary,
            experience=selected_exp,
            leadership=selected_lead,
            target_role=resolved_role,
            top_skills=top_keywords,
            language=language,
        )
        if sec_res.summary:
            final_summary = sec_res.summary
        for item in sec_res.experience:
            if item.bullets:
                exp_bullets_map[_normalize_lookup_key(item.title)].append(item.bullets)
        for item in sec_res.leadership:
            if item.bullets:
                lead_bullets_map[_normalize_lookup_key(item.title)].append(item.bullets)
    except Exception as exc:
        log.warning(f"Section tailoring failed, preserving source items: {exc}")

    tailored_experience: list[ExperienceItem] = []
    for idx, exp in enumerate(selected_exp):
        role = str(exp.get("role") or "")
        role_key = _normalize_lookup_key(role)
        bullets: list[str] | None = None
        if role_key and exp_bullets_map.get(role_key):
            bullets = exp_bullets_map[role_key].popleft()
        elif sec_res and idx < len(sec_res.experience):
            bullets = sec_res.experience[idx].bullets
        resolved_bullets = bullets or exp.get("bullets") or []

        tailored_experience.append(
            ExperienceItem(
                role=role,
                company=str(exp.get("company") or ""),
                location=str(exp.get("location") or ""),
                employment_type=str(exp.get("employment_type") or ""),
                start_date=str(exp.get("start_date") or ""),
                end_date=str(exp.get("end_date") or ""),
                bullets=resolved_bullets,
            )
        )

    tailored_leadership: list[LeadershipItem] = []
    for idx, lead in enumerate(selected_lead):
        role = str(lead.get("role") or "")
        role_key = _normalize_lookup_key(role)
        bullets: list[str] | None = None
        if role_key and lead_bullets_map.get(role_key):
            bullets = lead_bullets_map[role_key].popleft()
        elif sec_res and idx < len(sec_res.leadership):
            bullets = sec_res.leadership[idx].bullets
        resolved_bullets = bullets or lead.get("bullets") or []

        tailored_leadership.append(
            LeadershipItem(
                role=role,
                organization=str(lead.get("organization") or ""),
                date=str(lead.get("date") or ""),
                link=str(lead.get("link") or ""),
                bullets=resolved_bullets,
            )
        )

    def _parse_items(model: type[BaseModel], raw: list[Any]) -> list[Any]:
        items = []
        for entry in raw:
            try:
                items.append(model.model_validate(entry))
            except Exception:  # noqa: S112
                continue
        return items

    profile_dir = profile_input.parent if isinstance(profile_input, Path) else None
    photo: str | None = resolve_profile_photo(personal.get("photo_path"), profile_dir=profile_dir)

    resolved_title = personal.get("title") or ""
    if not resolved_title:
        resolved_title = resolved_role if candidate_tier == "junior" else f"{resolved_role} Intern"

    resume_data = ResumeData(
        name=personal.get("name") or "",
        title=resolved_title,
        location=personal.get("location") or "",
        email=personal.get("email") or "",
        phone=personal.get("phone") or "",
        linkedin_url=personal.get("linkedin_url") or "",
        github_url=personal.get("github_url") or "",
        portfolio_url=personal.get("portfolio_url") or "",
        photo_path=photo or "",
        summary=final_summary,
        education=_parse_items(EducationItem, raw_dict.get("education") or []),
        skills=tailored_skills,
        languages=_parse_items(SpokenLanguageItem, raw_dict.get("languages") or []),
        projects=tailored_projects[:dynamic_max_projects],
        experience=tailored_experience[:MAX_EXPERIENCE],
        candidate_tier=candidate_tier,
        leadership=tailored_leadership[:MAX_LEADERSHIP],
        certifications=_parse_items(CertificationItem, selected_certs)[:MAX_CERTIFICATIONS],
        hobbies=raw_dict.get("hobbies") or [],
    )

    return resume_data
