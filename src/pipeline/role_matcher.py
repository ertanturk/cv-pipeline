"""Analyzes candidate profile data and matches optimal professional roles."""

from pathlib import Path

from src.llm import GeminiEngine
from src.models import RoleMatchResult
from src.utils.exit_codes import ExitCode
from src.utils.logger import SessionLogger, log


def run_role_matcher(
    profile_input: str | Path,
    engine: GeminiEngine | None = None,
) -> RoleMatchResult:
    """Evaluates raw candidate profile data and returns structured role recommendations.

    Args:
        profile_input: Raw YAML/JSON string content or Path to profile file.
        engine: Optional pre-configured GeminiEngine instance.

    Returns:
        RoleMatchResult containing primary role, search queries, and alternatives.
    """
    # Resolve profile input
    if isinstance(profile_input, Path):
        if not profile_input.exists():
            SessionLogger.fatal(
                f"Candidate profile file not found at: {profile_input}",
                code=ExitCode.CONFIG_ERROR,
            )
        raw_profile_text = profile_input.read_text(encoding="utf-8")
    else:
        raw_profile_text = str(profile_input).strip()

    if not raw_profile_text:
        SessionLogger.fatal(
            "Candidate profile input is empty.",
            code=ExitCode.CONFIG_ERROR,
        )

    log.info("Initiating Role Matching & Search Query Optimization...")
    log.debug(f"Candidate profile payload size: {len(raw_profile_text)} characters")

    # Execute Gemini Engine inference
    active_engine = engine or GeminiEngine()

    try:
        result: RoleMatchResult = active_engine.match_role(
            profile_summary=raw_profile_text,
            schema=RoleMatchResult,
        )
    except Exception as exc:
        SessionLogger.fatal(
            "Role matching inference failed",
            code=ExitCode.LLM_ERROR,
            exc=exc,
        )
        raise

    # Handle insufficient profile data edge cases
    if result.status == "insufficient_data":
        log.warning(
            f"Candidate profile lacks sufficient data for role matching: {result.clarification_needed}"
        )
        return result

    if not result.primary_role or not result.primary_role.title:
        SessionLogger.fatal(
            "Gemini returned an empty primary role recommendation despite status='ok'",
            code=ExitCode.LLM_ERROR,
        )

    log.info(
        f"Primary role matched: '{result.primary_role.title}' "
        f"(Confidence: {result.primary_role.confidence_score}%)"
    )
    log.debug(f"Generated search queries: {result.primary_role.search_queries}")

    return result
