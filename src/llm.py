"""Gemini LLM client with rate limiting, prompt injection, and schema enforcement."""

from typing import Any, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from pydantic import BaseModel

from src.models import (
    BatchJobExtractionResult,
    BatchTailoredProjectsResult,
    CompactResumeContent,
    ResumeData,
    TailoredNonProjectContent,
)
from src.prompts import (
    COMPACT_SYSTEM_INSTRUCTION_EN,
    COMPACT_SYSTEM_INSTRUCTION_TR,
    JOB_EXTRACTOR_SYSTEM_INSTRUCTION,
    ROLE_MATCHER_SYSTEM_INSTRUCTION,
    TAILOR_SYSTEM_INSTRUCTION_EN,
    TAILOR_SYSTEM_INSTRUCTION_TR,
    build_compact_resume_prompt,
    build_job_extractor_batch_prompt,
    build_job_extractor_prompt,
    build_role_matcher_prompt,
    build_tailor_batch_prompt,
    build_tailor_profile_sections_prompt,
    build_tailor_prompt,
)
from src.utils.exit_codes import ExitCode
from src.utils.key_manager import KeyManager
from src.utils.logger import SessionLogger, log
from src.utils.rate_limiter import BudgetExhaustedError, get_limiter

T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiEngine:
    """Safe wrapper around google-genai using pre-built prompts and strict schemas."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        api_key = KeyManager.get_key()
        self.client = genai.Client(api_key=api_key)
        self.limiter = get_limiter(model)

    @staticmethod
    def _tailor_system_instruction(language: str) -> str:
        return (
            TAILOR_SYSTEM_INSTRUCTION_TR
            if language.lower().strip() == "tr"
            else TAILOR_SYSTEM_INSTRUCTION_EN
        )

    @staticmethod
    def _compact_system_instruction(language: str) -> str:
        return (
            COMPACT_SYSTEM_INSTRUCTION_TR
            if language.lower().strip() == "tr"
            else COMPACT_SYSTEM_INSTRUCTION_EN
        )

    def _generate(
        self,
        prompt: str,
        system_instruction: str = "",
        response_schema: type[T] | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.1,
        *,
        schema: type[T] | None = None,
    ) -> T:
        """Executes API request with rate limiting and Pydantic validation."""
        target_schema = response_schema if response_schema is not None else schema
        if target_schema is None:
            raise ValueError("A response schema must be provided to _generate.")

        try:
            self.limiter.acquire()
        except BudgetExhaustedError as e:
            SessionLogger.fatal(str(e), code=ExitCode.TOO_MANY_REQUESTS)
            raise

        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            response_mime_type="application/json",
            response_schema=target_schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        try:
            import time

            start_t = time.perf_counter()
            log.debug(
                f"Calling Gemini ({self.model_name}) with schema {target_schema.__name__} "
                f"(max_tokens={max_output_tokens}, temp={temperature})"
            )
            chat = self.client.chats.create(model=self.model_name, config=config)
            response = chat.send_message(prompt)
            elapsed_s = time.perf_counter() - start_t

            if not response.text:
                raise ValueError("Empty response from Gemini API.")

            log.debug(
                f"Gemini ({self.model_name}) responded in {elapsed_s:.2f}s "
                f"({len(response.text)} chars received, validated against {target_schema.__name__})"
            )
            return target_schema.model_validate_json(response.text)

        except ServerError as e:
            self.limiter.abort()
            if e.code == 503:
                SessionLogger.fatal(
                    "Gemini is temporarily unavailable (503 SERVICE_UNAVAILABLE): "
                    "the model is experiencing high demand. Please retry in a moment.",
                    code=ExitCode.SERVICE_UNAVAILABLE,
                    exc=e,
                )
            else:
                SessionLogger.fatal(
                    f"Gemini server error ({e.code}): {e.message}",
                    code=ExitCode.LLM_ERROR,
                    exc=e,
                )
            raise
        except ClientError as e:
            if e.code == 429:
                self.limiter.abort()
                SessionLogger.fatal(
                    "Gemini rate limit exceeded (429 TOO_MANY_REQUESTS): "
                    "wait before retrying or reduce request frequency.",
                    code=ExitCode.TOO_MANY_REQUESTS,
                    exc=e,
                )
            elif e.code == 400:
                SessionLogger.fatal(
                    f"Gemini rejected the request (400 BAD_REQUEST): {e.message}",
                    code=ExitCode.BAD_REQUEST,
                    exc=e,
                )
            elif e.code == 404:
                SessionLogger.fatal(
                    f"Gemini model '{self.model_name}' not found (404 NOT_FOUND). "
                    "Check the model name in your configuration.",
                    code=ExitCode.NOT_FOUND,
                    exc=e,
                )
            else:
                SessionLogger.fatal(
                    f"Gemini client error ({e.code}): {e.message}",
                    code=ExitCode.LLM_ERROR,
                    exc=e,
                )
            raise
        except APIError as e:
            SessionLogger.fatal(
                f"Gemini API error ({e.code}): {e.message}",
                code=ExitCode.LLM_ERROR,
                exc=e,
            )
            raise
        except Exception as e:
            SessionLogger.fatal(
                "Failed to validate LLM response schema.",
                code=ExitCode.LLM_ERROR,
                exc=e,
            )
            raise

    def match_role(self, profile_summary: str, schema: type[T]) -> T:
        """Runs role matching."""
        prompt = build_role_matcher_prompt(profile_summary)
        return self._generate(
            prompt=prompt,
            system_instruction=ROLE_MATCHER_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=2048,
        )

    def extract_job_skills(self, job_text: str, schema: type[T]) -> T:
        """Runs map extraction on a single job description."""
        prompt = build_job_extractor_prompt(job_text)
        return self._generate(
            prompt=prompt,
            system_instruction=JOB_EXTRACTOR_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=2048,
        )

    def tailor_project(
        self,
        raw_project_data: str,
        top_skills: list[str],
        target_role: str,
        schema: type[T],
        language: str = "en",
    ) -> T:
        """Runs project bullet tailoring: overview bullet first, then details ordered by impact."""
        prompt = build_tailor_prompt(raw_project_data, top_skills, target_role, language)
        return self._generate(
            prompt=prompt,
            system_instruction=self._tailor_system_instruction(language),
            response_schema=schema,
            max_output_tokens=2048,
            temperature=0.35,
        )

    def extract_batch_job_skills(
        self,
        job_payloads: list[dict[str, str]],
        schema: type[BatchJobExtractionResult] = BatchJobExtractionResult,
    ) -> BatchJobExtractionResult:
        """Extracts technical tools and methodologies across multiple job postings in a single prompt."""
        prompt = build_job_extractor_batch_prompt(job_payloads)

        return self._generate(
            prompt=prompt,
            system_instruction=JOB_EXTRACTOR_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=8192,
        )

    def tailor_projects_batch(
        self,
        projects_data: list[dict[str, Any]],
        top_skills: list[str],
        target_role: str,
        schema: type[BatchTailoredProjectsResult] = BatchTailoredProjectsResult,
        language: str = "en",
    ) -> BatchTailoredProjectsResult:
        """Tailors multiple projects simultaneously: overview bullet first, then details ordered by impact."""
        prompt = build_tailor_batch_prompt(
            projects_data=projects_data,
            top_skills=top_skills,
            target_role=target_role,
            language=language,
        )

        return self._generate(
            prompt=prompt,
            system_instruction=self._tailor_system_instruction(language),
            response_schema=schema,
            max_output_tokens=8192,
            temperature=0.35,
        )

    def tailor_profile_sections(
        self,
        summary: str,
        experience: list[dict[str, Any]],
        leadership: list[dict[str, Any]],
        target_role: str,
        top_skills: list[str],
        language: str = "en",
    ) -> TailoredNonProjectContent:
        """Tailors summary, experience bullets, and leadership bullets to the target role and language."""
        prompt = build_tailor_profile_sections_prompt(
            summary=summary,
            experience=experience,
            leadership=leadership,
            target_role=target_role,
            top_skills=top_skills,
            language=language,
        )
        return self._generate(
            prompt=prompt,
            system_instruction=self._tailor_system_instruction(language),
            response_schema=TailoredNonProjectContent,
            max_output_tokens=4096,
            temperature=0.35,
        )

    def compact_resume_content(
        self,
        resume_data: ResumeData,
        language: str = "en",
        max_project_bullets: int = 3,
        max_experience_bullets: int = 2,
        max_leadership_bullets: int = 2,
        max_bullet_words: int = 22,
        max_summary_words: int = 36,
    ) -> CompactResumeContent:
        """Compacts summary and bullets to improve one-page fit while preserving factual content."""
        payload: dict[str, Any] = {
            "summary": resume_data.summary,
            "projects": [
                {
                    "project_name": p.name,
                    "team_size": p.team_size,
                    "description": p.description,
                    "bullets": p.bullets,
                }
                for p in resume_data.projects
            ],
            "experience": [
                {
                    "title": e.role,
                    "start_date": e.start_date,
                    "end_date": e.end_date,
                    "bullets": e.bullets,
                }
                for e in resume_data.experience
            ],
            "leadership": [
                {"title": lead.role, "date": lead.date, "bullets": lead.bullets}
                for lead in resume_data.leadership
            ],
        }
        prompt = build_compact_resume_prompt(
            resume_payload=payload,
            language=language,
            max_project_bullets=max_project_bullets,
            max_experience_bullets=max_experience_bullets,
            max_leadership_bullets=max_leadership_bullets,
            max_bullet_words=max_bullet_words,
            max_summary_words=max_summary_words,
        )
        return self._generate(
            prompt=prompt,
            system_instruction=self._compact_system_instruction(language),
            response_schema=CompactResumeContent,
            max_output_tokens=4096,
            temperature=0.1,
        )
