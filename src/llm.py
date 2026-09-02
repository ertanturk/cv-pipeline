"""Gemini LLM client with rate limiting, prompt injection, and schema enforcement."""

from typing import Any, TypeVar

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from pydantic import BaseModel

from src.models import BatchJobExtractionResult, BatchTailoredProjectsResult
from src.prompts import (
    JOB_EXTRACTOR_SYSTEM_INSTRUCTION,
    ROLE_MATCHER_SYSTEM_INSTRUCTION,
    TAILOR_SYSTEM_INSTRUCTION,
    build_job_extractor_prompt,
    build_role_matcher_prompt,
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
            log.debug(f"Calling Gemini ({self.model_name}) with schema {target_schema.__name__}")
            chat = self.client.chats.create(model=self.model_name, config=config)
            response = chat.send_message(prompt)
            if not response.text:
                raise ValueError("Empty response from Gemini API.")

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
        """Runs project bullet tailoring with Google XYZ constraints."""
        prompt = build_tailor_prompt(raw_project_data, top_skills, target_role, language)
        return self._generate(
            prompt=prompt,
            system_instruction=TAILOR_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=2048,
        )

    def extract_batch_job_skills(
        self,
        job_payloads: list[dict[str, str]],
        schema: type[BatchJobExtractionResult] = BatchJobExtractionResult,
    ) -> BatchJobExtractionResult:
        """Extracts technical tools and methodologies across multiple job postings in a single prompt."""
        formatted_jobs: list[str] = []
        for idx, item in enumerate(job_payloads, start=1):
            formatted_jobs.append(
                f"[Job #{idx}: {item.get('title', 'Unknown')} at {item.get('company', 'Unknown')}]\n"
                f"{item.get('text', '').strip()}"
            )

        prompt = (
            f"You are an expert technical recruiter and market intelligence analyst.\n"
            f"Analyze the following {len(job_payloads)} job postings simultaneously.\n\n"
            f"For EACH job posting:\n"
            f"1. Extract canonical tools, technologies, databases, frameworks, and programming languages under 'tools'.\n"
            f"   - Set priority to 'required' if the skill is mandatory or core.\n"
            f"   - Set priority to 'preferred' if mentioned as 'nice to have', 'plus', or optional.\n"
            f"   - Use canonical industry naming (e.g. 'PostgreSQL', 'Docker', 'FastAPI', 'AWS', 'Redis').\n"
            f"2. Extract functional methodologies, system architectures, and engineering practices under 'methodologies' "
            f"(e.g. 'Microservices', 'RESTful API', 'CI/CD', 'TDD', 'Event-Driven Architecture').\n"
            f"3. If a posting has no discernible technical requirements, set status='no_data'. Otherwise set status='success'.\n\n"
            f"JOB POSTINGS TO ANALYZE:\n"
            f"{'=' * 60}\n" + f"\n\n{'=' * 60}\n".join(formatted_jobs)
        )

        return self._generate(
            prompt=prompt,
            system_instruction=JOB_EXTRACTOR_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=4096,
        )

    def tailor_projects_batch(
        self,
        projects_data: list[dict[str, Any]],
        top_skills: list[str],
        target_role: str,
        schema: type[BatchTailoredProjectsResult] = BatchTailoredProjectsResult,
        language: str = "en",
    ) -> BatchTailoredProjectsResult:
        """Tailors multiple projects simultaneously using the Google XYZ framework."""
        formatted_projects: list[str] = []
        for idx, p in enumerate(projects_data, start=1):
            highlights_str = "\n".join(f"- {h}" for h in p.get("highlights", []))
            formatted_projects.append(
                f"[Project #{idx}: {p.get('name', 'Project')}]\n"
                f"Tech Stack: {', '.join(p.get('stack', []))}\n"
                f"Overview: {p.get('description', '')}\n"
                f"Verified Accomplishments:\n{highlights_str}"
            )

        lang_instruction = "English" if language.lower() == "en" else "Turkish"
        prompt = (
            f"You are an elite executive resume writer specializing in the Google XYZ framework:\n"
            f"'Accomplished [X] as measured by [Y], by doing [Z]'\n\n"
            f"Target Role: {target_role}\n"
            f"Top Market Competencies: {', '.join(top_skills)}\n"
            f"Output Language: {lang_instruction}\n\n"
            f"Below are {len(projects_data)} verified candidate projects.\n"
            f"For EACH project, produce exactly 3 high-impact accomplishment bullets adhering strictly to:\n"
            f"1. Google XYZ structure: Start with a strong action verb ([X]), quantify the achievement or outcome ([Y]), "
            f"and describe the technical implementation or tools utilized ([Z]).\n"
            f"2. Prioritize technologies from the candidate's actual stack that intersect with the Top Market Competencies.\n"
            f"3. Strict truthfulness: NEVER invent or hallucinate metrics, technologies, or responsibilities not verified in the project input.\n"
            f"4. Output exactly 3 bullets per project under its respective 'project_name'.\n\n"
            f"PROJECTS TO TAILOR:\n"
            f"{'=' * 60}\n" + f"\n\n{'=' * 60}\n".join(formatted_projects)
        )

        return self._generate(
            prompt=prompt,
            system_instruction=TAILOR_SYSTEM_INSTRUCTION,
            response_schema=schema,
            max_output_tokens=4096,
        )
