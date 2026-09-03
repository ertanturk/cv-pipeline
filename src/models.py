from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None and str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(v)]


class PrimaryRole(BaseModel):
    """Primary target role recommendation with high-intent search queries."""

    title: str = Field(default="", description="Target job/internship title")
    confidence_score: int = Field(default=0, description="Confidence score from 0 to 100")
    search_queries: list[str] = Field(
        default_factory=list, description="Targeted query strings for scrapers"
    )
    rationale: str = Field(default="", description="Evidence-based justification from profile")


class AlternativeRole(BaseModel):
    """Secondary career path alternative."""

    title: str = Field(default="", description="Alternative role title")
    confidence_score: int = Field(default=0, description="Confidence score from 0 to 100")
    rationale: str = Field(default="", description="Supporting evidence from profile")


class RoleMatchResult(BaseModel):
    """Output schema for role matching."""

    status: str = Field(default="ok")
    candidate_tier: str = Field(
        default="student",
        description="Either 'student' (active undergraduate/seeking internship) or 'junior' (graduated/seeking full-time entry-level)",
    )
    primary_role: PrimaryRole = Field(default_factory=PrimaryRole)
    alternative_roles: list[AlternativeRole] = Field(default_factory=list)
    clarification_needed: str = Field(default="")


class JobPosting(BaseModel):
    """In-memory representation of a single scraped job posting."""

    title: str = Field(default="", description="Job title")
    company: str = Field(default="", description="Hiring company name")
    location: str = Field(default="", description="Job location or remote status")
    description: str = Field(default="", description="Full job description or summary text")
    job_url: str = Field(default="", description="Direct URL to posting")
    site: str = Field(default="", description="Platform source (linkedin, indeed, etc.)")


class ExtractedTool(BaseModel):
    """Individual tool, language, or platform qualification."""

    name: str = Field(default="", description="Canonical tool or technology name")
    priority: str = Field(default="required", description="'required' or 'preferred'")


class JobExtractionResult(BaseModel):
    """Output schema for Stage 3 single job extraction."""

    status: str = Field(default="ok", description="'ok' or 'no_data'")
    seniority_level: str = Field(default="", description="Intern, Junior, Mid, Senior, etc.")
    min_years_experience: int = Field(default=0, description="Minimum years required")
    tools: list[ExtractedTool] = Field(default_factory=list)
    methodologies: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    """Academic degree, institution, GPA, and relevant coursework record."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    institution: str = ""
    location: str = ""
    degree: str = ""
    gpa: str = ""
    start_date: str = ""
    end_date: str = ""
    coursework: list[str] = Field(default_factory=list)

    @field_validator("coursework", mode="before")
    @classmethod
    def _coerce_coursework(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class TechnicalSkills(BaseModel):
    """Categorized technical proficiencies, ordered by live market demand."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools_and_systems: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)

    @field_validator("languages", "frameworks", "tools_and_systems", "databases", mode="before")
    @classmethod
    def _coerce_skills(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class SpokenLanguageItem(BaseModel):
    """Natural spoken language and verified CEFR or fluency level."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    name: str = ""
    level: str = ""


class ProjectItem(BaseModel):
    """Technical software project with verified highlights and technologies."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    name: str = ""
    stack: list[str] = Field(default_factory=list)
    team_size: int | None = None
    link: str = ""
    description: str = Field(
        default="", description="One-line project overview, carried from source profile"
    )
    bullets: list[str] = Field(default_factory=list)

    @field_validator("stack", "bullets", mode="before")
    @classmethod
    def _coerce_project_lists(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class ExperienceItem(BaseModel):
    """Professional or academic employment history item."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    role: str = ""
    company: str = ""
    location: str = ""
    employment_type: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[str] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce_experience_bullets(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class LeadershipItem(BaseModel):
    """Leadership, mentoring, student club, or open-source community contribution."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    role: str = ""
    organization: str = ""
    date: str = ""
    link: str = ""
    bullets: list[str] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce_leadership_bullets(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class CertificationItem(BaseModel):
    """Formal industry certification, license, or verified award."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    name: str = ""
    issuer: str = ""
    date: str = ""


class ResumeData(BaseModel):
    """Final validated payload injected into resume_tr.tex.j2 or resume_en.tex.j2."""

    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    candidate_tier: str = "student"
    name: str = ""
    title: str = ""
    location: str = ""
    email: str = ""
    phone: str = ""
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    photo_path: str = ""
    summary: str = ""

    education: list[EducationItem] = Field(default_factory=list)
    skills: TechnicalSkills = Field(default_factory=TechnicalSkills)
    languages: list[SpokenLanguageItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    leadership: list[LeadershipItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)


class MarketSkill(BaseModel):
    """Ranked market competency with frequency and market share metrics."""

    name: str = Field(default="", description="Canonical skill or tool name")
    mentions: int = Field(default=0, description="Total postings requiring this skill")
    share_percentage: float = Field(
        default=0.0, description="Percentage of analyzed jobs containing this skill"
    )
    required_count: int = Field(
        default=0, description="Number of times explicitly marked as required"
    )


class MarketAnalysisResult(BaseModel):
    """Aggregated market requirement output."""

    total_analyzed: int = Field(default=0, description="Total postings successfully parsed")
    top_skills: list[MarketSkill] = Field(
        default_factory=list, description="Ranked technical tools and languages"
    )
    top_methodologies: list[MarketSkill] = Field(
        default_factory=list, description="Ranked methodologies and practices"
    )
    recommended_keywords: list[str] = Field(
        default_factory=list, description="Top canonical keywords for ATS alignment"
    )


class BulletItem(BaseModel):
    """Single accomplishment bullet."""

    text: str = Field(default="", description="Concise, factual accomplishment bullet")


class BatchJobExtractionResult(BaseModel):
    """Batch extraction of technical tools and methodologies across multiple job descriptions."""

    jobs: list[JobExtractionResult] = Field(
        default_factory=list,
        description="Extracted tools and methodologies for each input job posting in sequential order",
    )


class TailoredProjectBatchItem(BaseModel):
    """Tailored accomplishment container for a single project within a batch."""

    project_name: str = Field(default="", description="Exact name of the project")
    bullets: list[BulletItem] = Field(
        default_factory=list,
        description="3 high-impact accomplishment bullets: bullet 1 is a one-sentence overview, bullets 2-3 are ordered by importance",
    )


class BatchTailoredProjectsResult(BaseModel):
    """Batch container for multiple tailored projects."""

    projects: list[TailoredProjectBatchItem] = Field(
        default_factory=list,
        description="List of tailored accomplishment bullet sets corresponding to input projects",
    )


class TailoredSectionItem(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    title: str = Field(default="", description="Role or organization title matching source")
    bullets: list[str] = Field(
        default_factory=list,
        description="Action-oriented accomplishment bullets in the target language",
    )

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce_section_bullets(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class TailoredNonProjectContent(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    summary: str = Field(
        default="",
        description="High-impact engineering summary tailored to target role and language",
    )
    experience: list[TailoredSectionItem] = Field(
        default_factory=list,
        description="Tailored experience bullet sets in target language",
    )
    leadership: list[TailoredSectionItem] = Field(
        default_factory=list,
        description="Tailored leadership bullet sets in target language",
    )


class CompactProjectItem(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    project_name: str = Field(default="", description="Project name matching resume project item")
    bullets: list[str] = Field(default_factory=list, description="Compressed project bullets")

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce_compact_bullets(cls, v: Any) -> list[str]:
        return _coerce_str_list(v)


class CompactResumeContent(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True, extra="ignore")

    summary: str = Field(default="", description="Compressed resume summary")
    projects: list[CompactProjectItem] = Field(
        default_factory=list, description="Compressed bullets per project"
    )
    experience: list[TailoredSectionItem] = Field(
        default_factory=list, description="Compressed bullets per experience item"
    )
    leadership: list[TailoredSectionItem] = Field(
        default_factory=list, description="Compressed bullets per leadership item"
    )
