"""Pydantic validation models strictly compatible with Gemini OpenAPI schema engine."""

from pydantic import BaseModel, Field


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
    """Output schema for Stage 1 role matching."""

    status: str = Field(
        default="ok", description="'ok' if profile has enough data, else 'insufficient_data'"
    )
    primary_role: PrimaryRole = Field(default_factory=PrimaryRole)
    alternative_roles: list[AlternativeRole] = Field(default_factory=list)
    clarification_needed: str = Field(
        default="", description="Explanation if status is insufficient_data"
    )


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


class BulletPoint(BaseModel):
    """Single Google XYZ accomplishment statement."""

    text: str = Field(default="", description="Accomplishment statement using Google XYZ syntax")
    has_quantified_metric: bool = Field(
        default=False, description="True if statement includes a quantified metric"
    )


class TailoredBulletsResult(BaseModel):
    """Output schema for Stage 4 project/experience tailoring."""

    status: str = Field(default="ok", description="'ok' or 'insufficient_data'")
    bullets: list[BulletPoint] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str = ""
    location: str = ""
    degree: str = ""
    gpa: str = ""
    start_date: str = ""
    end_date: str = ""
    coursework: list[str] = Field(default_factory=list)


class TechnicalSkills(BaseModel):
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools_and_systems: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)


class SpokenLanguageItem(BaseModel):
    name: str = ""
    level: str = ""


class ProjectItem(BaseModel):
    name: str = ""
    stack: list[str] = Field(default_factory=list)
    link: str = ""
    bullets: list[str] = Field(default_factory=list)


class ExperienceItem(BaseModel):
    role: str = ""
    company: str = ""
    location: str = ""
    employment_type: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[str] = Field(default_factory=list)


class LeadershipItem(BaseModel):
    role: str = ""
    organization: str = ""
    date: str = ""
    bullets: list[str] = Field(default_factory=list)


class CertificationItem(BaseModel):
    name: str = ""
    issuer: str = ""
    date: str = ""


class ResumeData(BaseModel):
    """Final validated payload injected into resume_tr.tex.j2 or resume_en.tex.j2."""

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

    text: str = Field(default="", description="Accomplishment bullet following Google XYZ formula")


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
        description="3 high-impact Google XYZ accomplishment bullets",
    )


class BatchTailoredProjectsResult(BaseModel):
    """Batch container for multiple tailored projects."""

    projects: list[TailoredProjectBatchItem] = Field(
        default_factory=list,
        description="List of tailored accomplishment bullet sets corresponding to input projects",
    )
