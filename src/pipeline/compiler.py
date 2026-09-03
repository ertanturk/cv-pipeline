"""Hardened template rendering and PDF generation via pdflatex."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models import ResumeData
from src.utils.exit_codes import ExitCode
from src.utils.latex import escape_latex
from src.utils.logger import SessionLogger, log


def _sanitize_resume_data(data: ResumeData) -> ResumeData:
    """Recursively escapes special LaTeX characters across all text fields in ResumeData."""
    # Personal Info (Top-level text fields)
    data.name = escape_latex(data.name)
    data.title = escape_latex(data.title)
    data.location = escape_latex(data.location)
    data.summary = escape_latex(data.summary)

    # Spoken Languages
    for lang in data.languages:
        lang.name = escape_latex(lang.name)
        lang.level = escape_latex(lang.level)

    # Skills
    data.skills.languages = [escape_latex(s) for s in data.skills.languages]
    data.skills.frameworks = [escape_latex(s) for s in data.skills.frameworks]
    data.skills.tools_and_systems = [escape_latex(s) for s in data.skills.tools_and_systems]
    data.skills.databases = [escape_latex(s) for s in data.skills.databases]

    # Projects
    for proj in data.projects:
        proj.name = escape_latex(proj.name)
        proj.description = escape_latex(proj.description)
        proj.stack = [escape_latex(t) for t in proj.stack]
        proj.bullets = [escape_latex(b) for b in proj.bullets]

    # Experience
    for exp in data.experience:
        exp.role = escape_latex(exp.role)
        exp.company = escape_latex(exp.company)
        exp.location = escape_latex(exp.location)
        exp.employment_type = escape_latex(exp.employment_type)
        exp.start_date = escape_latex(exp.start_date)
        exp.end_date = escape_latex(exp.end_date)
        exp.bullets = [escape_latex(b) for b in exp.bullets]

    # Education
    for edu in data.education:
        edu.institution = escape_latex(edu.institution)
        edu.degree = escape_latex(edu.degree)
        edu.location = escape_latex(edu.location)
        edu.gpa = escape_latex(edu.gpa)
        edu.start_date = escape_latex(edu.start_date)
        edu.end_date = escape_latex(edu.end_date)
        edu.coursework = [escape_latex(c) for c in edu.coursework]

    # Leadership & Certifications
    for lead in data.leadership:
        lead.role = escape_latex(lead.role)
        lead.organization = escape_latex(lead.organization)
        lead.date = escape_latex(lead.date)
        lead.bullets = [escape_latex(b) for b in lead.bullets]

    for cert in data.certifications:
        cert.name = escape_latex(cert.name)
        cert.issuer = escape_latex(cert.issuer)
        cert.date = escape_latex(cert.date)

    return data


def run_compiler(
    resume_data: ResumeData,
    template_dir: Path | str = "templates",
    template_name: str = "resume_en.tex.j2",
    output_dir: Path | str = "output",
    output_filename: str = "resume.pdf",
    keep_tex: bool = True,
) -> Path:
    """Compiles tailored resume data into an ATS-friendly PDF using hardened execution paths.

    Args:
        resume_data: Validated and tailored ResumeData instance.
        template_dir: Directory holding Jinja2 LaTeX templates.
        template_name: File name of the Jinja2 LaTeX template (e.g. 'resume_en.tex.j2').
        output_dir: Destination directory for the generated PDF and TeX files.
        output_filename: Target output file name (e.g. 'John_Doe_Resume.pdf').
        keep_tex: Whether to retain the intermediate .tex source file on disk.

    Returns:
        Path to the compiled PDF file.
    """
    tpl_dir = Path(template_dir).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve absolute executable path with explicit type narrowing
    pdflatex_raw: str | None = shutil.which("pdflatex")
    if pdflatex_raw is None:
        SessionLogger.fatal(
            "pdflatex binary not found on PATH. Please install texlive.",
            code=ExitCode.SYSTEM_ERROR,
        )
        raise FileNotFoundError("pdflatex executable not found on system PATH")

    pdflatex_bin: str = str(Path(pdflatex_raw))

    log.info(f"Initiating Stage 5: LaTeX Compilation using '{template_name}'...")
    log.debug(f"Resolved pdflatex binary at: {pdflatex_bin}")

    # Align output stems so pdflatex names match the requested file
    file_stem = Path(output_filename).stem
    pdf_target = out_dir / f"{file_stem}.pdf"
    tex_path = out_dir / f"{file_stem}.tex"

    # Sanitize data fields via dedicated LaTeX escaping
    sanitized_data = _sanitize_resume_data(resume_data)

    # Setup Jinja2 (Using standard delimiters to match resume templates)
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
    )

    try:
        template = env.get_template(template_name)
    except Exception as exc:
        SessionLogger.fatal(
            f"Failed to locate LaTeX template '{template_name}' in '{tpl_dir}': {exc}",
            code=ExitCode.CONFIG_ERROR,
        )

    # Inject variables to support both direct {{ name }} and scoped {{ r.name }} accesses
    render_context = sanitized_data.model_dump()
    render_context["r"] = sanitized_data
    rendered_tex = template.render(**render_context)

    tex_path.write_text(rendered_tex, encoding="utf-8")
    log.debug(
        f"Rendered LaTeX source written to: {tex_path} "
        f"({len(rendered_tex)} characters, {len(rendered_tex.splitlines())} lines)"
    )

    # Compile with pdflatex (2 passes for hyperref link resolving)
    log.info(f"Compiling '{tex_path.name}' with pdflatex (2 passes)...")

    for pass_idx in range(1, 3):
        log.debug(f"Starting pdflatex pass #{pass_idx} for '{tex_path.name}'")
        proc = subprocess.run(  # noqa: S603
            [
                pdflatex_bin,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                f"-output-directory={out_dir}",
                str(tex_path),
            ],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            log.error(f"pdflatex failed on pass #{pass_idx}:\n{proc.stdout[-2000:]}")
            SessionLogger.fatal(
                "LaTeX compilation failed. Inspect output logs for syntax errors.",
                code=ExitCode.SYSTEM_ERROR,
            )

    # Clean auxiliary files
    cleaned_aux: list[str] = []
    for ext in [".aux", ".log", ".out"]:
        aux_file = out_dir / f"{file_stem}{ext}"
        if aux_file.exists():
            aux_file.unlink()
            cleaned_aux.append(ext)

    if cleaned_aux:
        log.debug(f"Cleaned auxiliary LaTeX files: {', '.join(cleaned_aux)}")

    if not keep_tex and tex_path.exists():
        tex_path.unlink()
        log.debug(f"Deleted intermediate TeX file: {tex_path}")

    pdf_size_kb = pdf_target.stat().st_size / 1024 if pdf_target.exists() else 0.0
    log.info(f"Compilation successful: PDF generated at '{pdf_target}' ({pdf_size_kb:.1f} KB)")
    return pdf_target
