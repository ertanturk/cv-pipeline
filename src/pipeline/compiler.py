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
    """Recursively escapes special LaTeX characters across the ResumeData contract."""
    # Skills
    data.skills.languages = [escape_latex(s) for s in data.skills.languages]
    data.skills.frameworks = [escape_latex(s) for s in data.skills.frameworks]
    data.skills.tools_and_systems = [escape_latex(s) for s in data.skills.tools_and_systems]
    data.skills.databases = [escape_latex(s) for s in data.skills.databases]

    # Projects
    for proj in data.projects:
        proj.name = escape_latex(proj.name)
        proj.stack = [escape_latex(t) for t in proj.stack]
        proj.bullets = [escape_latex(b) for b in proj.bullets]

    # Experience
    for exp in data.experience:
        exp.role = escape_latex(exp.role)
        exp.company = escape_latex(exp.company)
        exp.location = escape_latex(exp.location)
        exp.bullets = [escape_latex(b) for b in exp.bullets]

    # Education
    for edu in data.education:
        edu.institution = escape_latex(edu.institution)
        edu.degree = escape_latex(edu.degree)
        edu.location = escape_latex(edu.location)
        edu.coursework = [escape_latex(c) for c in edu.coursework]

    # Leadership & Certifications
    for lead in data.leadership:
        lead.role = escape_latex(lead.role)
        lead.organization = escape_latex(lead.organization)
        lead.bullets = [escape_latex(b) for b in lead.bullets]

    for cert in data.certifications:
        cert.name = escape_latex(cert.name)
        cert.issuer = escape_latex(cert.issuer)

    return data


def run_compiler(
    resume_data: ResumeData,
    template_dir: Path | str = "templates",
    template_name: str = "modern_ats.tex.j2",
    output_dir: Path | str = "output",
    output_filename: str = "resume.pdf",
    keep_tex: bool = True,
) -> Path:
    """Compiles tailored resume data into an ATS-friendly PDF using hardened execution paths."""
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
    log.debug(f"Using pdflatex binary at: {pdflatex_bin}")

    # Align output stems so pdflatex names match the requested file
    file_stem = Path(output_filename).stem
    pdf_target = out_dir / f"{file_stem}.pdf"
    tex_path = out_dir / f"{file_stem}.tex"

    # Sanitize data fields via dedicated LaTeX escaping
    sanitized_data = _sanitize_resume_data(resume_data)

    # Setup Jinja2 with select_autoescape
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        block_start_string=r"\BLOCK{",
        block_end_string=r"}",
        variable_start_string=r"\VAR{",
        variable_end_string=r"}",
        comment_start_string=r"\#{",
        comment_end_string=r"}",
        line_statement_prefix=r"%%",
        line_comment_prefix=r"%#",
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

    rendered_tex = template.render(r=sanitized_data)
    tex_path.write_text(rendered_tex, encoding="utf-8")
    log.debug(f"Rendered LaTeX source written to: {tex_path}")

    # Compile with pdflatex
    log.info(f"Compiling '{tex_path.name}' with pdflatex...")

    for pass_idx in range(1, 3):
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

    # 6. Clean auxiliary files
    for ext in [".aux", ".log", ".out"]:
        aux_file = out_dir / f"{file_stem}{ext}"
        if aux_file.exists():
            aux_file.unlink()

    if not keep_tex and tex_path.exists():
        tex_path.unlink()

    log.info(f"Compilation successful: PDF generated at '{pdf_target}'")
    return pdf_target
