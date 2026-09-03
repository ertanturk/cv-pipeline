import contextlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import typer
import yaml

from src.languages import tr_message
from src.llm import GeminiEngine
from src.models import CompactResumeContent, MarketAnalysisResult, ResumeData, RoleMatchResult
from src.pipeline.compiler import cleanup_latex_artifacts, run_compiler
from src.pipeline.market_analyzer import run_market_analyzer
from src.pipeline.role_matcher import run_role_matcher
from src.pipeline.scraper import run_scraper
from src.pipeline.tailor import run_tailor
from src.ui.selection import confirm_inline, select_ui_language
from src.ui.theme import (
    GLYPH_CHECK,
    GLYPH_CROSS,
    GLYPH_WARN,
    build_market_table,
    console,
    err_console,
    format_step,
    render_app_header,
)
from src.utils.key_manager import KeyManager
from src.utils.logger import SessionLogger, log

app = typer.Typer(
    name="cv",
    help="AI-powered ATS resume generation engine. / Yapay zeka destekli ATS uyumlu özgeçmiş üretim motoru.",
    no_args_is_help=True,
    add_completion=False,
)

auth_app = typer.Typer(
    name="auth",
    help="Manage Gemini API key authentication in OS keyring. / İşletim sistemi anahtarlığında Gemini API anahtarını yönetir.",
    no_args_is_help=True,
)
app.add_typer(auth_app, name="auth")


def count_pdf_pages(pdf_path: Path) -> int:
    """Calculates total page count with robust PDF metadata parsing."""
    pdfinfo_bin = shutil.which("pdfinfo")
    if pdfinfo_bin:
        try:
            proc = subprocess.run(  # noqa: S603
                [pdfinfo_bin, str(pdf_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                match = re.search(r"^Pages:\s+(\d+)$", proc.stdout, flags=re.MULTILINE)
                if match:
                    return int(match.group(1))
        except OSError:
            ...

    try:
        content = pdf_path.read_bytes()
        count_refs = [int(n) for n in re.findall(rb"/Count\s+(\d+)", content)]
        if count_refs:
            return max(count_refs)
        pages = re.findall(rb"/Type\s*/Page\b", content)
        return len(pages) if pages else 1
    except Exception:
        return 1


def _extract_numeric_signals(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in re.finditer(r"O\(\d+\)|\d+(?:[.,]\d+)?\+?%?", text, flags=re.IGNORECASE)
    }


def _has_new_numeric_signals(source_texts: list[str], candidate_texts: list[str]) -> bool:
    source_signals = _extract_numeric_signals(" ".join(source_texts))
    candidate_signals = _extract_numeric_signals(" ".join(candidate_texts))
    return not candidate_signals.issubset(source_signals)


def _compact_introduces_new_numbers(original: ResumeData, compacted: CompactResumeContent) -> bool:
    if _has_new_numeric_signals([original.summary], [compacted.summary]):
        return True

    original_projects_by_name: dict[str, deque[list[str]]] = defaultdict(deque)
    for project in original.projects:
        original_projects_by_name[project.name.lower().strip()].append(project.bullets)
    for idx, project in enumerate(compacted.projects):
        source_bullets: list[str] = []
        key = project.project_name.lower().strip()
        if key and original_projects_by_name.get(key):
            source_bullets = original_projects_by_name[key].popleft()
        elif idx < len(original.projects):
            source_bullets = original.projects[idx].bullets
        if source_bullets and _has_new_numeric_signals(source_bullets, project.bullets):
            return True

    original_exp_by_title: dict[str, deque[list[str]]] = defaultdict(deque)
    for exp in original.experience:
        original_exp_by_title[exp.role.lower().strip()].append(exp.bullets)
    for idx, item in enumerate(compacted.experience):
        source_bullets = []
        key = item.title.lower().strip()
        if key and original_exp_by_title.get(key):
            source_bullets = original_exp_by_title[key].popleft()
        elif idx < len(original.experience):
            source_bullets = original.experience[idx].bullets
        if source_bullets and _has_new_numeric_signals(source_bullets, item.bullets):
            return True

    original_lead_by_title: dict[str, deque[list[str]]] = defaultdict(deque)
    for lead in original.leadership:
        original_lead_by_title[lead.role.lower().strip()].append(lead.bullets)
    for idx, item in enumerate(compacted.leadership):
        source_bullets = []
        key = item.title.lower().strip()
        if key and original_lead_by_title.get(key):
            source_bullets = original_lead_by_title[key].popleft()
        elif idx < len(original.leadership):
            source_bullets = original.leadership[idx].bullets
        if source_bullets and _has_new_numeric_signals(source_bullets, item.bullets):
            return True

    return False


def _build_fallback_search_queries(role_title: str) -> list[str]:
    """Build lightweight fallback queries when role matching lacks enough data."""
    role = role_title.strip()
    candidates = [
        role,
        f"{role} intern",
        f"{role} internship",
        f"junior {role}",
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item.strip())
    return deduped


def _select_ui_language() -> str:
    """Prompt for UI language before pipeline starts with two inline options on the same line."""
    return select_ui_language(default="tr")


def _apply_compact_content(resume_data: ResumeData, compacted: CompactResumeContent) -> ResumeData:
    """Apply compacted summary and bullets while preserving original section structure."""
    compact_summary = compacted.summary.strip()
    if compact_summary:
        resume_data.summary = compact_summary

    project_bullets_by_name: dict[str, deque[list[str]]] = defaultdict(deque)
    project_bullets_by_index: dict[int, list[str]] = {}
    for idx, item in enumerate(compacted.projects):
        cleaned = [b.strip() for b in item.bullets if b and b.strip()]
        if cleaned:
            project_bullets_by_index[idx] = cleaned
            key = item.project_name.lower().strip()
            if key:
                project_bullets_by_name[key].append(cleaned)

    for idx, project in enumerate(resume_data.projects):
        key = project.name.lower().strip()
        bullets: list[str] | None = None
        if key and project_bullets_by_name.get(key):
            bullets = project_bullets_by_name[key].popleft()
        elif idx in project_bullets_by_index:
            bullets = project_bullets_by_index[idx]
        if bullets:
            project.bullets = bullets

    exp_bullets_by_title: dict[str, deque[list[str]]] = defaultdict(deque)
    exp_bullets_by_index: dict[int, list[str]] = {}
    for idx, item in enumerate(compacted.experience):
        cleaned = [b.strip() for b in item.bullets if b and b.strip()]
        if cleaned:
            exp_bullets_by_index[idx] = cleaned
            key = item.title.lower().strip()
            if key:
                exp_bullets_by_title[key].append(cleaned)

    for idx, exp in enumerate(resume_data.experience):
        key = exp.role.lower().strip()
        bullets = None
        if key and exp_bullets_by_title.get(key):
            bullets = exp_bullets_by_title[key].popleft()
        elif idx in exp_bullets_by_index:
            bullets = exp_bullets_by_index[idx]
        if bullets:
            exp.bullets = bullets

    lead_bullets_by_title: dict[str, deque[list[str]]] = defaultdict(deque)
    lead_bullets_by_index: dict[int, list[str]] = {}
    for idx, item in enumerate(compacted.leadership):
        cleaned = [b.strip() for b in item.bullets if b and b.strip()]
        if cleaned:
            lead_bullets_by_index[idx] = cleaned
            key = item.title.lower().strip()
            if key:
                lead_bullets_by_title[key].append(cleaned)

    for idx, lead in enumerate(resume_data.leadership):
        key = lead.role.lower().strip()
        bullets = None
        if key and lead_bullets_by_title.get(key):
            bullets = lead_bullets_by_title[key].popleft()
        elif idx in lead_bullets_by_index:
            bullets = lead_bullets_by_index[idx]
        if bullets:
            lead.bullets = bullets

    return resume_data


@contextlib.contextmanager
def execute_step(text: str, verbose: bool, quiet: bool):
    if quiet:
        yield
    elif verbose:
        console.print(f"[muted]Started:[/muted] {text}")
        yield
        console.print(f"[muted]Finished:[/muted] {text}\n")
    else:
        with console.status(f"[muted]{text}[/muted]"):
            yield


@auth_app.command(
    "set",
    help="Store Gemini API key securely in OS keyring. / Gemini API anahtarını anahtarlıkta güvenle saklar.",
)
def auth_set():
    """Prompt for a Gemini API key and store it securely in the OS keyring."""
    render_app_header()
    api_key = typer.prompt("Enter your Gemini API Key", hide_input=True)
    if not api_key.strip():
        err_console.print(f"{GLYPH_CROSS} [error]API key cannot be empty.[/error]")
        raise typer.Exit(code=1)

    KeyManager.set_key(api_key)
    console.print(
        f"{GLYPH_CHECK} [success]API key securely encrypted and stored in OS keyring.[/success]"
    )


@auth_app.command(
    "show",
    help="Display stored API key in masked format. / Kayıtlı API anahtarını maskelenmiş biçimde gösterir.",
)
def auth_show():
    """Retrieve and display the active Gemini API key in a masked string format."""
    key = KeyManager.get_key()
    if not key:
        err_console.print(f"{GLYPH_CROSS} [error]No API key found in keyring.[/error]")
        raise typer.Exit(code=1)

    masked = f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}" if len(key) > 8 else "****"
    console.print(f"Stored Key: [brand]{masked}[/brand]")


@auth_app.command(
    "validate",
    help="Verify API key validity against Gemini REST API. / API anahtarının geçerliliğini ve kotasını doğrular.",
)
def auth_validate():
    """Verify stored credentials by querying the Google Generative Language API endpoint."""
    key = KeyManager.get_key()
    if not key:
        err_console.print(
            f"{GLYPH_CROSS} [error]No API key found. Run 'cv auth set' first.[/error]"
        )
        raise typer.Exit(code=1)

    console.print("[muted]Pinging Gemini API...[/muted]")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        headers={"User-Agent": "cv-pipeline/0.1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
            if response.status == 200:
                console.print(
                    f"{GLYPH_CHECK} [success]API key is valid and active. Quota is OK.[/success]"
                )
            else:
                err_console.print(
                    f"{GLYPH_CROSS} [error]API returned unexpected status code: {response.status}[/error]"
                )
                raise typer.Exit(code=1)
    except urllib.error.HTTPError as err:
        if err.code in (400, 401, 403):
            err_console.print(
                f"{GLYPH_CROSS} [error]API key is invalid or unauthorized (HTTP {err.code}).[/error]"
            )
        elif err.code == 429:
            err_console.print(
                f"{GLYPH_CROSS} [error]API quota exceeded or rate limited (HTTP 429).[/error]"
            )
        else:
            err_console.print(
                f"{GLYPH_CROSS} [error]API check failed with HTTP {err.code}:[/error] {err.reason!s}"
            )
        raise typer.Exit(code=1) from err
    except urllib.error.URLError as err:
        err_console.print(
            f"{GLYPH_CROSS} [error]Network error while connecting to Gemini API:[/error] {err.reason!s}"
        )
        raise typer.Exit(code=1) from err
    except Exception as err:
        err_console.print(
            f"{GLYPH_CROSS} [error]Unexpected error during API check:[/error] {err!s}"
        )
        raise typer.Exit(code=1) from err


@auth_app.command(
    "delete",
    help="Permanently delete the Gemini API key from OS keyring. / Gemini API anahtarını anahtarlıktan siler.",
)
def auth_delete(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation prompt and delete key immediately. / Onay sormadan siler.",
        ),
    ] = False,
):
    """Remove stored Gemini API credentials from the system keyring service."""
    if not yes and sys.stdin.isatty():
        confirmed = confirm_inline(
            "Are you sure you want to delete the stored Gemini API key?",
            default=False,
            ui_lang="en",
        )
        if not confirmed:
            console.print(f"{GLYPH_WARN} [warning]Deletion cancelled.[/warning]")
            raise typer.Exit(code=0)

    KeyManager.delete_key()
    console.print(f"{GLYPH_CHECK} [success]API key successfully removed.[/success]")


@auth_app.command(
    "status",
    help="Check whether a Gemini API key is currently configured. / API anahtarının yapılandırılıp yapılandırılmadığını kontrol eder.",
)
def auth_status():
    """Verify presence of stored API credentials in the OS keyring."""
    key = KeyManager.get_key()
    if key:
        console.print(f"{GLYPH_CHECK} [success]API key is configured and ready.[/success]")
    else:
        console.print(f"{GLYPH_WARN} [warning]API key is NOT configured.[/warning]")
        raise typer.Exit(code=1)


@app.command(
    "validate",
    help="Validate candidate profile YAML syntax. / Aday profil YAML sözdizimini ve yapısını doğrular.",
)
def validate_profile(
    profile_path: Annotated[
        Path,
        typer.Argument(
            help="Path to candidate profile YAML document. / Aday profil YAML dosya yolu."
        ),
    ],
):
    """Parse candidate profile YAML and verify schema conformance without API calls."""
    if not profile_path.exists():
        err_console.print(f"{GLYPH_CROSS} [error]File not found: {profile_path}[/error]")
        raise typer.Exit(code=1)

    try:
        with open(profile_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            err_console.print(
                f"{GLYPH_CROSS} [error]Invalid profile format: root element must be a dictionary.[/error]"
            )
            raise typer.Exit(code=1)
        console.print(f"{GLYPH_CHECK} [success]YAML profile syntax is valid.[/success]")
    except yaml.YAMLError as err:
        err_console.print(f"{GLYPH_CROSS} [error]YAML Parsing Error:[/error]\n{err!s}")
        raise typer.Exit(code=1) from err


@app.command(
    "clean",
    help="Remove LaTeX compilation build artifacts. / LaTeX derleme kalıntılarını (.aux, .log, .tex vb.) temizler.",
)
def clean_workspace(
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Target output directory to scan and clean. / Taranacak ve temizlenecek hedef çıktı dizini."
        ),
    ],
    all_files: Annotated[
        bool,
        typer.Option(
            "--all",
            "-a",
            help="Delete compiled PDF files as well. / Derlenmiş PDF dosyalarını da siler.",
        ),
    ] = False,
):
    """Purge intermediate LaTeX compiler outputs and optionally delete compiled PDFs."""
    if not output_dir.exists():
        console.print(f"{GLYPH_WARN} [warning]Directory does not exist: {output_dir}[/warning]")
        raise typer.Exit()

    extensions = [
        "*.aux",
        "*.log",
        "*.out",
        "*.fls",
        "*.fdb_latexmk",
        "*.synctex.gz",
        "*.synctex",
        "*.toc",
        "*.nav",
        "*.snm",
        "*.vrb",
        "*.bbl",
        "*.blg",
        "*.tex",
    ]
    if all_files:
        extensions.append("*.pdf")

    seen_paths: set[Path] = set()
    deleted = 0
    for ext in extensions:
        for f in output_dir.glob(ext):
            if f.is_file() and f not in seen_paths:
                seen_paths.add(f)
                with contextlib.suppress(OSError):
                    f.unlink()
                    deleted += 1

    console.print(
        f"{GLYPH_CHECK} [success]Cleaned {deleted} compilation artifacts from {output_dir}.[/success]"
    )


@app.command(
    "run",
    help="Execute the end-to-end 5-stage automated resume generation pipeline. / 5 aşamalı uçtan uca özgeçmiş üretim hattını çalıştırır.",
)
def run_pipeline(
    profile_path: Annotated[
        Path,
        typer.Argument(help="Path to candidate profile YAML file. / Aday profil YAML dosya yolu."),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory where compiled PDF will be generated. / Derlenen PDF ve çıktıların üretileceği dizin."
        ),
    ],
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            "-l",
            help="Target output resume language code ('en' or 'tr'). / Hedef özgeçmiş dili ('en' veya 'tr').",
        ),
    ] = "en",
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            "-j",
            help="Target number of active job postings to scrape. / Taranacak aktif iş ilanı sayısı.",
        ),
    ] = 100,
    max_analyze: Annotated[
        int | None,
        typer.Option(
            "--max-analyze",
            "-ma",
            help="Maximum job postings to analyze in market phase (default: dynamic based on scraped count). / Pazar analizinde incelenecek maksimum ilan sayısı.",
        ),
    ] = None,
    location: Annotated[
        str,
        typer.Option(
            "--location",
            "-loc",
            help="Target geographic job market location query. / Hedef pazar konumu (örn. 'Istanbul, Turkey').",
        ),
    ] = "Istanbul, Turkey",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Stream real-time debug diagnostics and logs to terminal. / Hata ayıklama loglarını terminale yazdırır.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress UI banners, tables, and progress spinners. / Terminal başlık ve ilerleme animasyonlarını gizler.",
        ),
    ] = False,
    target_role: Annotated[
        str | None,
        typer.Option(
            "--target-role",
            help="Fallback role title if detection is insufficient. / Otomatik tespit yetersiz kalırsa kullanılacak rol başlığı.",
        ),
    ] = None,
    ui_language: Annotated[
        str | None,
        typer.Option(
            "--ui-lang",
            help="CLI interface language code ('en' or 'tr'). / CLI arayüz dili ('en' veya 'tr').",
        ),
    ] = None,
):
    """Run all pipeline stages: Role Matching, Scraping, Market Analysis, Tailoring, and Compilation."""
    ui_lang: str
    if ui_language and ui_language.strip().lower() in ("en", "tr"):
        ui_lang = ui_language.strip().lower()
    elif quiet:
        ui_lang = lang if lang in ("en", "tr") else "tr"
    else:
        ui_lang = _select_ui_language()

    if not quiet:
        render_app_header()

    SessionLogger.setup(verbose=verbose, quiet=quiet)
    console.quiet = quiet
    err_console.quiet = False

    output_dir.mkdir(parents=True, exist_ok=True)

    if not profile_path.exists():
        profile_message = tr_message("profile_not_found", ui_lang).format(
            path=profile_path.resolve()
        )
        err_console.print(f"{GLYPH_CROSS} [error]{profile_message}[/error]")
        raise typer.Exit(code=1)

    key = KeyManager.get_key()
    if not key:
        err_console.print(
            f"{GLYPH_WARN} [warning]{tr_message('api_key_missing', ui_lang)}[/warning]"
        )
        raise typer.Exit(code=1)

    if not shutil.which("pdflatex"):
        err_console.print(f"{GLYPH_CROSS} [error]{tr_message('pdflatex_missing', ui_lang)}[/error]")
        raise typer.Exit(code=1)

    template_name = "resume_en.tex.j2" if lang == "en" else "resume_tr.tex.j2"
    template_path = PROJECT_ROOT / "templates" / template_name
    if not template_path.is_file():
        template_message = tr_message("template_not_found", ui_lang).format(
            path=template_path.resolve()
        )
        err_console.print(f"{GLYPH_CROSS} [error]{template_message}[/error]")
        raise typer.Exit(code=1)

    start_total = time.perf_counter()

    try:
        if not quiet:
            format_step(1, 5, tr_message("step_role", ui_lang))
        with execute_step(tr_message("step_role_exec", ui_lang), verbose, quiet):
            role_result = run_role_matcher(profile_path)

        resolved_role_title: str
        scraper_queries: list[str]
        tailor_target_role: str | RoleMatchResult
        if role_result.status == "insufficient_data" or not role_result.primary_role:
            if not quiet:
                console.print(
                    f"{GLYPH_WARN} [warning]{tr_message('insufficient_role_warning', ui_lang)}[/warning]"
                )
            fallback_role = (target_role or "").strip()
            if not fallback_role:
                fallback_role = typer.prompt(
                    tr_message("role_prompt", ui_lang),
                    default="Backend Engineer",
                ).strip()
            if not fallback_role:
                err_console.print(
                    f"{GLYPH_CROSS} [error]{tr_message('invalid_role_error', ui_lang)}[/error]"
                )
                raise typer.Exit(code=1)

            resolved_role_title = fallback_role
            scraper_queries = _build_fallback_search_queries(fallback_role)
            tailor_target_role = fallback_role
        else:
            primary = role_result.primary_role
            resolved_role_title = primary.title
            scraper_queries = primary.search_queries or _build_fallback_search_queries(
                primary.title
            )
            tailor_target_role = role_result

        if not quiet:
            resolved_role_message = tr_message("resolved_role", ui_lang).format(
                role=resolved_role_title
            )
            console.print(
                f"    [brand]↳[/brand] {resolved_role_message.replace(resolved_role_title, f'[highlight]{resolved_role_title}[/highlight]')}"
            )
            if role_result.status != "insufficient_data" and role_result.primary_role:
                confidence_message = tr_message("role_confidence", ui_lang).format(
                    score=role_result.primary_role.confidence_score
                )
                console.print(f"    [brand]↳[/brand] [muted]{confidence_message}[/muted]")

        if not quiet:
            format_step(2, 5, tr_message("step_scrape", ui_lang).format(location=location))
        with execute_step(
            tr_message("step_scrape_exec", ui_lang).format(jobs=jobs),
            verbose,
            quiet,
        ):
            postings = run_scraper(
                queries=scraper_queries,
                target_count=jobs,
                location=location,
                role_title=resolved_role_title,
            )
        if not quiet:
            retained_message = tr_message("retained_listings", ui_lang).format(count=len(postings))
            console.print(
                f"    [brand]↳[/brand] {retained_message.replace(str(len(postings)), f'[highlight]{len(postings)}[/highlight]')}"
            )

        if not quiet:
            format_step(3, 5, tr_message("step_market", ui_lang))
        with execute_step(tr_message("step_market_exec", ui_lang), verbose, quiet):
            market_result = run_market_analyzer(
                postings=postings,
                max_analyze=max_analyze,
            )

        if market_result.total_analyzed == 0 or not market_result.top_skills:
            should_continue = confirm_inline(
                prompt=tr_message("market_missing_confirm", ui_lang),
                default=False,
                ui_lang=ui_lang,
            )
            if not should_continue:
                err_console.print(
                    f"{GLYPH_CROSS} [error]{tr_message('market_aborted_error', ui_lang)}[/error]"
                )
                raise typer.Exit(code=1)

            market_result = MarketAnalysisResult(total_analyzed=0)
            if not quiet:
                console.print(
                    f"{GLYPH_WARN} [warning]{tr_message('market_continue_warning', ui_lang)}[/warning]"
                )

        if not quiet:
            skills_count = len(market_result.top_skills)
            analyzed_count = market_result.total_analyzed
            discovered_message = tr_message("discovered_skills", ui_lang).format(
                count=f"[highlight]{skills_count}[/highlight]",
                analyzed=f"[highlight]{analyzed_count}[/highlight]",
            )
            console.print(f"    [brand]↳[/brand] {discovered_message}")
            if market_result.top_skills and not verbose:
                console.print()
                table_data = [
                    (s.name, s.mentions, s.share_percentage) for s in market_result.top_skills[:5]
                ]
                console.print(
                    build_market_table(
                        table_data,
                        total_postings=market_result.total_analyzed,
                        language=ui_lang,
                    )
                )
                console.print()

        if not quiet:
            format_step(4, 5, tr_message("step_tailor", ui_lang))
        with execute_step(tr_message("step_tailor_exec", ui_lang), verbose, quiet):
            tailored_resume = run_tailor(
                profile_input=profile_path,
                market_analysis=market_result,
                target_role=tailor_target_role,
                language=lang,
            )
        if not quiet:
            tailored_message = tr_message("tailored_projects", ui_lang).format(
                count=len(tailored_resume.projects)
            )
            console.print(
                f"    [brand]↳[/brand] {tailored_message.replace(str(len(tailored_resume.projects)), f'[highlight]{len(tailored_resume.projects)}[/highlight]')}"
            )

        template_name = "resume_en.tex.j2" if lang == "en" else "resume_tr.tex.j2"
        safe_name = re.sub(r"[^\w\-]", "_", tailored_resume.name.strip()).strip("_") or "Candidate"
        out_filename = f"{safe_name}_Resume.pdf"
        tex_filename = out_filename.replace(".pdf", ".tex")

        if not quiet:
            format_step(5, 5, tr_message("step_compile", ui_lang).format(template=template_name))
        with execute_step(tr_message("step_compile_exec", ui_lang), verbose, quiet):
            pdf_path = run_compiler(
                resume_data=tailored_resume,
                template_dir=str(PROJECT_ROOT / "templates"),
                template_name=template_name,
                output_dir=str(output_dir),
                output_filename=out_filename,
                keep_tex=True,
            )
        compact_mode_applied = False
        compact_pass_used = 0
        page_count = count_pdf_pages(pdf_path)
        log.info(
            f"Initial PDF compiled: {page_count} page(s) ({pdf_path.stat().st_size / 1024:.1f} KB)"
        )
        if page_count > 1:
            compact_mode_applied = True
            initial_page_count = page_count
            log.warning(
                f"Resume exceeded 1 page target ({page_count} pages). Initiating automated compact mode."
            )
            if not quiet:
                compact_start = tr_message("compact_mode_start", ui_lang).format(pages=page_count)
                console.print(f"{GLYPH_WARN} [warning]{compact_start}[/warning]")

            engine = GeminiEngine()
            is_tr_resume = lang.lower().strip() == "tr"
            compact_passes: list[dict[str, int]] = [
                {
                    "max_project_bullets": 3,
                    "max_experience_bullets": 2,
                    "max_leadership_bullets": 2,
                    "max_bullet_words": 24 if is_tr_resume else 20,
                    "max_summary_words": 36,
                },
                {
                    "max_project_bullets": 2,
                    "max_experience_bullets": 2,
                    "max_leadership_bullets": 1,
                    "max_bullet_words": 22 if is_tr_resume else 17,
                    "max_summary_words": 30,
                },
                {
                    "max_project_bullets": 2,
                    "max_experience_bullets": 1,
                    "max_leadership_bullets": 1,
                    "max_bullet_words": 20 if is_tr_resume else 15,
                    "max_summary_words": 24,
                },
            ]

            fitted = False
            for pass_index, pass_limits in enumerate(compact_passes, start=1):
                with execute_step(
                    tr_message("compact_pass_exec", ui_lang).format(
                        current=pass_index, total=len(compact_passes)
                    ),
                    verbose,
                    quiet,
                ):
                    compacted = engine.compact_resume_content(
                        resume_data=tailored_resume,
                        language=lang,
                        **pass_limits,
                    )

                if _compact_introduces_new_numbers(tailored_resume, compacted):
                    if not quiet:
                        compact_warn = tr_message("compact_new_numbers", ui_lang).format(
                            pass_index=pass_index
                        )
                        console.print(f"{GLYPH_WARN} [warning]{compact_warn}[/warning]")
                    continue

                compact_candidate = _apply_compact_content(
                    tailored_resume.model_copy(deep=True), compacted
                )

                with execute_step(tr_message("compact_recompile_exec", ui_lang), verbose, quiet):
                    pdf_path = run_compiler(
                        resume_data=compact_candidate,
                        template_dir=str(PROJECT_ROOT / "templates"),
                        template_name=template_name,
                        output_dir=str(output_dir),
                        output_filename=out_filename,
                        keep_tex=True,
                    )

                page_count = count_pdf_pages(pdf_path)
                if page_count <= 1:
                    tailored_resume = compact_candidate
                    compact_pass_used = pass_index
                    fitted = True
                    log.info(
                        f"Compact pass #{pass_index} succeeded: resume fitted to 1 page "
                        f"({pdf_path.stat().st_size / 1024:.1f} KB)."
                    )
                    break

                log.debug(f"Compact pass #{pass_index} yielded {page_count} pages; continuing.")
                if not quiet:
                    pass_oversize = tr_message("compact_pass_still_oversize", ui_lang).format(
                        pass_index=pass_index, pages=page_count
                    )
                    console.print(f"{GLYPH_WARN} [warning]{pass_oversize}[/warning]")

            if not fitted and len(tailored_resume.projects) > 3:
                log.warning(
                    f"Page budget still exceeded ({page_count} pages). "
                    "Pruning lowest-priority project to guarantee 1-page fit..."
                )
                while not fitted and len(tailored_resume.projects) > 3:
                    tailored_resume.projects = tailored_resume.projects[:-1]
                    log.info(
                        f"Retrying compilation with top {len(tailored_resume.projects)} projects..."
                    )
                    pdf_path = run_compiler(
                        resume_data=tailored_resume,
                        template_dir=str(PROJECT_ROOT / "templates"),
                        template_name=template_name,
                        output_dir=str(output_dir),
                        output_filename=out_filename,
                        keep_tex=True,
                    )
                    page_count = count_pdf_pages(pdf_path)
                    if page_count <= 1:
                        fitted = True
                        log.info(
                            f"Resume fitted to 1 page with {len(tailored_resume.projects)} projects "
                            f"({pdf_path.stat().st_size / 1024:.1f} KB)."
                        )
                        break

            if not fitted:
                log.error(
                    f"Page budget exceeded after {len(compact_passes)} compact passes "
                    f"({page_count} pages). Removing oversized artifact."
                )
                cleanup_latex_artifacts(
                    output_dir=output_dir,
                    file_stem=Path(out_filename).stem,
                    include_tex=True,
                    include_pdf=True,
                )
                compact_fail = tr_message("compact_failed_limit", ui_lang).format(
                    passes=len(compact_passes), pages=page_count
                )
                err_console.print(f"{GLYPH_CROSS} [error]{compact_fail}[/error]")
                raise typer.Exit(code=1)

        if not quiet:
            if compact_mode_applied:
                compact_ok = tr_message("compact_applied", ui_lang).format(
                    initial=initial_page_count,
                    final=page_count,
                    pass_used=compact_pass_used,
                )
                console.print(f"    [brand]↳[/brand] {compact_ok}")
            pdf_msg = tr_message("pdf_generated", ui_lang).format(
                name=pdf_path.name,
                size_kb=pdf_path.stat().st_size / 1024,
            )
            console.print(f"    [brand]↳[/brand] {pdf_msg}")
            tex_msg = tr_message("latex_saved", ui_lang).format(name=tex_filename)
            console.print(f"    [brand]↳[/brand] {tex_msg}")

        total_dur = time.perf_counter() - start_total
        if not quiet:
            success_msg = tr_message("pipeline_success", ui_lang).format(seconds=total_dur)
            console.print(f"\n{GLYPH_CHECK} [success]{success_msg}[/success]\n")

    except typer.Exit:
        raise
    except Exception as e:
        if "out_filename" in locals():
            with contextlib.suppress(Exception):
                cleanup_latex_artifacts(
                    output_dir=output_dir,
                    file_stem=Path(out_filename).stem,
                    include_tex=True,
                    include_pdf=True,
                )
        fail_msg = tr_message("pipeline_failed", ui_lang).format(error=e)
        err_console.print(f"\n{GLYPH_CROSS} [error]{fail_msg}[/error]")
        if verbose:
            raise
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
