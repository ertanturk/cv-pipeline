"""UI design tokens, stream separation, and glyph helpers."""

import sys

from rich.box import SIMPLE_HEAD
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

PIPELINE_THEME = Theme(
    {
        "brand": "bold #38BDF8",
        "step": "bold #38BDF8",
        "highlight": "bold #F8FAFC",
        "muted": "#64748B",
        "metric": "#0EA5E9",
        "success": "bold #34D399",
        "warning": "bold #FBBF24",
        "error": "bold #F87171",
        "lang.tr": "bold white on #DC2626",
        "lang.en": "bold white on #2563EB",
        "lang.border": "bold #334155",
        "lang.selected": "bold #22C55E",
    }
)

console = Console(theme=PIPELINE_THEME, file=sys.stdout)
err_console = Console(theme=PIPELINE_THEME, file=sys.stderr)

GLYPH_CHECK = "[success]✓[/success]"
GLYPH_CROSS = "[error]✗[/error]"
GLYPH_ARROW = "[muted]↳[/muted]"
GLYPH_WARN = "[warning]![/warning]"


def render_app_header() -> None:
    """Renders single-line application header to stdout."""
    console.print("\n[brand]cv-pipeline[/brand] [muted]0.1.0[/muted]\n")


def render_language_selector() -> None:
    """Renders two colored language selection boxes."""
    tr_box = Panel(
        Text("1  TR\nTürkçe Arayüz", justify="center"),
        border_style="lang.border",
        style="lang.tr",
        padding=(1, 3),
    )
    en_box = Panel(
        Text("2  EN\nEnglish UI", justify="center"),
        border_style="lang.border",
        style="lang.en",
        padding=(1, 3),
    )
    console.print(Columns([tr_box, en_box], equal=True, expand=True))


def format_step(current: int, total: int, title: str) -> None:
    """Renders structured execution step to stdout."""
    console.print(f"[step][{current}/{total}][/step] [highlight]{title}[/highlight]")


def render_distribution_bar(percentage: float, width: int = 10) -> str:
    """High-density text progress bar."""
    filled = int((percentage / 100) * width)
    empty = width - filled
    return f"[metric]{'█' * filled}[/metric][muted]{'░' * empty}[/muted] [muted]{percentage:.0f}%[/muted]"


def build_market_table(
    skills_data: list[tuple[str, int, float]],
    total_postings: int | None = None,
    language: str = "en",
) -> Table:
    """Renders flat skill market analysis table to stdout with multi-language support."""
    from src.languages import tr_message

    table = Table(
        box=SIMPLE_HEAD,
        pad_edge=False,
        show_edge=False,
        header_style="muted",
    )
    table.add_column(
        tr_message("market_col_num", language), justify="right", style="muted", width=3
    )
    table.add_column(tr_message("market_col_skill", language), style="highlight", min_width=22)
    mentions_header = tr_message("market_col_mentions", language)
    if total_postings and total_postings > 0:
        mentions_header = f"{mentions_header} (/{total_postings})"
    table.add_column(mentions_header, justify="right", style="muted", min_width=14)
    table.add_column(
        tr_message("market_col_share", language), justify="right", style="highlight", width=12
    )
    table.add_column(tr_message("market_col_saturation", language), justify="left", width=18)

    for idx, (skill, count, share) in enumerate(skills_data, start=1):
        table.add_row(
            str(idx),
            skill,
            str(count),
            f"{share:.1f}%",
            render_distribution_bar(share),
        )

    return table
