"""UI design tokens, stream separation, and glyph helpers."""

import sys

from rich.box import SIMPLE_HEAD
from rich.console import Console
from rich.table import Table
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


def format_step(current: int, total: int, title: str) -> None:
    """Renders structured execution step to stdout."""
    console.print(f"[step][{current}/{total}][/step] [highlight]{title}[/highlight]")


def render_distribution_bar(percentage: float, width: int = 10) -> str:
    """High-density text progress bar."""
    filled = int((percentage / 100) * width)
    empty = width - filled
    return f"[metric]{'█' * filled}[/metric][muted]{'░' * empty}[/muted] [muted]{percentage:.0f}%[/muted]"


def build_market_table(skills_data: list[tuple[str, int, float]]) -> Table:
    """Renders flat skill market analysis table to stdout."""
    table = Table(
        box=SIMPLE_HEAD,
        pad_edge=False,
        show_edge=False,
        header_style="muted",
    )
    table.add_column("#", justify="right", style="muted", width=3)
    table.add_column("Skill / Keyword", style="highlight", min_width=22)
    table.add_column("Mentions", justify="right", style="muted", width=10)
    table.add_column("Share", justify="right", style="highlight", width=8)
    table.add_column("Market Saturation", justify="left", width=18)

    for idx, (skill, count, share) in enumerate(skills_data, start=1):
        table.add_row(
            str(idx),
            skill,
            str(count),
            f"{share:.1f}%",
            render_distribution_bar(share),
        )

    return table
