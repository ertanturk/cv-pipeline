"""Interactive inline selection components for terminal prompts and confirmations."""

import contextlib
import os
import sys
import termios
import tty
from typing import Any

from rich.markup import escape

from src.ui.theme import console


def _parse_keys(data: str) -> list[str]:
    """Parses raw terminal byte string into clean key tokens."""
    keys: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] == "\x1b":
            if i + 2 < n and data[i + 1] in ("[", "O"):
                code = data[i + 1 : i + 3]
                if code in ("[D", "OD"):
                    keys.append("left")
                elif code in ("[C", "OC"):
                    keys.append("right")
                elif code in ("[A", "OA"):
                    keys.append("up")
                elif code in ("[B", "OB"):
                    keys.append("down")
                elif code == "[Z":
                    keys.append("shift_tab")
                else:
                    keys.append("esc")
                i += 3
                continue
            keys.append("esc")
            i += 1
            continue

        if data[i] in ("\r", "\n"):
            keys.append("enter")
        elif data[i] == "\t":
            keys.append("tab")
        elif data[i] == " ":
            keys.append("space")
        elif data[i] == "\x03":
            keys.append("ctrl_c")
        elif data[i] == "\x04":
            keys.append("ctrl_d")
        else:
            keys.append(data[i].lower())
        i += 1
    return keys


def _format_inline_line(
    prompt: str,
    options: list[tuple[str, Any]],
    current_index: int,
    style: str = "brand",
) -> str:
    """Formats the single-line interactive prompt with options and visual focus."""
    parts = []
    for idx, (label, val) in enumerate(options):
        escaped_label = escape(label)
        if idx == current_index:
            if style == "confirm":
                color = "#16A34A" if bool(val) else "#DC2626"
                parts.append(f"[bold white on {color}]{escaped_label}[/bold white on {color}]")
            elif style == "lang":
                color = "#2563EB" if str(val) == "en" else "#DC2626"
                parts.append(f"[bold white on {color}]{escaped_label}[/bold white on {color}]")
            else:
                parts.append(f"[bold white on #0284C7]{escaped_label}[/bold white on #0284C7]")
        else:
            parts.append(f"[dim]{escaped_label}[/dim]")

    options_str = " ".join(parts)
    prompt_clean = prompt.rstrip()
    return f"{prompt_clean} {options_str}"


def select_inline[T](
    prompt: str,
    options: list[tuple[str, T]],
    default_index: int = 0,
    hotkeys: dict[str, int] | None = None,
    style: str = "brand",
) -> T:
    """Renders options horizontally on the same line and lets the user select interactively.

    Args:
        prompt: Display question or title.
        options: List of (label, value) tuples presented on the same line.
        default_index: Initially highlighted option index.
        hotkeys: Mapping of key presses to option index.
        style: Theme color style ('brand', 'lang', or 'confirm').

    Returns:
        The selected value corresponding to options[chosen_index][1].
    """
    if not options:
        raise ValueError("Options list cannot be empty.")

    current_index = max(0, min(default_index, len(options) - 1))

    # Non-interactive fallback when stdin/stdout is not a TTY (CI, pipes, scripts)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        with contextlib.suppress(Exception):
            if sys.stdin.readable():
                raw = sys.stdin.readline().strip().lower()
                if raw:
                    if hotkeys and raw in hotkeys:
                        current_index = hotkeys[raw]
                    else:
                        for i, (label, val) in enumerate(options):
                            clean_label = label.strip("[]").lower()
                            if raw in (str(val).lower(), label.lower(), clean_label):
                                current_index = i
                                break

        selected_label, selected_val = options[current_index]
        clean_choice = str(selected_val) if style == "lang" else selected_label.strip("[]")
        prompt_clean = prompt.rstrip()
        if style == "confirm":
            color = "success" if bool(selected_val) else "error"
            glyph = "[success]✓[/success]" if bool(selected_val) else "[error]✗[/error]"
            console.print(f"{glyph} {prompt_clean} [{color}]{clean_choice}[/{color}]")
        else:
            console.print(f"[success]✓[/success] {prompt_clean} [brand]{clean_choice}[/brand]")
        return options[current_index][1]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Interactive TTY mode: hide cursor during navigation
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()

    try:
        tty.setraw(fd)
        # Initial render
        line = _format_inline_line(prompt, options, current_index, style=style)
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.flush()
        console.print(line, end="")
        sys.stdout.flush()

        done = False
        while not done:
            raw_bytes = os.read(fd, 32)
            if not raw_bytes:
                break
            try:
                decoded = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue

            keys = _parse_keys(decoded)
            for key in keys:
                if key == "ctrl_c":
                    raise KeyboardInterrupt
                if key == "ctrl_d":
                    raise EOFError

                if key in ("left", "up"):
                    current_index = (current_index - 1) % len(options)
                elif key in ("right", "down", "tab", "shift_tab"):
                    current_index = (current_index + 1) % len(options)
                elif hotkeys and key in hotkeys:
                    current_index = hotkeys[key]
                elif key in ("enter", "space", "esc"):
                    done = True
                    break

            # Redraw line with updated active index
            line = _format_inline_line(prompt, options, current_index, style=style)
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            console.print(line, end="")
            sys.stdout.flush()

    except KeyboardInterrupt:
        sys.stdout.write("\x1b[?25h\r\x1b[2K\n")
        sys.stdout.flush()
        raise
    finally:
        # Always restore terminal settings and cursor
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

    # Clear interactive line and print confirmed state
    sys.stdout.write("\r\x1b[2K")
    sys.stdout.flush()

    selected_label, selected_val = options[current_index]
    clean_choice = str(selected_val) if style == "lang" else selected_label.strip("[]")
    prompt_clean = prompt.rstrip()

    if style == "confirm":
        if bool(selected_val):
            console.print(f"[success]✓[/success] {prompt_clean} [success]{clean_choice}[/success]")
        else:
            console.print(f"[error]✗[/error] {prompt_clean} [error]{clean_choice}[/error]")
    else:
        console.print(f"[success]✓[/success] {prompt_clean} [brand]{clean_choice}[/brand]")

    return options[current_index][1]


def select_ui_language(default: str = "en") -> str:
    """Presents [en] [tr] options on the same line with arrow key highlight navigation."""
    options = [("[en]", "en"), ("[tr]", "tr")]
    default_idx = 0 if default == "en" else 1
    hotkeys = {
        "e": 0,
        "en": 0,
        "1": 0,
        "t": 1,
        "tr": 1,
        "2": 1,
    }
    return select_inline(
        prompt="Select CLI language (en/tr):",
        options=options,
        default_index=default_idx,
        hotkeys=hotkeys,
        style="lang",
    )


def confirm_inline(
    prompt: str,
    default: bool = False,
    ui_lang: str = "tr",
) -> bool:
    """Presents Yes/No options on the same line with arrow key highlight navigation."""
    if ui_lang == "tr":
        options = [("[Evet]", True), ("[Hayır]", False)]
        hotkeys = {
            "e": 0,
            "y": 0,
            "1": 0,
            "h": 1,
            "n": 1,
            "2": 1,
        }
    else:
        options = [("[Yes]", True), ("[No]", False)]
        hotkeys = {
            "y": 0,
            "1": 0,
            "n": 1,
            "2": 1,
        }

    default_idx = 0 if default else 1
    return select_inline(
        prompt=prompt,
        options=options,
        default_index=default_idx,
        hotkeys=hotkeys,
        style="confirm",
    )
