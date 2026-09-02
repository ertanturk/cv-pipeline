"""LaTeX sanitization and escaping utilities."""

from __future__ import annotations

import re

# Characters that must be escaped in plain-text LaTeX environments
LATEX_SPECIAL_CHARS: dict[str, str] = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters while preserving math mode and pre-escaped entities."""
    if not text:
        return ""

    parts = re.split(r"(\$[^\$]+\$)", text)
    escaped_parts: list[str] = []

    for idx, part in enumerate(parts):
        # Odd indices are math blocks: retain as-is
        if idx % 2 == 1:
            escaped_parts.append(part)
            continue

        # Even indices are plain text
        sub_text = part
        # Do not double-escape already escaped symbols
        sub_text = re.sub(
            r"(?<!\\)([\\&%\$#_{}~^])",
            lambda m: LATEX_SPECIAL_CHARS.get(m.group(1), m.group(1)),
            sub_text,
        )

        sub_text = sub_text.replace("C++", r"C\texttt{++}")
        escaped_parts.append(sub_text)

    return "".join(escaped_parts)
