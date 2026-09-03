"""LaTeX sanitization and escaping utilities."""

from __future__ import annotations

import re
from typing import Any

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
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "\\": r"\textbackslash{}",
}


LLM_ARTIFACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\$(?:1|I)/O\$", re.IGNORECASE), "I/O"),
    (re.compile(r"C\\texttt\\?\{(\+\+)\\?\}"), "C++"),
    (re.compile(r"C\{\+\+\}"), "C++"),
    (re.compile(r"C\\texttt\\?\{(\#)\\?\}"), "C#"),
    (re.compile(r"C\{\#\}"), "C#"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
]


PRESERVED_URL_KEYS: set[str] = {
    "link",
    "url",
    "linkedin_url",
    "github_url",
    "portfolio_url",
    "photo_path",
}


def clean_llm_artifacts(text: str) -> str:
    """Generated LaTeX artifacts (e.g. $1/O$, C\texttt{++}) are cleaned to avoid double escaping."""
    for pattern, replacement in LLM_ARTIFACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def escape_latex(text: str) -> str:
    """Escapes LaTeX special characters while preserving URL keys and LLM artifacts."""
    if not text:
        return ""

    cleaned = clean_llm_artifacts(text)

    unescaped = re.sub(r"\\([&%$#_{}])", r"\1", cleaned)
    unescaped = unescaped.replace(r"\textasciitilde{}", "~")
    unescaped = unescaped.replace(r"\textasciicircum{}", "^")
    unescaped = unescaped.replace(r"\textless{}", "<")
    unescaped = unescaped.replace(r"\textgreater{}", ">")
    unescaped = unescaped.replace(r"\textbackslash{}", "\\")

    def _char_replacer(match: re.Match[str]) -> str:
        char = match.group(0)
        return LATEX_SPECIAL_CHARS.get(char, char)

    special_pattern = re.compile(r"[&%$#_{}~^<>\\]")
    escaped = special_pattern.sub(_char_replacer, unescaped)

    escaped = re.sub(r"\bC\+\+(?!\+)", r"C\\texttt{++}", escaped)
    escaped = re.sub(r"\bC\\#", r"C\\texttt{\\#}", escaped)
    escaped = re.sub(r"\bF\\#", r"F\\texttt{\\#}", escaped)

    return escaped


def sanitize_latex(data: Any, preserve_keys: set[str] | None = None) -> Any:
    """Recursively sanitizes LaTeX data, preserving URL keys and LLM artifacts."""
    if preserve_keys is None:
        preserve_keys = PRESERVED_URL_KEYS

    if isinstance(data, dict):
        return {
            k: v if k in preserve_keys else sanitize_latex(v, preserve_keys)
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [sanitize_latex(item, preserve_keys) for item in data]

    if isinstance(data, str):
        return escape_latex(data)

    return data
