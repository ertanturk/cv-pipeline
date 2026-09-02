"""Standardized application exit codes."""

from enum import IntEnum


class ExitCode(IntEnum):
    """POSIX-compliant process exit codes."""

    SUCCESS = 0
    GENERAL_ERROR = 1
    CONFIG_ERROR = 2
    SCRAPER_ERROR = 3
    LLM_ERROR = 4
    COMPILER_ERROR = 5
    SYSTEM_ERROR = 6
    BAD_REQUEST = 400
    NOT_FOUND = 404
    TOO_MANY_REQUESTS = 429
    SERVICE_UNAVAILABLE = 503
