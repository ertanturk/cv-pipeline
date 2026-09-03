"""Session-based rolling logging engine for cv-pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.exit_codes import ExitCode

LOGS_DIR = Path("logs")
MAX_SESSIONS = 5


class JobSpyFilter(logging.Filter):
    """Drops log records originating from JobSpy or noisy network packages."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name.lower()
        return not (name.startswith("jobspy") or "jobspy" in name)


class SessionLogger:
    """Manages session-scoped file logging with strict rolling disk retention."""

    _instance: logging.Logger | None = None
    _session_path: Path | None = None

    @classmethod
    def setup(cls, verbose: bool = False, quiet: bool = False) -> logging.Logger:
        """Initializes a new session log file and binds stream handlers, or reconfigures existing ones."""
        if cls._instance is not None:
            # Reconfigure stream handler to respect updated verbose or quiet flags
            stream_handlers = [
                h
                for h in cls._instance.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            if quiet:
                for h in stream_handlers:
                    cls._instance.removeHandler(h)
            elif stream_handlers:
                for h in stream_handlers:
                    h.setLevel(logging.DEBUG if verbose else logging.ERROR)
            else:
                stderr_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
                stderr_handler = logging.StreamHandler(sys.stderr)
                stderr_handler.setLevel(logging.DEBUG if verbose else logging.ERROR)
                stderr_handler.setFormatter(stderr_formatter)
                stderr_handler.addFilter(JobSpyFilter())
                cls._instance.addHandler(stderr_handler)

            cls._instance.debug(f"Logger reconfigured: verbose={verbose}, quiet={quiet}")
            return cls._instance

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls._rotate_sessions(max_sessions=MAX_SESSIONS - 1)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls._session_path = LOGS_DIR / f"session_{session_id}.log"

        logger = logging.getLogger("cv_pipeline")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()

        file_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(cls._session_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        jobspy_filter = JobSpyFilter()
        logging.getLogger().addFilter(jobspy_filter)

        noisy_targets = [
            "JobSpy",
            "jobspy",
            "JobSpy:Linkedin",
            "JobSpy:LinkedIn",
            "JobSpy:Indeed",
            "JobSpy:Google",
            "JobSpy:ZipRecruiter",
            "JobSpy:Glassdoor",
            "JobSpy:Bayt",
            "urllib3",
            "selenium",
            "tls_client",
            "curl_cffi",
        ]
        for name in noisy_targets:
            lgr = logging.getLogger(name)
            lgr.setLevel(logging.CRITICAL)
            lgr.propagate = False
            lgr.handlers.clear()
            lgr.addFilter(jobspy_filter)

        if not quiet:
            stderr_formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setLevel(logging.DEBUG if verbose else logging.ERROR)
            stderr_handler.setFormatter(stderr_formatter)
            stderr_handler.addFilter(jobspy_filter)
            logger.addHandler(stderr_handler)

        cls._instance = logger
        cls._instance.debug(
            f"Session started. Log allocated at: {cls._session_path.resolve()} | "
            f"Python {sys.version.split()[0]} on {sys.platform}"
        )
        return cls._instance

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            return cls.setup()
        return cls._instance

    @classmethod
    def get_session_path(cls) -> Path | None:
        return cls._session_path

    @classmethod
    def fatal(
        cls, message: str, code: ExitCode = ExitCode.GENERAL_ERROR, exc: Exception | None = None
    ) -> None:
        logger = cls.get_logger()
        if exc:
            logger.exception(message)
        else:
            logger.error(message)

        session_file = cls.get_session_path()
        sys.stderr.write(f"\n[ERROR] {message}\n")
        if session_file:
            sys.stderr.write(
                f"[DIAGNOSTICS] Detailed trace available: {session_file.resolve()}\n\n"
            )
        sys.stderr.flush()
        sys.exit(code)

    @classmethod
    def _rotate_sessions(cls, max_sessions: int = 4) -> None:
        try:
            sessions = sorted(
                LOGS_DIR.glob("session_*.log"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_session in sessions[max_sessions:]:
                old_session.unlink(missing_ok=True)
        except OSError:
            pass


log = SessionLogger.get_logger()
