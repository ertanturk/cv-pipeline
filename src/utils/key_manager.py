"""Key management utilities for cv-pipeline."""

import keyring
import keyring.errors

from src.utils.exit_codes import ExitCode
from src.utils.logger import SessionLogger, log

KEYRING_SERVICE_NAME = "cv_pipeline"
KEYRING_USERNAME = "gemini_api_key"


class KeyManager:
    """Handles secure OS keyring storage and retrieval for Gemini API keys."""

    @classmethod
    def get_key(cls) -> str | None:
        """Retrieves key from keyring; falls back to GEMINI_API_KEY environment variable."""
        try:
            stored_key = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
            if stored_key and stored_key.strip():
                return stored_key.strip()
        except Exception as e:
            SessionLogger.fatal(
                "Gemini API key not found.",
                code=ExitCode.CONFIG_ERROR,
            )
            raise RuntimeError("Unreachable") from e

    @classmethod
    def set_key(cls, key: str) -> None:
        """Saves or updates the Gemini API key in the OS keyring."""
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("API key cannot be empty.")
        try:
            keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, clean_key)
        except Exception as e:
            SessionLogger.fatal(
                "Failed to store Gemini API key in OS keyring.",
                code=ExitCode.CONFIG_ERROR,
            )
            raise RuntimeError("Unreachable") from e
        log.info("Gemini API key successfully stored in OS keyring.")

    @classmethod
    def delete_key(cls) -> None:
        """Deletes the stored API key from the OS keyring."""
        try:
            keyring.delete_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
            log.info("Gemini API key deleted from OS keyring.")
        except keyring.errors.PasswordDeleteError:
            log.warning("No API key found in keyring to delete.")
        except Exception as e:
            SessionLogger.fatal(
                "Failed to delete Gemini API key from OS keyring.",
                code=ExitCode.CONFIG_ERROR,
            )
            raise RuntimeError("Unreachable") from e
