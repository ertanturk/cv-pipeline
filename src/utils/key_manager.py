"""Secure credentials management interface for Gemini API access."""

from __future__ import annotations

import os

import keyring
import keyring.errors

from src.utils.logger import log

KEYRING_SERVICE_NAME = "cv_pipeline"
KEYRING_USERNAME = "gemini_api_key"


class KeyManager:
    """Manages Gemini API key resolution and persistent storage across OS keyrings and environment variables.

    Key Resolution Precedence:
    1. Operating System Keyring (service='cv_pipeline', username='gemini_api_key')
    2. GEMINI_API_KEY environment variable
    """

    @classmethod
    def get_key(cls) -> str | None:
        """Retrieves the active Gemini API key from the OS keyring or environment.

        Returns:
            The trimmed API key string if found, otherwise None.
        """
        # Attempt OS keyring lookup first
        try:
            stored_key = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
            if stored_key and stored_key.strip():
                log.debug("Gemini API key resolved from OS keyring.")
                return stored_key.strip()
        except Exception as exc:
            log.debug(f"OS keyring unavailable or query failed: {exc}")

        # Fallback to environment variable
        env_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if env_key:
            log.debug("Gemini API key resolved from GEMINI_API_KEY environment variable.")
            return env_key

        log.debug(
            "No Gemini API key found in either OS keyring or GEMINI_API_KEY environment variable."
        )
        return None

    @classmethod
    def set_key(cls, key: str) -> None:
        """Stores or updates the Gemini API key in the OS keyring.

        Args:
            key: The raw API key string to persist.

        Raises:
            ValueError: If the provided key is blank.
            RuntimeError: If the OS keyring service encounters a fatal error.
        """
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("API key cannot be empty.")

        try:
            keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, clean_key)
            log.info("Gemini API key successfully stored in OS keyring.")
        except Exception as exc:
            log.error(f"Failed to store Gemini API key in OS keyring: {exc}")
            raise RuntimeError(f"Failed to store Gemini API key in OS keyring: {exc}") from exc

    @classmethod
    def delete_key(cls) -> None:
        """Removes the stored API key from the OS keyring service.

        Raises:
            RuntimeError: If an unexpected error occurs during keyring deletion.
        """
        try:
            keyring.delete_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
            log.info("Gemini API key deleted from OS keyring.")
        except keyring.errors.PasswordDeleteError:
            log.warning("No Gemini API key found in OS keyring to delete.")
        except Exception as exc:
            log.error(f"Failed to delete Gemini API key from OS keyring: {exc}")
            raise RuntimeError(f"Failed to delete Gemini API key from OS keyring: {exc}") from exc
