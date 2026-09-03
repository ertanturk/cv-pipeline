"""Thread-safe rate limiter enforcing per-model RPM and RPD budgets."""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from threading import Lock

from src.utils.logger import log


@dataclass(frozen=True)
class ModelLimits:
    rpm: int
    rpd: int


MODEL_LIMITS: dict[str, ModelLimits] = {
    "gemini-2.5-flash-lite": ModelLimits(rpm=15, rpd=500),
    "gemini-3.5-flash-lite": ModelLimits(rpm=15, rpd=500),
    "gemini-3.5-flash": ModelLimits(rpm=5, rpd=20),
    "gemini-3.6-flash": ModelLimits(rpm=5, rpd=20),
}

_DEFAULT_LIMITS = ModelLimits(rpm=5, rpd=20)


class BudgetExhaustedError(Exception):
    """Raised when the daily request budget for an LLM model is exhausted or when aborted."""


class RateLimiter:
    """Enforces per-model RPM and RPD budgets with thread safety."""

    def __init__(self, model: str) -> None:
        limits: ModelLimits = MODEL_LIMITS.get(model, _DEFAULT_LIMITS)
        self._model: str = model
        self._rpm_interval: float = 60.0 / max(1, limits.rpm)
        self._rpd_limit: int = limits.rpd
        self._last_call: float = 0.0
        self._day_start: float = self._today_midnight()
        self._day_count: int = 0
        self._aborted: bool = False
        self._lock: Lock = Lock()

    @staticmethod
    def _today_midnight() -> float:
        today: datetime.date = datetime.date.today()
        return datetime.datetime(today.year, today.month, today.day).timestamp()

    def abort(self) -> None:
        """Signals all future acquire() calls to fail immediately."""
        with self._lock:
            self._aborted = True

    def acquire(self) -> None:
        """Blocks until safe to send. Raises BudgetExhaustedError if budget exhausted or aborted."""
        with self._lock:
            if self._aborted:
                raise BudgetExhaustedError(
                    f"Requests to '{self._model}' aborted due to a prior fatal API error."
                )

            now: float = time.time()

            midnight: float = self._today_midnight()
            if midnight > self._day_start:
                self._day_start = midnight
                self._day_count = 0
                self._aborted = False

            if self._day_count >= self._rpd_limit:
                reset_in: float = (self._day_start + 86400) - now
                raise BudgetExhaustedError(
                    f"Daily request budget exhausted for '{self._model}' "
                    f"({self._rpd_limit} RPD). Quota resets in {reset_in / 3600:.1f}h."
                )

            elapsed: float = now - self._last_call
            if elapsed < self._rpm_interval:
                sleep_for: float = self._rpm_interval - elapsed
                log.debug(
                    f"Rate limiter: sleeping {sleep_for:.2f}s "
                    f"(model={self._model}, RPM limit={60 / self._rpm_interval:.0f})"
                )
                time.sleep(sleep_for)

            self._last_call = time.time()
            self._day_count += 1
            log.debug(
                f"Rate limiter: request {self._day_count}/{self._rpd_limit} RPD for '{self._model}'"
            )


_REGISTRY: dict[str, RateLimiter] = {}
_REGISTRY_LOCK: Lock = Lock()


def get_limiter(model: str) -> RateLimiter:
    """Returns the process-wide singleton RateLimiter for a given model."""
    with _REGISTRY_LOCK:
        if model not in _REGISTRY:
            _REGISTRY[model] = RateLimiter(model)
        return _REGISTRY[model]
