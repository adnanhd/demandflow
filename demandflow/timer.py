from __future__ import annotations

import time
from typing import Callable


class Timer:
    """Periodic callback timer driven by the executor."""

    def __init__(self, period: float, callback: Callable[[], None]) -> None:
        self._period = period
        self._callback = callback
        self._last_fire = 0.0

    @property
    def period(self) -> float:
        return self._period

    def ready(self) -> bool:
        return time.monotonic() - self._last_fire >= self._period

    def fire(self) -> None:
        self._last_fire = time.monotonic()
        self._callback()

    def reset(self) -> None:
        self._last_fire = time.monotonic()
