"""Client-side token-bucket rate limiter. Design: doc 02 §1, §8.

Rate limits are enforced client-side (one bucket per provider) so we never
depend on provider 429s for pacing. The bucket blocks (sleeps) by default;
``try_acquire`` supports non-blocking use.

Time and sleep are injectable for deterministic tests.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class TokenBucket:
    """Classic token bucket: ``rate`` tokens/second, burst up to ``capacity``."""

    def __init__(
        self,
        rate: float,
        capacity: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._sleep = sleep
        self._tokens = capacity
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` if immediately available; never blocks."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> None:
        """Block (sleep) until ``tokens`` are available, then take them."""
        if tokens > self.capacity:
            raise ValueError("requested more tokens than bucket capacity")
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self.rate
            self._sleep(wait)
