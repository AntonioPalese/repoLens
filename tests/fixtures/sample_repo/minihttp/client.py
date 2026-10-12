"""HTTP client with configurable retries and timeouts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

DEFAULT_TIMEOUT = 5.0
RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """Decides whether and when a failed request is retried."""

    max_retries: int = 3
    backoff_factor: float = 0.5
    status_codes: frozenset[int] = field(default=RETRYABLE_STATUS_CODES)

    def should_retry(self, attempt: int, status_code: int) -> bool:
        """Return True if another attempt is allowed for this status code."""
        return attempt < self.max_retries and status_code in self.status_codes

    def backoff(self, attempt: int) -> float:
        """Exponential backoff delay in seconds before the next attempt."""
        return self.backoff_factor * (2**attempt)


class Client:
    """Sends HTTP requests, retrying according to a RetryPolicy."""

    user_agent = "minihttp/0.0.1"

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_policy = RetryPolicy()

    @property
    def base_url(self) -> str:
        """Base URL every relative request path is joined to."""
        return self._base_url

    # Retries are handled here rather than in the transport so that the
    # policy can see the final status code.
    def send(self, method: str, path: str) -> int:
        """Send a request and return the final status code."""

        def build_url(p: str) -> str:
            return f"{self._base_url}/{p.lstrip('/')}"

        url = build_url(path)
        attempt = 0
        while True:
            status = self._transport(method, url)
            if not self.retry_policy.should_retry(attempt, status):
                return status
            time.sleep(self.retry_policy.backoff(attempt))
            attempt += 1

    def get(self, path: str) -> int:
        """Shortcut for ``send("GET", path)``."""
        return self.send("GET", path)

    async def aget(self, path: str) -> int:
        """Async variant of :meth:`get` (synchronous under the hood)."""
        return self.get(path)

    def _transport(self, method: str, url: str) -> int:
        return 200

    class Config:
        """Per-client options."""

        follow_redirects = True

        def as_dict(self) -> dict[str, bool]:
            return {"follow_redirects": self.follow_redirects}

    max_redirects = 20
