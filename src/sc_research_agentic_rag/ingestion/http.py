import time
from collections.abc import Callable

import httpx

from sc_research_agentic_rag import __version__

USER_AGENT = f"sc-research-agentic-rag/{__version__}"
RETRY_STATUS = frozenset({429, 502, 503, 504})
MAX_BACKOFF = 30.0


def default_sleep(seconds: float) -> None:
    time.sleep(seconds)


class RateLimiter:
    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self._min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict,
    headers: dict,
    max_retries: int = 3,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = default_sleep,
    before_request: Callable[[], None] | None = None,
) -> httpx.Response:
    attempt = 0
    while True:
        if before_request:
            before_request()
        retry_after: float | None = None
        try:
            response = client.get(url, params=params, headers=headers)
        except httpx.TransportError:
            if attempt >= max_retries:
                raise
        else:
            if response.status_code not in RETRY_STATUS or attempt >= max_retries:
                response.raise_for_status()
                return response
            retry_after = _retry_after(response)

        delay = retry_after if retry_after is not None else backoff * 2**attempt
        sleep(min(delay, MAX_BACKOFF))
        attempt += 1


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    return float(value) if value.isdigit() else None