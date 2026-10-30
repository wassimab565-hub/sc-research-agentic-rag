import httpx
import pytest

from sc_research_agentic_rag.ingestion.http import MAX_BACKOFF, RateLimiter, get_with_retry


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responses: list) -> tuple[httpx.Client, list[httpx.Request]]:
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def fetch(client: httpx.Client, fake: FakeTime, **kwargs) -> httpx.Response:
    return get_with_retry(client, "https://example.org", params={}, headers={}, sleep=fake.sleep, **kwargs)


def test_retries_then_succeeds_with_exponential_backoff() -> None:
    fake = FakeTime()
    client, calls = make_client([httpx.Response(503), httpx.Response(429), httpx.Response(200)])
    assert fetch(client, fake).status_code == 200
    assert len(calls) == 3
    assert fake.sleeps == [1.0, 2.0]


def test_retry_after_header_is_honored() -> None:
    fake = FakeTime()
    client, _ = make_client([httpx.Response(429, headers={"Retry-After": "5"}), httpx.Response(200)])
    fetch(client, fake)
    assert fake.sleeps == [5.0]


def test_backoff_is_capped() -> None:
    fake = FakeTime()
    client, _ = make_client([httpx.Response(429, headers={"Retry-After": "9999"}), httpx.Response(200)])
    fetch(client, fake)
    assert fake.sleeps == [MAX_BACKOFF]


def test_gives_up_after_max_retries() -> None:
    fake = FakeTime()
    client, calls = make_client([httpx.Response(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        fetch(client, fake, max_retries=2)
    assert len(calls) == 3


def test_non_retryable_status_fails_immediately() -> None:
    fake = FakeTime()
    client, calls = make_client([httpx.Response(404)])
    with pytest.raises(httpx.HTTPStatusError):
        fetch(client, fake)
    assert len(calls) == 1
    assert fake.sleeps == []


def test_transport_error_is_retried() -> None:
    fake = FakeTime()
    client, calls = make_client([httpx.ConnectError("boom"), httpx.Response(200)])
    assert fetch(client, fake).status_code == 200
    assert len(calls) == 2


def test_transport_error_raised_after_max_retries() -> None:
    fake = FakeTime()
    client, _ = make_client([httpx.ConnectError("boom")] * 2)
    with pytest.raises(httpx.ConnectError):
        fetch(client, fake, max_retries=1)


def test_before_request_runs_on_every_attempt() -> None:
    fake = FakeTime()
    client, _ = make_client([httpx.Response(503), httpx.Response(200)])
    count = []
    fetch(client, fake, before_request=lambda: count.append(1))
    assert len(count) == 2


def test_rate_limiter_spaces_requests() -> None:
    fake = FakeTime()
    limiter = RateLimiter(3.0, clock=fake.clock, sleep=fake.sleep)
    limiter.wait()
    assert fake.sleeps == []
    fake.now += 1.0
    limiter.wait()
    assert fake.sleeps == [2.0]
    fake.now += 10.0
    limiter.wait()
    assert fake.sleeps == [2.0]