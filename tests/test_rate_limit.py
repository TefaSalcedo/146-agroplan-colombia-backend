"""Tests for configurable IP rate limiting."""

from app.middleware.rate_limit import IPRateLimiter


def test_ip_rate_limiter_enforces_window_limit():
    limiter = IPRateLimiter()

    first = limiter.check("127.0.0.1", "general", limit=2, window_seconds=60)
    second = limiter.check("127.0.0.1", "general", limit=2, window_seconds=60)
    blocked = limiter.check("127.0.0.1", "general", limit=2, window_seconds=60)

    assert first[0] is True
    assert first[1] == 1
    assert second[0] is True
    assert second[1] == 0
    assert blocked[0] is False
    assert blocked[1] == 0
    assert blocked[2] >= 1


def test_ip_rate_limiter_separates_ips_and_groups():
    limiter = IPRateLimiter()

    assert limiter.check("127.0.0.1", "general", limit=1, window_seconds=60)[0] is True
    assert limiter.check("127.0.0.2", "general", limit=1, window_seconds=60)[0] is True
    assert limiter.check("127.0.0.1", "ml_llm", limit=1, window_seconds=60)[0] is True
    assert limiter.check("127.0.0.1", "general", limit=1, window_seconds=60)[0] is False


def test_request_body_size_limit_rejects_oversized_payload(client):
    response = client.post(
        "/api/v1/calendars/predict-batch",
        content=b"x" * 65_537,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the allowed size"}
