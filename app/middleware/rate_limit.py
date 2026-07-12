"""Configurable in-memory IP rate limiting for HTTP endpoints."""

from collections import defaultdict, deque
from functools import wraps
from inspect import iscoroutinefunction
from threading import Lock
from time import monotonic
from typing import Callable, Deque, Dict, Optional, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

RATE_LIMIT_GROUP_ATTRIBUTE = "__rate_limit_group__"
ML_LLM_GROUP = "ml_llm"
GENERAL_GROUP = "general"


class IPRateLimiter:
    """Thread-safe fixed-window limiter shared by requests in one process."""

    def __init__(self) -> None:
        self._requests: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, ip: str, group: str, limit: int, window_seconds: int) -> Tuple[bool, int, int]:
        """Register a request and return allowed status, remaining count and retry seconds."""
        now = monotonic()
        key = (group, ip)
        with self._lock:
            timestamps = self._requests[key]
            cutoff = now - window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= limit:
                retry_after = max(1, int(timestamps[0] + window_seconds - now) + 1)
                return False, 0, retry_after

            timestamps.append(now)
            return True, max(0, limit - len(timestamps)), window_seconds

    def clear(self) -> None:
        """Clear all counters, primarily for tests."""
        with self._lock:
            self._requests.clear()


rate_limiter = IPRateLimiter()


def rate_limit(group: str = GENERAL_GROUP) -> Callable:
    """Mark an endpoint with a rate-limit group enforced by the decorator."""
    if group not in {GENERAL_GROUP, ML_LLM_GROUP}:
        raise ValueError(f"Unsupported rate-limit group: {group}")

    def decorator(endpoint: Callable) -> Callable:
        setattr(endpoint, RATE_LIMIT_GROUP_ATTRIBUTE, group)
        return endpoint

    return decorator


def _get_request(args: tuple, kwargs: dict) -> Optional[Request]:
    for value in (*args, *kwargs.values()):
        if isinstance(value, Request):
            return value
    return None


def _enforce(request: Request, group: str) -> Optional[JSONResponse]:
    settings = get_settings()
    if not settings.rate_limit_enabled:
        return None

    limit = (
        settings.rate_limit_ml_llm_per_minute
        if group == ML_LLM_GROUP
        else settings.rate_limit_general_per_minute
    )
    ip = request.client.host if request.client else "unknown"
    allowed, remaining, retry_after = rate_limiter.check(
        ip,
        group,
        limit,
        settings.rate_limit_window_seconds,
    )
    headers = {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "Retry-After": str(retry_after),
    }
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded",
                "retry_after_seconds": retry_after,
            },
            headers=headers,
        )
    return None


def enforce_rate_limit(endpoint: Callable) -> Callable:
    """Wrap an endpoint and enforce the group assigned with ``rate_limit``."""
    group = getattr(endpoint, RATE_LIMIT_GROUP_ATTRIBUTE, GENERAL_GROUP)
    if iscoroutinefunction(endpoint):
        @wraps(endpoint)
        async def async_wrapper(*args, **kwargs):
            request = _get_request(args, kwargs)
            if request is not None:
                response = _enforce(request, group)
                if response is not None:
                    return response
            return await endpoint(*args, **kwargs)

        setattr(async_wrapper, RATE_LIMIT_GROUP_ATTRIBUTE, group)
        return async_wrapper

    @wraps(endpoint)
    def sync_wrapper(*args, **kwargs):
        request = _get_request(args, kwargs)
        if request is not None:
            response = _enforce(request, group)
            if response is not None:
                return response
        return endpoint(*args, **kwargs)

    setattr(sync_wrapper, RATE_LIMIT_GROUP_ATTRIBUTE, group)
    return sync_wrapper


class GeneralRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply the general limit to requests not handled by a decorated endpoint."""

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return await call_next(request)

        response = _enforce(request, GENERAL_GROUP)
        if response is not None:
            return response
        response = await call_next(request)
        return response
