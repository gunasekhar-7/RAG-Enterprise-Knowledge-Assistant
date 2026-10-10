import time

import structlog
from fastapi import HTTPException, Request

logger = structlog.get_logger(__name__)


class RateLimiter:
    """
    Basic in-memory rate limiter for the API.
    In a production enterprise environment, this would use Redis.
    """

    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        # IP -> List of timestamps
        self._history = {}  # type: ignore

    def __call__(self, request: Request):  # type: ignore
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        if client_ip not in self._history:
            self._history[client_ip] = []

        # Clean old requests
        self._history[client_ip] = [t for t in self._history[client_ip] if now - t < 60]

        if len(self._history[client_ip]) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded", ip=client_ip)
            raise HTTPException(status_code=429, detail="Too Many Requests")

        self._history[client_ip].append(now)
