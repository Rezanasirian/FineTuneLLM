#!/usr/bin/env python3
"""
src/api/middleware.py
API middleware for authentication and rate limiting
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Basic API key authentication middleware."""

    def __init__(self, app, api_keys: Dict[str, str]):
        super().__init__(app)
        self.api_keys = api_keys  # {user_id: api_key}

    async def dispatch(self, request: Request, call_next):
        """Validate API key in headers."""
        api_key = request.headers.get("X-API-Key")
        user_id = None

        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        for uid, key in self.api_keys.items():
            if key == api_key:
                user_id = uid
                break

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Add user_id to request state for later use
        request.state.user_id = user_id
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts: Dict[str, list] = {}  # {user_id: [timestamps]}

    async def dispatch(self, request: Request, call_next):
        """Enforce rate limits per user."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")

        current_time = time.time()

        # Clean old requests
        self.request_counts.setdefault(user_id, [])
        self.request_counts[user_id] = [
            t for t in self.request_counts[user_id]
            if current_time - t < self.window_seconds
        ]

        if len(self.request_counts[user_id]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s"
            )

        self.request_counts[user_id].append(current_time)
        return await call_next(request)