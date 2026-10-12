"""
Rate limiting dependencies module.

This module is deprecated. Use rate_limit_checker.py for FastAPI dependencies
that enforce rate limits using the DistributedRateLimiter.

All rate limiting is now handled by the DistributedRateLimiter in app.core.rate_limiter
with per-user, per-tenant, and per-IP policies configured via RateLimitConfig.
"""
