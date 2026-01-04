"""
API Security Middleware
Comprehensive security utilities for FastAPI/Flask applications
"""

from functools import wraps
from typing import Dict, Optional, Callable
import time
import hashlib
import secrets
from collections import defaultdict
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.
    Prevents abuse and ensures fair usage.
    """

    def __init__(self):
        self.buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": 0, "last_update": time.time()}
        )
        self.limits = {
            "default": {"requests": 100, "window": 60},  # 100 req/min
            "strict": {"requests": 10, "window": 60},  # 10 req/min
            "generous": {"requests": 1000, "window": 60},  # 1000 req/min
        }

    def _get_bucket_key(self, identifier: str, endpoint: str) -> str:
        """Generate unique bucket key for identifier and endpoint."""
        return f"{identifier}:{endpoint}"

    def _refill_tokens(self, bucket: Dict, max_tokens: int, refill_rate: float):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        tokens_to_add = elapsed * refill_rate
        bucket["tokens"] = min(max_tokens, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

    def check_rate_limit(
        self, identifier: str, endpoint: str, limit_type: str = "default"
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is within rate limit.

        Args:
            identifier: User identifier (IP, API key, user ID)
            endpoint: API endpoint being accessed
            limit_type: Type of rate limit to apply

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        limit_config = self.limits.get(limit_type, self.limits["default"])
        max_tokens = limit_config["requests"]
        window = limit_config["window"]
        refill_rate = max_tokens / window

        bucket_key = self._get_bucket_key(identifier, endpoint)
        bucket = self.buckets[bucket_key]

        # Refill tokens
        self._refill_tokens(bucket, max_tokens, refill_rate)

        # Check if request can be made
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, None
        else:
            # Calculate retry after
            tokens_needed = 1 - bucket["tokens"]
            retry_after = int(tokens_needed / refill_rate)
            return False, retry_after

    def reset_limit(self, identifier: str, endpoint: str):
        """Reset rate limit for specific identifier and endpoint."""
        bucket_key = self._get_bucket_key(identifier, endpoint)
        if bucket_key in self.buckets:
            del self.buckets[bucket_key]


class APIKeyManager:
    """
    Manages API keys for authentication and authorization.
    """

    def __init__(self):
        self.keys: Dict[str, Dict] = {}
        self.key_usage: Dict[str, int] = defaultdict(int)

    def generate_api_key(
        self,
        user_id: str,
        tier: str = "free",
        expires_in_days: Optional[int] = None,
    ) -> str:
        """
        Generate a new API key.

        Args:
            user_id: User identifier
            tier: API tier (free, pro, enterprise)
            expires_in_days: Optional expiration in days

        Returns:
            Generated API key
        """
        # Generate secure random key
        api_key = f"credx_{secrets.token_urlsafe(32)}"

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        # Store key metadata
        self.keys[api_key] = {
            "user_id": user_id,
            "tier": tier,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "is_active": True,
            "rate_limit": self._get_tier_rate_limit(tier),
        }

        logger.info(f"Generated API key for user {user_id} with tier {tier}")
        return api_key

    def _get_tier_rate_limit(self, tier: str) -> str:
        """Get rate limit type based on tier."""
        tier_limits = {"free": "strict", "pro": "default", "enterprise": "generous"}
        return tier_limits.get(tier, "strict")

    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate API key and return metadata.

        Args:
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, key_metadata)
        """
        if api_key not in self.keys:
            return False, None

        key_data = self.keys[api_key]

        # Check if key is active
        if not key_data["is_active"]:
            return False, {"error": "API key is inactive"}

        # Check expiration
        if key_data["expires_at"] and datetime.now() > key_data["expires_at"]:
            return False, {"error": "API key has expired"}

        # Track usage
        self.key_usage[api_key] += 1

        return True, key_data

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.keys:
            self.keys[api_key]["is_active"] = False
            logger.info(f"Revoked API key: {api_key[:20]}...")
            return True
        return False

    def get_usage_stats(self, api_key: str) -> Optional[Dict]:
        """Get usage statistics for an API key."""
        if api_key not in self.keys:
            return None

        return {
            "total_requests": self.key_usage[api_key],
            "tier": self.keys[api_key]["tier"],
            "created_at": self.keys[api_key]["created_at"],
            "is_active": self.keys[api_key]["is_active"],
        }


class SecurityValidator:
    """
    Validates and sanitizes input data for security.
    """

    @staticmethod
    def sanitize_input(data: str, max_length: int = 10000) -> str:
        """
        Sanitize input string.

        Args:
            data: Input string to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not isinstance(data, str):
            return ""

        # Truncate to max length
        data = data[:max_length]

        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "&", '"', "'", "/", "\\"]
        for char in dangerous_chars:
            data = data.replace(char, "")

        return data.strip()

    @staticmethod
    def validate_file_upload(
        filename: str, allowed_extensions: set = {"pdf", "doc", "docx"}
    ) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file.

        Args:
            filename: Name of uploaded file
            allowed_extensions: Set of allowed file extensions

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "No filename provided"

        # Check extension
        if "." not in filename:
            return False, "File has no extension"

        extension = filename.rsplit(".", 1)[1].lower()
        if extension not in allowed_extensions:
            return (
                False,
                f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
            )

        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Invalid filename"

        return True, None

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """
        Hash sensitive data for logging/storage.

        Args:
            data: Sensitive data to hash

        Returns:
            SHA256 hash of data
        """
        return hashlib.sha256(data.encode()).hexdigest()


# Global instances
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
security_validator = SecurityValidator()


def require_api_key(tier_required: str = "free"):
    """
    Decorator to require API key authentication.

    Args:
        tier_required: Minimum tier required (free, pro, enterprise)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a placeholder - actual implementation depends on framework
            # For Flask: from flask import request
            # For FastAPI: from fastapi import Header

            # Example implementation:
            # api_key = request.headers.get('X-API-Key')
            # is_valid, key_data = api_key_manager.validate_api_key(api_key)

            # if not is_valid:
            #     return {"error": "Invalid API key"}, 401

            # Check tier
            # tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}
            # if tier_hierarchy[key_data["tier"]] < tier_hierarchy[tier_required]:
            #     return {"error": "Insufficient API tier"}, 403

            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(limit_type: str = "default", endpoint: Optional[str] = None):
    """
    Decorator to apply rate limiting.

    Args:
        limit_type: Type of rate limit (default, strict, generous)
        endpoint: Optional endpoint name (defaults to function name)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is a placeholder - actual implementation depends on framework
            # identifier = request.remote_addr or "unknown"
            # endpoint_name = endpoint or func.__name__

            # is_allowed, retry_after = rate_limiter.check_rate_limit(
            #     identifier, endpoint_name, limit_type
            # )

            # if not is_allowed:
            #     return {
            #         "error": "Rate limit exceeded",
            #         "retry_after": retry_after
            #     }, 429

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions for framework integration


def get_client_identifier(request) -> str:
    """
    Get client identifier from request.
    Tries API key, then IP address.
    """
    # Try API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        is_valid, key_data = api_key_manager.validate_api_key(api_key)
        if is_valid:
            return f"key:{key_data['user_id']}"

    # Fall back to IP address
    return f"ip:{request.remote_addr or 'unknown'}"


def log_security_event(event_type: str, details: Dict):
    """
    Log security-related events.

    Args:
        event_type: Type of security event
        details: Event details
    """
    logger.warning(
        f"SECURITY EVENT: {event_type}",
        extra={"event_type": event_type, "details": details, "timestamp": datetime.now()},
    )
