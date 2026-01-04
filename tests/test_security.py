"""
Unit Tests for Security Module
Tests for rate limiting, API key management, and security validation
"""

import pytest
import time
from src.credx_ai.security import (
    RateLimiter,
    APIKeyManager,
    SecurityValidator,
    rate_limiter,
    api_key_manager,
    security_validator,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter()
        assert limiter is not None
        assert "default" in limiter.limits
        assert "strict" in limiter.limits
        assert "generous" in limiter.limits

    def test_rate_limit_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter()
        identifier = "test_user"
        endpoint = "/api/test"

        # First request should be allowed
        is_allowed, retry_after = limiter.check_rate_limit(identifier, endpoint, "strict")
        assert is_allowed is True
        assert retry_after is None

    def test_rate_limit_blocks_exceeding_limit(self):
        """Test that requests exceeding limit are blocked."""
        limiter = RateLimiter()
        identifier = "test_user_2"
        endpoint = "/api/test"

        # Make requests up to limit (10 for strict)
        for i in range(10):
            is_allowed, _ = limiter.check_rate_limit(identifier, endpoint, "strict")
            assert is_allowed is True

        # Next request should be blocked
        is_allowed, retry_after = limiter.check_rate_limit(identifier, endpoint, "strict")
        assert is_allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_rate_limit_different_endpoints(self):
        """Test that rate limits are per endpoint."""
        limiter = RateLimiter()
        identifier = "test_user_3"

        # Use up limit on endpoint1
        for i in range(10):
            limiter.check_rate_limit(identifier, "/api/endpoint1", "strict")

        # endpoint2 should still be available
        is_allowed, _ = limiter.check_rate_limit(identifier, "/api/endpoint2", "strict")
        assert is_allowed is True

    def test_rate_limit_reset(self):
        """Test rate limit reset functionality."""
        limiter = RateLimiter()
        identifier = "test_user_4"
        endpoint = "/api/test"

        # Use up limit
        for i in range(10):
            limiter.check_rate_limit(identifier, endpoint, "strict")

        # Reset limit
        limiter.reset_limit(identifier, endpoint)

        # Should be allowed again
        is_allowed, _ = limiter.check_rate_limit(identifier, endpoint, "strict")
        assert is_allowed is True


class TestAPIKeyManager:
    """Tests for APIKeyManager class."""

    def test_api_key_generation(self):
        """Test API key generation."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("user123", tier="free")

        assert api_key is not None
        assert api_key.startswith("credx_")
        assert len(api_key) > 20

    def test_api_key_validation_valid(self):
        """Test validation of valid API key."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("user123", tier="pro")

        is_valid, key_data = manager.validate_api_key(api_key)

        assert is_valid is True
        assert key_data is not None
        assert key_data["user_id"] == "user123"
        assert key_data["tier"] == "pro"
        assert key_data["is_active"] is True

    def test_api_key_validation_invalid(self):
        """Test validation of invalid API key."""
        manager = APIKeyManager()

        is_valid, key_data = manager.validate_api_key("invalid_key")

        assert is_valid is False
        assert key_data is None

    def test_api_key_revocation(self):
        """Test API key revocation."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("user123", tier="free")

        # Revoke key
        success = manager.revoke_api_key(api_key)
        assert success is True

        # Validation should fail
        is_valid, key_data = manager.validate_api_key(api_key)
        assert is_valid is False
        assert key_data["error"] == "API key is inactive"

    def test_api_key_expiration(self):
        """Test API key expiration."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("user123", tier="free", expires_in_days=-1)

        # Should be expired
        is_valid, key_data = manager.validate_api_key(api_key)
        assert is_valid is False
        assert "expired" in key_data["error"].lower()

    def test_api_key_usage_tracking(self):
        """Test API key usage tracking."""
        manager = APIKeyManager()
        api_key = manager.generate_api_key("user123", tier="free")

        # Make some requests
        for i in range(5):
            manager.validate_api_key(api_key)

        # Check usage stats
        stats = manager.get_usage_stats(api_key)
        assert stats is not None
        assert stats["total_requests"] == 5

    def test_api_key_tier_rate_limits(self):
        """Test that different tiers have different rate limits."""
        manager = APIKeyManager()

        free_key = manager.generate_api_key("user1", tier="free")
        pro_key = manager.generate_api_key("user2", tier="pro")
        enterprise_key = manager.generate_api_key("user3", tier="enterprise")

        _, free_data = manager.validate_api_key(free_key)
        _, pro_data = manager.validate_api_key(pro_key)
        _, enterprise_data = manager.validate_api_key(enterprise_key)

        assert free_data["rate_limit"] == "strict"
        assert pro_data["rate_limit"] == "default"
        assert enterprise_data["rate_limit"] == "generous"


class TestSecurityValidator:
    """Tests for SecurityValidator class."""

    def test_sanitize_input_removes_dangerous_chars(self):
        """Test that dangerous characters are removed."""
        validator = SecurityValidator()

        dangerous_input = '<script>alert("xss")</script>'
        sanitized = validator.sanitize_input(dangerous_input)

        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "script" in sanitized  # Text remains, tags removed

    def test_sanitize_input_truncates_long_strings(self):
        """Test that long strings are truncated."""
        validator = SecurityValidator()

        long_input = "a" * 20000
        sanitized = validator.sanitize_input(long_input, max_length=1000)

        assert len(sanitized) == 1000

    def test_sanitize_input_handles_non_string(self):
        """Test that non-string input is handled."""
        validator = SecurityValidator()

        result = validator.sanitize_input(12345)
        assert result == ""

    def test_validate_file_upload_valid(self):
        """Test validation of valid file upload."""
        validator = SecurityValidator()

        is_valid, error = validator.validate_file_upload("resume.pdf")

        assert is_valid is True
        assert error is None

    def test_validate_file_upload_invalid_extension(self):
        """Test validation of invalid file extension."""
        validator = SecurityValidator()

        is_valid, error = validator.validate_file_upload("malicious.exe")

        assert is_valid is False
        assert "Invalid file type" in error

    def test_validate_file_upload_path_traversal(self):
        """Test detection of path traversal attempts."""
        validator = SecurityValidator()

        is_valid, error = validator.validate_file_upload("../../etc/passwd")

        assert is_valid is False
        assert "Invalid filename" in error

    def test_validate_file_upload_no_extension(self):
        """Test validation of file without extension."""
        validator = SecurityValidator()

        is_valid, error = validator.validate_file_upload("noextension")

        assert is_valid is False
        assert "no extension" in error

    def test_hash_sensitive_data(self):
        """Test hashing of sensitive data."""
        validator = SecurityValidator()

        data = "sensitive_password_123"
        hashed = validator.hash_sensitive_data(data)

        assert hashed != data
        assert len(hashed) == 64  # SHA256 produces 64 character hex string

        # Same input should produce same hash
        hashed2 = validator.hash_sensitive_data(data)
        assert hashed == hashed2


class TestIntegration:
    """Integration tests for security module."""

    def test_rate_limiter_with_api_key(self):
        """Test rate limiter integration with API key manager."""
        limiter = RateLimiter()
        manager = APIKeyManager()

        # Create API key
        api_key = manager.generate_api_key("user123", tier="free")
        is_valid, key_data = manager.validate_api_key(api_key)

        # Use rate limiter with user ID from API key
        identifier = f"key:{key_data['user_id']}"
        endpoint = "/api/test"
        limit_type = key_data["rate_limit"]

        # Should be allowed
        is_allowed, _ = limiter.check_rate_limit(identifier, endpoint, limit_type)
        assert is_allowed is True

    def test_security_validator_with_file_upload(self):
        """Test security validator with file upload scenario."""
        validator = SecurityValidator()

        # Valid file
        is_valid, error = validator.validate_file_upload("resume.pdf", {"pdf", "doc", "docx"})
        assert is_valid is True

        # Invalid file
        is_valid, error = validator.validate_file_upload("malware.exe", {"pdf", "doc", "docx"})
        assert is_valid is False


# Fixtures


@pytest.fixture
def clean_rate_limiter():
    """Provide a clean rate limiter instance."""
    return RateLimiter()


@pytest.fixture
def clean_api_key_manager():
    """Provide a clean API key manager instance."""
    return APIKeyManager()


@pytest.fixture
def security_validator_instance():
    """Provide a security validator instance."""
    return SecurityValidator()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
