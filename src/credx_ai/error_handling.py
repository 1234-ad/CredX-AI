"""
Error Handling and Logging
Comprehensive error handling with structured logging
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import json


class CustomFormatter(logging.Formatter):
    """
    Custom formatter with colors for console output.
    """

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class StructuredLogger:
    """
    Structured logger with JSON output support.
    """

    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        console_handler.setFormatter(CustomFormatter(console_format))
        self.logger.addHandler(console_handler)

        # File handler with JSON format
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(file_handler)

    def _add_context(self, message: str, context: Optional[Dict] = None) -> str:
        """Add context to log message."""
        if context:
            return f"{message} | Context: {json.dumps(context)}"
        return message

    def debug(self, message: str, context: Optional[Dict] = None):
        """Log debug message."""
        self.logger.debug(self._add_context(message, context))

    def info(self, message: str, context: Optional[Dict] = None):
        """Log info message."""
        self.logger.info(self._add_context(message, context))

    def warning(self, message: str, context: Optional[Dict] = None):
        """Log warning message."""
        self.logger.warning(self._add_context(message, context))

    def error(self, message: str, context: Optional[Dict] = None, exc_info: bool = False):
        """Log error message."""
        self.logger.error(self._add_context(message, context), exc_info=exc_info)

    def critical(self, message: str, context: Optional[Dict] = None, exc_info: bool = False):
        """Log critical message."""
        self.logger.critical(self._add_context(message, context), exc_info=exc_info)


class AppError(Exception):
    """
    Base application error class.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict] = None,
    ):
        """
        Initialize application error.

        Args:
            message: Error message
            error_code: Error code for identification
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)

    def to_dict(self) -> Dict:
        """Convert error to dictionary."""
        return {
            "error": {
                "message": self.message,
                "code": self.error_code,
                "details": self.details,
                "timestamp": self.timestamp,
            }
        }


class ValidationError(AppError):
    """Validation error."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "VALIDATION_ERROR", 400, details)


class AuthenticationError(AppError):
    """Authentication error."""

    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", 401, details)


class AuthorizationError(AppError):
    """Authorization error."""

    def __init__(self, message: str = "Access denied", details: Optional[Dict] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", 403, details)


class NotFoundError(AppError):
    """Resource not found error."""

    def __init__(self, message: str = "Resource not found", details: Optional[Dict] = None):
        super().__init__(message, "NOT_FOUND", 404, details)


class RateLimitError(AppError):
    """Rate limit exceeded error."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None
    ):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429, details)


class ExternalServiceError(AppError):
    """External service error."""

    def __init__(self, service: str, message: str, details: Optional[Dict] = None):
        details = details or {}
        details["service"] = service
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", 502, details)


class ModelError(AppError):
    """AI model error."""

    def __init__(self, message: str, model_name: str, details: Optional[Dict] = None):
        details = details or {}
        details["model"] = model_name
        super().__init__(message, "MODEL_ERROR", 500, details)


class ErrorHandler:
    """
    Centralized error handler.
    """

    def __init__(self, logger: StructuredLogger):
        """
        Initialize error handler.

        Args:
            logger: Structured logger instance
        """
        self.logger = logger
        self.error_counts = {}

    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict:
        """
        Handle error and return formatted response.

        Args:
            error: Exception to handle
            context: Additional context

        Returns:
            Error response dictionary
        """
        # Track error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error
        if isinstance(error, AppError):
            self.logger.error(
                f"{error.error_code}: {error.message}",
                context={
                    "error_code": error.error_code,
                    "status_code": error.status_code,
                    "details": error.details,
                    **(context or {}),
                },
            )
            return error.to_dict(), error.status_code
        else:
            # Unexpected error
            self.logger.critical(
                f"Unexpected error: {str(error)}",
                context={"error_type": error_type, **(context or {})},
                exc_info=True,
            )
            return {
                "error": {
                    "message": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "timestamp": datetime.now().isoformat(),
                }
            }, 500

    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts,
        }


def handle_exceptions(logger: StructuredLogger):
    """
    Decorator to handle exceptions in functions.

    Args:
        logger: Structured logger instance
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError as e:
                logger.error(
                    f"{e.error_code}: {e.message}",
                    context={"error_code": e.error_code, "details": e.details},
                )
                raise
            except Exception as e:
                logger.critical(
                    f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True
                )
                raise AppError(
                    message="An unexpected error occurred",
                    error_code="INTERNAL_ERROR",
                    details={"function": func.__name__, "error": str(e)},
                )

        return wrapper

    return decorator


def log_function_call(logger: StructuredLogger):
    """
    Decorator to log function calls.

    Args:
        logger: Structured logger instance
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__}",
                context={"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise

        return wrapper

    return decorator


def validate_input(validation_func):
    """
    Decorator to validate function input.

    Args:
        validation_func: Function to validate input
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input
            is_valid, error_message = validation_func(*args, **kwargs)

            if not is_valid:
                raise ValidationError(error_message)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Global logger instance
app_logger = StructuredLogger("credx_ai", log_file="credx_ai.log")
error_handler = ErrorHandler(app_logger)


# Utility functions


def log_api_request(endpoint: str, method: str, params: Dict, user_id: Optional[str] = None):
    """
    Log API request.

    Args:
        endpoint: API endpoint
        method: HTTP method
        params: Request parameters
        user_id: Optional user identifier
    """
    app_logger.info(
        f"API Request: {method} {endpoint}",
        context={
            "endpoint": endpoint,
            "method": method,
            "params": params,
            "user_id": user_id,
        },
    )


def log_api_response(
    endpoint: str, status_code: int, response_time: float, cache_hit: bool = False
):
    """
    Log API response.

    Args:
        endpoint: API endpoint
        status_code: HTTP status code
        response_time: Response time in seconds
        cache_hit: Whether response was from cache
    """
    app_logger.info(
        f"API Response: {endpoint} - {status_code}",
        context={
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": f"{response_time:.3f}s",
            "cache_hit": cache_hit,
        },
    )


def log_model_inference(model_name: str, input_size: int, inference_time: float):
    """
    Log model inference.

    Args:
        model_name: Name of the model
        input_size: Size of input data
        inference_time: Inference time in seconds
    """
    app_logger.info(
        f"Model Inference: {model_name}",
        context={
            "model": model_name,
            "input_size": input_size,
            "inference_time": f"{inference_time:.3f}s",
        },
    )
