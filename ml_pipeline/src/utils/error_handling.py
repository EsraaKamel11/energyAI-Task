"""
Comprehensive error handling and retry mechanisms for the ML pipeline
Provides decorators, fallback strategies, and robust error recovery
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum
import traceback
import json
import os

# Retry mechanisms
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        after_log,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    import backoff

    BACKOFF_AVAILABLE = True
except ImportError:
    BACKOFF_AVAILABLE = False


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Structured error information"""

    error_type: str
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any]
    timestamp: float
    retry_count: int = 0
    fallback_used: bool = False


class ErrorHandler:
    """Centralized error handling and logging"""

    def __init__(self, logger_name: str = "ErrorHandler"):
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[ErrorInfo] = []
        self.fallback_strategies: Dict[str, Callable] = {}

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> ErrorInfo:
        """Log error with structured information"""

        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            context=context,
            timestamp=time.time(),
        )

        self.error_history.append(error_info)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"CRITICAL ERROR: {error_info.error_type} - {error_info.message}"
            )
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"HIGH SEVERITY: {error_info.error_type} - {error_info.message}"
            )
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"MEDIUM SEVERITY: {error_info.error_type} - {error_info.message}"
            )
        else:
            self.logger.info(
                f"LOW SEVERITY: {error_info.error_type} - {error_info.message}"
            )

        return error_info

    def register_fallback(self, error_type: str, fallback_func: Callable):
        """Register a fallback strategy for a specific error type"""
        self.fallback_strategies[error_type] = fallback_func
        self.logger.info(f"Registered fallback for {error_type}")

    def get_fallback(self, error_type: str) -> Optional[Callable]:
        """Get fallback strategy for error type"""
        return self.fallback_strategies.get(error_type)

    def save_error_report(self, filepath: str = "error_report.json"):
        """Save error history to file"""
        try:
            with open(filepath, "w") as f:
                json.dump(
                    [
                        {
                            "error_type": e.error_type,
                            "message": e.message,
                            "severity": e.severity.value,
                            "context": e.context,
                            "timestamp": e.timestamp,
                            "retry_count": e.retry_count,
                            "fallback_used": e.fallback_used,
                        }
                        for e in self.error_history
                    ],
                    f,
                    indent=2,
                )
            self.logger.info(f"Error report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")


# Global error handler instance
error_handler = ErrorHandler()


def retry_with_fallback(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    fallback_func: Optional[Callable] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
):
    """
    Decorator that retries a function and falls back to alternative implementation

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to retry on
        fallback_func: Fallback function to call if all retries fail
        severity: Error severity level for logging
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            # Try the main function with retries
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "args": str(args),
                        "kwargs": str(kwargs),
                    }

                    error_info = error_handler.log_error(e, context, severity)
                    error_info.retry_count = attempt + 1

                    if attempt < max_attempts - 1:
                        wait_time = 2**attempt  # Exponential backoff
                        error_handler.logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)

            # All retries failed, try fallback
            if fallback_func:
                error_handler.logger.warning(
                    f"All retries failed for {func.__name__}, using fallback"
                )
                try:
                    result = fallback_func(*args, **kwargs)
                    error_info.fallback_used = True
                    error_handler.logger.info(f"Fallback succeeded for {func.__name__}")
                    return result
                except Exception as fallback_error:
                    error_handler.log_error(
                        fallback_error,
                        {
                            "function": f"{func.__name__}_fallback",
                            "original_error": str(last_error),
                        },
                        ErrorSeverity.HIGH,
                    )

            # No fallback or fallback failed, raise the last error
            raise last_error

        return wrapper

    return decorator


def tenacity_retry(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    wait_strategy: Optional[Callable] = None,
):
    """
    Tenacity-based retry decorator with advanced retry strategies

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to retry on
        wait_strategy: Custom wait strategy function
    """
    if not TENACITY_AVAILABLE:
        return retry_with_fallback(max_attempts, exceptions)

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_strategy or wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(
                logging.getLogger(func.__module__), logging.WARNING
            ),
            after=after_log(logging.getLogger(func.__module__), logging.INFO),
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def backoff_retry(
    max_attempts: int = 3, exceptions: tuple = (Exception,), backoff_factor: float = 2.0
):
    """
    Backoff-based retry decorator

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exceptions to retry on
        backoff_factor: Backoff multiplier
    """
    if not BACKOFF_AVAILABLE:
        return retry_with_fallback(max_attempts, exceptions)

    def decorator(func: Callable) -> Callable:
        @backoff.on_exception(
            backoff.expo, exceptions, max_tries=max_attempts, factor=backoff_factor
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def safe_execute(
    fallback_value: Any = None,
    exceptions: tuple = (Exception,),
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_error: bool = True,
):
    """
    Decorator that safely executes a function and returns fallback value on error

    Args:
        fallback_value: Value to return if function fails
        exceptions: Tuple of exceptions to catch
        severity: Error severity level
        log_error: Whether to log the error
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    context = {
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs),
                    }
                    error_handler.log_error(e, context, severity)

                return fallback_value

        return wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    exceptions: tuple = (Exception,),
):
    """
    Circuit breaker pattern decorator

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before trying to close circuit
        exceptions: Exceptions that count as failures
    """

    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        circuit_open = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, circuit_open

            # Check if circuit is open
            if circuit_open:
                if time.time() - last_failure_time > recovery_timeout:
                    circuit_open = False
                    failure_count = 0
                    error_handler.logger.info(
                        f"Circuit breaker for {func.__name__} closed"
                    )
                else:
                    raise Exception(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                # Reset failure count on success
                failure_count = 0
                return result
            except exceptions as e:
                failure_count += 1
                last_failure_time = time.time()

                if failure_count >= failure_threshold:
                    circuit_open = True
                    error_handler.logger.error(
                        f"Circuit breaker opened for {func.__name__} after {failure_count} failures"
                    )

                raise e

        return wrapper

    return decorator


# Specific error handling for common pipeline operations


@retry_with_fallback(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
def robust_web_request(url: str, timeout: int = 30, **kwargs) -> str:
    """Robust web request with retry and fallback"""
    import requests

    response = requests.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.text


@retry_with_fallback(max_attempts=2, exceptions=(Exception,))
def robust_pdf_extraction(pdf_path: str, **kwargs) -> Dict[str, Any]:
    """Robust PDF extraction with fallback strategies"""
    from data_collection.pdf_extractor import PDFExtractor

    extractor = PDFExtractor(**kwargs)
    return extractor.extract_single_pdf(pdf_path)


def robust_file_operation(operation: str = "read"):
    """
    Decorator for robust file operations with multiple fallback strategies
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(filepath: str, *args, **kwargs):
            # Try primary operation
            try:
                return func(filepath, *args, **kwargs)
            except FileNotFoundError:
                # Try alternative paths
                alt_paths = [
                    os.path.join("data", filepath),
                    os.path.join("documents", filepath),
                    os.path.join("samples", filepath),
                ]

                for alt_path in alt_paths:
                    try:
                        return func(alt_path, *args, **kwargs)
                    except FileNotFoundError:
                        continue

                raise FileNotFoundError(f"File not found: {filepath}")
            except PermissionError:
                # Try with different permissions
                try:
                    os.chmod(filepath, 0o644)
                    return func(filepath, *args, **kwargs)
                except Exception:
                    raise PermissionError(f"Permission denied: {filepath}")
            except Exception as e:
                error_handler.log_error(
                    e,
                    {"operation": operation, "filepath": filepath},
                    ErrorSeverity.MEDIUM,
                )
                raise

        return wrapper

    return decorator


# Error recovery utilities


def recover_from_error(error: Exception, context: Dict[str, Any]) -> bool:
    """Attempt to recover from an error based on context"""

    error_type = type(error).__name__

    # Check for registered fallback
    fallback = error_handler.get_fallback(error_type)
    if fallback:
        try:
            fallback(context)
            return True
        except Exception as fallback_error:
            error_handler.log_error(
                fallback_error,
                {"fallback_for": error_type, "context": context},
                ErrorSeverity.HIGH,
            )

    return False


def cleanup_on_error(func: Callable) -> Callable:
    """Decorator that performs cleanup operations on error"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Perform cleanup
            cleanup_operations = [
                lambda: error_handler.save_error_report(),
                lambda: logging.shutdown(),
            ]

            for cleanup_op in cleanup_operations:
                try:
                    cleanup_op()
                except Exception as cleanup_error:
                    print(f"Cleanup failed: {cleanup_error}")

            raise e

    return wrapper


# Error monitoring and reporting


class ErrorMonitor:
    """Monitor and report error patterns"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_timestamps: Dict[str, List[float]] = {}

    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        if error_type not in self.error_timestamps:
            self.error_timestamps[error_type] = []
        self.error_timestamps[error_type].append(time.time())

    def get_error_rate(self, error_type: str, window_minutes: int = 60) -> float:
        """Calculate error rate for a specific error type"""
        if error_type not in self.error_timestamps:
            return 0.0

        cutoff_time = time.time() - (window_minutes * 60)
        recent_errors = [
            t for t in self.error_timestamps[error_type] if t > cutoff_time
        ]

        return len(recent_errors) / window_minutes  # errors per minute

    def should_alert(self, error_type: str, threshold: float = 5.0) -> bool:
        """Check if error rate exceeds threshold"""
        return self.get_error_rate(error_type) > threshold


# Global error monitor
error_monitor = ErrorMonitor()
