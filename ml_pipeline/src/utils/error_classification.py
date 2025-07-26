#!/usr/bin/env python3
"""
Error Classification and Recovery System

This module provides comprehensive error classification and recovery strategies
for web scraping and PDF extraction operations.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import re
from dataclasses import dataclass
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""

    # Network errors
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_CONNECTION = "network_connection"
    NETWORK_DNS = "network_dns"
    NETWORK_SSL = "network_ssl"
    NETWORK_PROXY = "network_proxy"

    # HTTP errors
    HTTP_4XX = "http_4xx"
    HTTP_5XX = "http_5xx"
    HTTP_REDIRECT = "http_redirect"
    HTTP_RATE_LIMIT = "http_rate_limit"

    # Content errors
    CONTENT_PARSING = "content_parsing"
    CONTENT_ENCODING = "content_encoding"
    CONTENT_EMPTY = "content_empty"
    CONTENT_TOO_LARGE = "content_too_large"

    # Browser automation errors
    SELENIUM_TIMEOUT = "selenium_timeout"
    SELENIUM_ELEMENT_NOT_FOUND = "selenium_element_not_found"
    SELENIUM_DRIVER_ERROR = "selenium_driver_error"
    SELENIUM_COOKIE_CONSENT = "selenium_cookie_consent"

    # PDF errors
    PDF_CORRUPTED = "pdf_corrupted"
    PDF_PASSWORD_PROTECTED = "pdf_password_protected"
    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_FORMAT_UNSUPPORTED = "pdf_format_unsupported"
    PDF_EXTRACTION_FAILED = "pdf_extraction_failed"

    # Memory errors
    MEMORY_LIMIT_EXCEEDED = "memory_limit_exceeded"
    MEMORY_FRAGMENTATION = "memory_fragmentation"

    # Configuration errors
    CONFIG_INVALID = "config_invalid"
    CONFIG_MISSING = "config_missing"

    # Unknown errors
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Structured error information."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Exception
    url: Optional[str] = None
    file_path: Optional[str] = None
    timestamp: datetime = None
    retry_count: int = 0
    recovery_strategies: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recovery_strategies is None:
            self.recovery_strategies = []


class ErrorClassifier:
    """Classifies errors and provides recovery strategies."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Error pattern mappings
        self.error_patterns = {
            # Network patterns
            r"timeout": ErrorCategory.NETWORK_TIMEOUT,
            r"connection.*refused": ErrorCategory.NETWORK_CONNECTION,
            r"dns.*error": ErrorCategory.NETWORK_DNS,
            r"ssl.*error": ErrorCategory.NETWORK_SSL,
            r"proxy.*error": ErrorCategory.NETWORK_PROXY,
            # HTTP patterns
            r"404": ErrorCategory.HTTP_4XX,
            r"403": ErrorCategory.HTTP_4XX,
            r"401": ErrorCategory.HTTP_4XX,
            r"500": ErrorCategory.HTTP_5XX,
            r"502": ErrorCategory.HTTP_5XX,
            r"503": ErrorCategory.HTTP_5XX,
            r"rate.*limit": ErrorCategory.HTTP_RATE_LIMIT,
            # Content patterns
            r"parsing.*error": ErrorCategory.CONTENT_PARSING,
            r"encoding.*error": ErrorCategory.CONTENT_ENCODING,
            r"empty.*content": ErrorCategory.CONTENT_EMPTY,
            r"content.*too.*large": ErrorCategory.CONTENT_TOO_LARGE,
            # Selenium patterns
            r"timeout.*exception": ErrorCategory.SELENIUM_TIMEOUT,
            r"no.*such.*element": ErrorCategory.SELENIUM_ELEMENT_NOT_FOUND,
            r"webdriver.*error": ErrorCategory.SELENIUM_DRIVER_ERROR,
            r"cookie.*consent": ErrorCategory.SELENIUM_COOKIE_CONSENT,
            # PDF patterns
            r"pdf.*corrupted": ErrorCategory.PDF_CORRUPTED,
            r"password.*protected": ErrorCategory.PDF_PASSWORD_PROTECTED,
            r"encrypted.*pdf": ErrorCategory.PDF_ENCRYPTED,
            r"unsupported.*format": ErrorCategory.PDF_FORMAT_UNSUPPORTED,
            r"extraction.*failed": ErrorCategory.PDF_EXTRACTION_FAILED,
            # Memory patterns
            r"memory.*limit": ErrorCategory.MEMORY_LIMIT_EXCEEDED,
            r"out.*of.*memory": ErrorCategory.MEMORY_LIMIT_EXCEEDED,
            r"memory.*fragmentation": ErrorCategory.MEMORY_FRAGMENTATION,
        }

        # Severity mappings
        self.severity_mappings = {
            ErrorCategory.NETWORK_TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_CONNECTION: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK_DNS: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_SSL: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_PROXY: ErrorSeverity.MEDIUM,
            ErrorCategory.HTTP_4XX: ErrorSeverity.MEDIUM,
            ErrorCategory.HTTP_5XX: ErrorSeverity.HIGH,
            ErrorCategory.HTTP_REDIRECT: ErrorSeverity.LOW,
            ErrorCategory.HTTP_RATE_LIMIT: ErrorSeverity.HIGH,
            ErrorCategory.CONTENT_PARSING: ErrorSeverity.MEDIUM,
            ErrorCategory.CONTENT_ENCODING: ErrorSeverity.LOW,
            ErrorCategory.CONTENT_EMPTY: ErrorSeverity.MEDIUM,
            ErrorCategory.CONTENT_TOO_LARGE: ErrorSeverity.MEDIUM,
            ErrorCategory.SELENIUM_TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.SELENIUM_ELEMENT_NOT_FOUND: ErrorSeverity.LOW,
            ErrorCategory.SELENIUM_DRIVER_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.SELENIUM_COOKIE_CONSENT: ErrorSeverity.LOW,
            ErrorCategory.PDF_CORRUPTED: ErrorSeverity.HIGH,
            ErrorCategory.PDF_PASSWORD_PROTECTED: ErrorSeverity.MEDIUM,
            ErrorCategory.PDF_ENCRYPTED: ErrorSeverity.MEDIUM,
            ErrorCategory.PDF_FORMAT_UNSUPPORTED: ErrorSeverity.HIGH,
            ErrorCategory.PDF_EXTRACTION_FAILED: ErrorSeverity.HIGH,
            ErrorCategory.MEMORY_LIMIT_EXCEEDED: ErrorSeverity.CRITICAL,
            ErrorCategory.MEMORY_FRAGMENTATION: ErrorSeverity.HIGH,
            ErrorCategory.CONFIG_INVALID: ErrorSeverity.CRITICAL,
            ErrorCategory.CONFIG_MISSING: ErrorSeverity.CRITICAL,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
        }

        # Recovery strategy mappings
        self.recovery_strategies = {
            ErrorCategory.NETWORK_TIMEOUT: [
                "retry_with_backoff",
                "increase_timeout",
                "use_proxy",
            ],
            ErrorCategory.NETWORK_CONNECTION: [
                "retry_with_backoff",
                "use_proxy",
                "change_user_agent",
            ],
            ErrorCategory.NETWORK_DNS: ["retry_with_backoff", "use_alternative_dns"],
            ErrorCategory.NETWORK_SSL: [
                "disable_ssl_verification",
                "retry_with_backoff",
            ],
            ErrorCategory.NETWORK_PROXY: [
                "change_proxy",
                "disable_proxy",
                "retry_with_backoff",
            ],
            ErrorCategory.HTTP_4XX: ["change_user_agent", "add_headers", "use_session"],
            ErrorCategory.HTTP_5XX: [
                "retry_with_backoff",
                "use_proxy",
                "change_user_agent",
            ],
            ErrorCategory.HTTP_REDIRECT: [
                "follow_redirects",
                "handle_redirect_manually",
            ],
            ErrorCategory.HTTP_RATE_LIMIT: [
                "increase_delay",
                "use_proxy_rotation",
                "implement_rate_limiting",
            ],
            ErrorCategory.CONTENT_PARSING: [
                "try_different_parser",
                "use_robust_parsing",
                "extract_raw_content",
            ],
            ErrorCategory.CONTENT_ENCODING: [
                "try_different_encoding",
                "use_encoding_detection",
                "extract_raw_content",
            ],
            ErrorCategory.CONTENT_EMPTY: [
                "try_dynamic_scraping",
                "wait_for_content",
                "check_javascript",
            ],
            ErrorCategory.CONTENT_TOO_LARGE: [
                "use_chunked_processing",
                "increase_memory_limit",
                "stream_content",
            ],
            ErrorCategory.SELENIUM_TIMEOUT: [
                "increase_timeout",
                "wait_for_element",
                "use_static_scraping",
            ],
            ErrorCategory.SELENIUM_ELEMENT_NOT_FOUND: [
                "try_different_selectors",
                "wait_for_element",
                "use_static_scraping",
            ],
            ErrorCategory.SELENIUM_DRIVER_ERROR: [
                "restart_driver",
                "use_different_browser",
                "use_static_scraping",
            ],
            ErrorCategory.SELENIUM_COOKIE_CONSENT: [
                "handle_cookie_consent",
                "try_different_selectors",
                "use_static_scraping",
            ],
            ErrorCategory.PDF_CORRUPTED: [
                "try_different_extractor",
                "use_repair_tools",
                "skip_file",
            ],
            ErrorCategory.PDF_PASSWORD_PROTECTED: [
                "try_common_passwords",
                "skip_file",
                "request_password",
            ],
            ErrorCategory.PDF_ENCRYPTED: [
                "try_decryption",
                "skip_file",
                "request_password",
            ],
            ErrorCategory.PDF_FORMAT_UNSUPPORTED: [
                "try_different_extractor",
                "convert_format",
                "skip_file",
            ],
            ErrorCategory.PDF_EXTRACTION_FAILED: [
                "try_different_strategy",
                "use_chunked_extraction",
                "skip_file",
            ],
            ErrorCategory.MEMORY_LIMIT_EXCEEDED: [
                "force_garbage_collection",
                "use_chunked_processing",
                "increase_memory_limit",
            ],
            ErrorCategory.MEMORY_FRAGMENTATION: [
                "force_garbage_collection",
                "restart_process",
                "optimize_memory_usage",
            ],
            ErrorCategory.CONFIG_INVALID: [
                "validate_configuration",
                "use_default_config",
                "request_config_fix",
            ],
            ErrorCategory.CONFIG_MISSING: [
                "use_default_config",
                "load_from_environment",
                "request_config",
            ],
            ErrorCategory.UNKNOWN: [
                "retry_with_backoff",
                "log_error_details",
                "skip_operation",
            ],
        }

    def classify_error(
        self, exception: Exception, context: Dict[str, Any] = None
    ) -> ErrorInfo:
        """
        Classify an exception and return structured error information.

        Args:
            exception: The exception to classify
            context: Additional context (URL, file path, etc.)

        Returns:
            ErrorInfo object with classification and recovery strategies
        """
        if context is None:
            context = {}

        # Get error message
        error_message = str(exception).lower()
        error_type = type(exception).__name__.lower()

        # Classify error
        category = self._classify_error_message(error_message, error_type)
        severity = self.severity_mappings.get(category, ErrorSeverity.MEDIUM)

        # Get recovery strategies
        strategies = self.recovery_strategies.get(category, [])

        # Create error info
        error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            url=context.get("url"),
            file_path=context.get("file_path"),
            recovery_strategies=strategies,
        )

        self.logger.debug(
            f"Classified error: {category.value} ({severity.value}) - {error_message}"
        )

        return error_info

    def _classify_error_message(self, message: str, error_type: str) -> ErrorCategory:
        """Classify error based on message and type."""
        message_lower = message.lower()

        # Check patterns
        for pattern, category in self.error_patterns.items():
            if re.search(pattern, message_lower):
                return category

        # Check specific exception types
        if "timeout" in error_type:
            return ErrorCategory.NETWORK_TIMEOUT
        elif "connection" in error_type:
            return ErrorCategory.NETWORK_CONNECTION
        elif "ssl" in error_type:
            return ErrorCategory.NETWORK_SSL
        elif "http" in error_type:
            if any(code in message_lower for code in ["404", "403", "401"]):
                return ErrorCategory.HTTP_4XX
            elif any(code in message_lower for code in ["500", "502", "503"]):
                return ErrorCategory.HTTP_5XX
        elif "selenium" in error_type:
            return ErrorCategory.SELENIUM_DRIVER_ERROR
        elif "memory" in error_type:
            return ErrorCategory.MEMORY_LIMIT_EXCEEDED

        return ErrorCategory.UNKNOWN

    def get_recovery_strategy(self, error_info: ErrorInfo) -> Optional[str]:
        """
        Get the best recovery strategy for an error.

        Args:
            error_info: Classified error information

        Returns:
            Recommended recovery strategy or None
        """
        strategies = error_info.recovery_strategies

        if not strategies:
            return None

        # Prioritize strategies based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            # For critical errors, try aggressive recovery
            priority_strategies = [
                "restart_process",
                "increase_memory_limit",
                "use_default_config",
            ]
        elif error_info.severity == ErrorSeverity.HIGH:
            # For high severity, try moderate recovery
            priority_strategies = [
                "retry_with_backoff",
                "use_proxy",
                "change_user_agent",
            ]
        else:
            # For low/medium severity, try gentle recovery
            priority_strategies = [
                "retry_with_backoff",
                "try_different_parser",
                "wait_for_element",
            ]

        # Return first available priority strategy, or first available strategy
        for strategy in priority_strategies:
            if strategy in strategies:
                return strategy

        return strategies[0] if strategies else None

    def should_retry(
        self, error_info: ErrorInfo, retry_count: int, max_retries: int = 3
    ) -> bool:
        """
        Determine if an operation should be retried based on error classification.

        Args:
            error_info: Classified error information
            retry_count: Current retry count
            max_retries: Maximum allowed retries

        Returns:
            True if should retry, False otherwise
        """
        if retry_count >= max_retries:
            return False

        # Don't retry critical errors that can't be recovered
        if error_info.severity == ErrorSeverity.CRITICAL:
            if error_info.category in [
                ErrorCategory.CONFIG_INVALID,
                ErrorCategory.CONFIG_MISSING,
                ErrorCategory.PDF_CORRUPTED,
            ]:
                return False

        # Don't retry certain error types
        if error_info.category in [
            ErrorCategory.HTTP_4XX,  # Client errors
            ErrorCategory.PDF_PASSWORD_PROTECTED,
            ErrorCategory.PDF_ENCRYPTED,
            ErrorCategory.CONTENT_EMPTY,
        ]:
            return False

        return True

    def get_retry_delay(self, error_info: ErrorInfo, retry_count: int) -> float:
        """
        Calculate retry delay based on error classification.

        Args:
            error_info: Classified error information
            retry_count: Current retry count

        Returns:
            Delay in seconds
        """
        base_delay = 1.0

        # Adjust base delay based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            base_delay = 5.0
        elif error_info.severity == ErrorSeverity.HIGH:
            base_delay = 3.0
        elif error_info.severity == ErrorSeverity.MEDIUM:
            base_delay = 2.0

        # Apply exponential backoff
        delay = base_delay * (2**retry_count)

        # Cap maximum delay
        max_delay = 60.0
        return min(delay, max_delay)


# Global error classifier instance
error_classifier = ErrorClassifier()


def classify_and_handle_error(
    exception: Exception, context: Dict[str, Any] = None
) -> ErrorInfo:
    """
    Convenience function to classify and handle an error.

    Args:
        exception: The exception to classify
        context: Additional context

    Returns:
        ErrorInfo object
    """
    return error_classifier.classify_error(exception, context)


def get_error_recovery_strategy(error_info: ErrorInfo) -> Optional[str]:
    """
    Convenience function to get recovery strategy.

    Args:
        error_info: Classified error information

    Returns:
        Recommended recovery strategy
    """
    return error_classifier.get_recovery_strategy(error_info)


def should_retry_operation(
    error_info: ErrorInfo, retry_count: int, max_retries: int = 3
) -> bool:
    """
    Convenience function to determine if operation should be retried.

    Args:
        error_info: Classified error information
        retry_count: Current retry count
        max_retries: Maximum allowed retries

    Returns:
        True if should retry
    """
    return error_classifier.should_retry(error_info, retry_count, max_retries)


def get_retry_delay_for_error(error_info: ErrorInfo, retry_count: int) -> float:
    """
    Convenience function to get retry delay.

    Args:
        error_info: Classified error information
        retry_count: Current retry count

    Returns:
        Delay in seconds
    """
    return error_classifier.get_retry_delay(error_info, retry_count)
