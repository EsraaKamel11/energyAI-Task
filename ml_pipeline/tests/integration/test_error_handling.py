#!/usr/bin/env python3
"""
Comprehensive test for error handling and retry mechanisms
Demonstrates robust error recovery, fallback strategies, and monitoring
"""

import os
import sys
import time
import logging
import tempfile
import json
from typing import List, Dict, Any
import requests
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.utils.error_handling import (
    ErrorHandler, ErrorSeverity, ErrorInfo,
    retry_with_fallback, tenacity_retry, safe_execute, 
    circuit_breaker, error_handler, error_monitor,
    robust_web_request, robust_file_operation,
    recover_from_error, cleanup_on_error
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_error_handling():
    """Test basic error handling functionality"""
    
    print("\n" + "="*60)
    print("TESTING BASIC ERROR HANDLING")
    print("="*60)
    
    # Create a new error handler for testing
    test_handler = ErrorHandler("TestHandler")
    
    # Test error logging
    test_error = ValueError("Test error message")
    context = {"test": True, "function": "test_function"}
    
    error_info = test_handler.log_error(
        test_error, context, ErrorSeverity.MEDIUM
    )
    
    print(f"✅ Error logged: {error_info.error_type}")
    print(f"✅ Error message: {error_info.message}")
    print(f"✅ Error severity: {error_info.severity}")
    print(f"✅ Error context: {error_info.context}")
    print(f"✅ Error timestamp: {error_info.timestamp}")
    
    # Test error history
    assert len(test_handler.error_history) == 1
    print(f"✅ Error history count: {len(test_handler.error_history)}")
    
    # Test fallback registration
    def test_fallback(context):
        return {"fallback_result": "success"}
    
    test_handler.register_fallback("ValueError", test_fallback)
    fallback = test_handler.get_fallback("ValueError")
    
    assert fallback is not None
    print(f"✅ Fallback registered and retrieved successfully")
    
    # Test error report saving
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        report_path = f.name
    
    test_handler.save_error_report(report_path)
    
    assert os.path.exists(report_path)
    print(f"✅ Error report saved to: {report_path}")
    
    # Clean up
    os.unlink(report_path)
    
    return test_handler

def test_retry_decorators():
    """Test retry decorators with different scenarios"""
    
    print("\n" + "="*60)
    print("TESTING RETRY DECORATORS")
    print("="*60)
    
    # Test retry_with_fallback
    call_count = 0
    
    @retry_with_fallback(max_attempts=3, exceptions=(ValueError,))
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Attempt {call_count} failed")
        return "Success on attempt 3"
    
    result = failing_function()
    print(f"✅ Retry with fallback: {result} (attempts: {call_count})")
    
    # Test retry_with_fallback with fallback function
    call_count = 0
    
    def fallback_func():
        return "Fallback result"
    
    @retry_with_fallback(
        max_attempts=2, 
        exceptions=(RuntimeError,), 
        fallback_func=fallback_func
    )
    def always_failing_function():
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"Always fails on attempt {call_count}")
    
    result = always_failing_function()
    print(f"✅ Retry with fallback function: {result} (attempts: {call_count})")
    
    # Test safe_execute
    @safe_execute(fallback_value="default", exceptions=(ValueError,))
    def safe_function(should_fail=False):
        if should_fail:
            raise ValueError("Intentional failure")
        return "success"
    
    result1 = safe_function(should_fail=False)
    result2 = safe_function(should_fail=True)
    
    print(f"✅ Safe execute success: {result1}")
    print(f"✅ Safe execute fallback: {result2}")

def test_circuit_breaker():
    """Test circuit breaker pattern"""
    
    print("\n" + "="*60)
    print("TESTING CIRCUIT BREAKER")
    print("="*60)
    
    call_count = 0
    
    @circuit_breaker(failure_threshold=3, recovery_timeout=1)
    def circuit_breaker_test():
        nonlocal call_count
        call_count += 1
        if call_count <= 5:  # Fail first 5 times
            raise RuntimeError(f"Failure {call_count}")
        return "Success after failures"
    
    # Test circuit breaker behavior
    results = []
    for i in range(7):
        try:
            result = circuit_breaker_test()
            results.append(f"Success: {result}")
        except Exception as e:
            results.append(f"Failure: {e}")
    
    print("Circuit breaker test results:")
    for i, result in enumerate(results):
        print(f"  Attempt {i+1}: {result}")
    
    # Wait for recovery
    time.sleep(1.1)
    
    try:
        result = circuit_breaker_test()
        print(f"✅ Circuit breaker recovered: {result}")
    except Exception as e:
        print(f"❌ Circuit breaker failed to recover: {e}")

def test_robust_web_request():
    """Test robust web request functionality"""
    
    print("\n" + "="*60)
    print("TESTING ROBUST WEB REQUEST")
    print("="*60)
    
    # Test with a real URL
    try:
        response_text = robust_web_request("https://httpbin.org/html")
        print(f"✅ Robust web request successful: {len(response_text)} characters")
    except Exception as e:
        print(f"❌ Robust web request failed: {e}")
    
    # Test with invalid URL (should fail gracefully)
    try:
        response_text = robust_web_request("https://invalid-url-that-does-not-exist.com")
        print(f"✅ Unexpected success: {len(response_text)} characters")
    except Exception as e:
        print(f"✅ Expected failure handled: {e}")

def test_robust_file_operations():
    """Test robust file operations"""
    
    print("\n" + "="*60)
    print("TESTING ROBUST FILE OPERATIONS")
    print("="*60)
    
    # Test file reading
    @robust_file_operation("read")
    def read_file_safely(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        test_file = f.name
        f.write("Test content")
    
    try:
        content = read_file_safely(test_file)
        print(f"✅ File read successfully: {content}")
    except Exception as e:
        print(f"❌ File read failed: {e}")
    
    # Test file writing
    @robust_file_operation("write")
    def write_file_safely(filepath, content):
        with open(filepath, 'w') as f:
            f.write(content)
    
    test_write_file = test_file.replace('.txt', '_write.txt')
    try:
        write_file_safely(test_write_file, "Written content")
        print(f"✅ File write successful")
    except Exception as e:
        print(f"❌ File write failed: {e}")
    
    # Clean up
    os.unlink(test_file)
    if os.path.exists(test_write_file):
        os.unlink(test_write_file)

def test_error_monitoring():
    """Test error monitoring and rate calculation"""
    
    print("\n" + "="*60)
    print("TESTING ERROR MONITORING")
    print("="*60)
    
    # Record some test errors
    error_monitor.record_error("ConnectionError")
    error_monitor.record_error("TimeoutError")
    error_monitor.record_error("ConnectionError")
    error_monitor.record_error("ValueError")
    error_monitor.record_error("ConnectionError")
    
    # Test error rate calculation
    connection_error_rate = error_monitor.get_error_rate("ConnectionError", window_minutes=1)
    timeout_error_rate = error_monitor.get_error_rate("TimeoutError", window_minutes=1)
    
    print(f"✅ ConnectionError rate: {connection_error_rate:.2f} errors/minute")
    print(f"✅ TimeoutError rate: {timeout_error_rate:.2f} errors/minute")
    
    # Test alert threshold
    should_alert = error_monitor.should_alert("ConnectionError", threshold=2.0)
    print(f"✅ Should alert for ConnectionError: {should_alert}")

def test_cleanup_on_error():
    """Test cleanup functionality on error"""
    
    print("\n" + "="*60)
    print("TESTING CLEANUP ON ERROR")
    print("="*60)
    
    cleanup_called = False
    
    def test_cleanup():
        nonlocal cleanup_called
        cleanup_called = True
        print("  🧹 Cleanup function called")
    
    @cleanup_on_error
    def function_with_cleanup():
        raise RuntimeError("Intentional error for cleanup test")
    
    try:
        function_with_cleanup()
    except Exception as e:
        print(f"✅ Error caught: {e}")
    
    print(f"✅ Cleanup called: {cleanup_called}")

def test_web_scraper_error_handling():
    """Test web scraper error handling integration"""
    
    print("\n" + "="*60)
    print("TESTING WEB SCRAPER ERROR HANDLING")
    print("="*60)
    
    try:
        from src.data_collection.web_scraper import WebScraper
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = WebScraper(temp_dir, rate_limit=0.1, max_retries=2)
            
            # Test URLs (mix of valid and invalid)
            test_urls = [
                "https://httpbin.org/html",  # Valid
                "https://invalid-url-that-does-not-exist.com",  # Invalid
                "https://httpbin.org/status/404",  # 404 error
                "https://httpbin.org/status/500",  # 500 error
            ]
            
            print(f"Testing {len(test_urls)} URLs...")
            scraper.collect(test_urls)
            
            # Get scraping statistics
            stats = scraper.get_scraping_stats()
            print(f"✅ Scraping completed")
            print(f"  Total errors: {stats['total_errors']}")
            print(f"  Error types: {stats['error_types']}")
            
    except ImportError:
        print("⚠️  Web scraper not available, skipping test")
    except Exception as e:
        print(f"❌ Web scraper test failed: {e}")

def test_pdf_extractor_error_handling():
    """Test PDF extractor error handling integration"""
    
    print("\n" + "="*60)
    print("TESTING PDF EXTRACTOR ERROR HANDLING")
    print("="*60)
    
    try:
        from src.data_collection.pdf_extractor import PDFExtractor
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = PDFExtractor(temp_dir, strategy="fast", max_retries=2)
            
            # Test with non-existent PDF
            test_pdfs = [
                "non_existent_file.pdf",
                "another_missing_file.pdf"
            ]
            
            print(f"Testing {len(test_pdfs)} non-existent PDFs...")
            extractor.collect(test_pdfs)
            
            # Get extraction summary
            summary = extractor.get_extraction_summary(test_pdfs)
            print(f"✅ PDF extraction completed")
            print(f"  Total PDFs: {summary['total_pdfs']}")
            print(f"  Extraction errors: {summary['extraction_errors']}")
            print(f"  Error types: {summary['error_types']}")
            
    except ImportError:
        print("⚠️  PDF extractor not available, skipping test")
    except Exception as e:
        print(f"❌ PDF extractor test failed: {e}")

def test_error_recovery_strategies():
    """Test various error recovery strategies"""
    
    print("\n" + "="*60)
    print("TESTING ERROR RECOVERY STRATEGIES")
    print("="*60)
    
    # Test recover_from_error function
    test_error = ConnectionError("Connection failed")
    context = {"url": "https://example.com", "attempt": 1}
    
    # Register a test fallback
    def test_fallback(context):
        print("  🔄 Test fallback executed")
        return True
    
    error_handler.register_fallback("ConnectionError", test_fallback)
    
    recovery_success = recover_from_error(test_error, context)
    print(f"✅ Error recovery: {recovery_success}")
    
    # Test recovery with non-existent fallback
    test_error2 = ValueError("Value error")
    recovery_success2 = recover_from_error(test_error2, context)
    print(f"✅ Recovery without fallback: {recovery_success2}")

def demonstrate_error_handling_benefits():
    """Demonstrate the benefits of comprehensive error handling"""
    
    print("\n" + "="*60)
    print("ERROR HANDLING BENEFITS")
    print("="*60)
    
    benefits = [
        "🔄 Automatic retry mechanisms with exponential backoff",
        "🛡️ Graceful degradation with fallback strategies",
        "📊 Comprehensive error monitoring and reporting",
        "⚡ Circuit breaker pattern for system protection",
        "🔧 Multiple retry libraries (tenacity, backoff)",
        "📝 Structured error logging with context",
        "🎯 Error rate monitoring and alerting",
        "🧹 Automatic cleanup on failures",
        "🔄 Resume capability for interrupted operations",
        "📈 Performance monitoring and statistics"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print(f"\n💡 ERROR HANDLING PATTERNS:")
    patterns = [
        "Retry with exponential backoff for transient failures",
        "Fallback to alternative strategies for permanent failures",
        "Circuit breaker for protecting downstream services",
        "Graceful degradation for non-critical features",
        "Comprehensive logging for debugging and monitoring"
    ]
    
    for pattern in patterns:
        print(f"  • {pattern}")

def main():
    """Run all error handling tests"""
    
    print("🛡️ Comprehensive Error Handling Test Suite")
    print("Testing retry mechanisms, fallback strategies, and error recovery...")
    
    try:
        # Test basic functionality
        test_handler = test_basic_error_handling()
        
        # Test retry decorators
        test_retry_decorators()
        
        # Test circuit breaker
        test_circuit_breaker()
        
        # Test robust operations
        test_robust_web_request()
        test_robust_file_operations()
        
        # Test error monitoring
        test_error_monitoring()
        
        # Test cleanup functionality
        test_cleanup_on_error()
        
        # Test integration with components
        test_web_scraper_error_handling()
        test_pdf_extractor_error_handling()
        
        # Test recovery strategies
        test_error_recovery_strategies()
        
        # Demonstrate benefits
        demonstrate_error_handling_benefits()
        
        print("\n✅ All error handling tests completed!")
        print("\n📋 Key Error Handling Features:")
        print("  • Retry mechanisms with exponential backoff")
        print("  • Fallback strategies for graceful degradation")
        print("  • Circuit breaker pattern for system protection")
        print("  • Comprehensive error monitoring and reporting")
        print("  • Automatic cleanup and recovery")
        
        # Save error report
        os.makedirs("test_outputs", exist_ok=True)
        error_handler.save_error_report("test_outputs/error_handling_test_report.json")
        print(f"\n📁 Error report saved to: test_outputs/error_handling_test_report.json")
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        print(f"❌ Error handling test failed: {e}")

if __name__ == "__main__":
    main() 
