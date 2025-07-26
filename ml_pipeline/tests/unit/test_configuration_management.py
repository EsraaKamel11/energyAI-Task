#!/usr/bin/env python3
"""
Configuration Management Test Script
Demonstrates centralized configuration management for the ML pipeline
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Use absolute imports
from src.utils.config_manager import get_config, ConfigManager
from src.data_collection.web_scraper import WebScraper
from src.data_collection.pdf_extractor import PDFExtractor
from src.data_processing.deduplication import Deduplicator

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_configuration_loading():
    """Test basic configuration loading"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION LOADING")
    print("="*60)
    
    # Get global config
    config = get_config()
    
    # Test basic configuration access
    print(f"Pipeline Name: {config.get('pipeline.name')}")
    print(f"Environment: {config.get('pipeline.environment')}")
    print(f"Debug Mode: {config.get('pipeline.debug')}")
    print(f"Log Level: {config.get('pipeline.log_level')}")
    
    # Test configuration summary
    summary = config.get_config_summary()
    print(f"\nConfiguration Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test environment-specific configuration
    print(f"\nEnvironment-specific config:")
    env_config = config.get_environment_config()
    for key, value in env_config.items():
        print(f"  {key}: {value}")

def test_deduplication_configuration():
    """Test deduplication configuration"""
    print("\n" + "="*60)
    print("TESTING DEDUPLICATION CONFIGURATION")
    print("="*60)
    
    # Create deduplicator with default config
    deduplicator = Deduplicator()
    
    print(f"Deduplication Method: {deduplicator.method}")
    print(f"Semantic Threshold: {deduplicator.similarity_threshold}")
    print(f"Levenshtein Threshold: {deduplicator.levenshtein_threshold}")
    print(f"Fast Levenshtein Threshold: {deduplicator.fast_levenshtein_threshold}")
    
    # Test hybrid configuration
    if deduplicator.method == "hybrid":
        print(f"Hybrid Semantic First: {deduplicator.semantic_first}")
        print(f"Hybrid Semantic Threshold: {deduplicator.hybrid_semantic_threshold}")
        print(f"Hybrid Levenshtein Threshold: {deduplicator.hybrid_levenshtein_threshold}")
    
    # Test FAISS configuration
    if deduplicator.method == "faiss_semantic":
        print(f"FAISS Batch Size: {deduplicator.faiss_batch_size}")
        print(f"FAISS Search K: {deduplicator.faiss_search_k}")
        print(f"FAISS Normalize: {deduplicator.faiss_normalize}")
    
    # Test performance configuration
    print(f"Enable SimHash: {deduplicator.enable_simhash}")
    print(f"Chunk Size: {deduplicator.chunk_size}")
    print(f"Memory Limit MB: {deduplicator.memory_limit_mb}")
    
    # Get deduplication stats
    stats = deduplicator.get_deduplication_stats(100, 80)
    print(f"\nDeduplication Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_web_scraping_configuration():
    """Test web scraping configuration"""
    print("\n" + "="*60)
    print("TESTING WEB SCRAPING CONFIGURATION")
    print("="*60)
    
    # Create web scraper with default config
    scraper = WebScraper()
    
    print(f"Delay Range: {scraper.delay_range}s")
    print(f"Max Retries: {scraper.max_retries}")
    print(f"Timeout: {scraper.timeout}s")
    print(f"User Agents Count: {len(scraper.user_agents)}")
    print(f"Max Pages: {scraper.max_pages}")
    
    # Test content extraction configuration
    print(f"Extract Title: {scraper.extract_title}")
    print(f"Extract Meta: {scraper.extract_meta}")
    print(f"Extract Links: {scraper.extract_links}")
    print(f"Min Content Length: {scraper.min_content_length}")
    
    # Get scraping stats
    stats = scraper.get_scraping_stats()
    print(f"\nScraping Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_pdf_extraction_configuration():
    """Test PDF extraction configuration"""
    print("\n" + "="*60)
    print("TESTING PDF EXTRACTION CONFIGURATION")
    print("="*60)
    
    # Create PDF extractor with default config
    extractor = PDFExtractor()
    
    print(f"Extraction Strategy: {extractor.strategy}")
    print(f"Extract Text: {extractor.extract_text}")
    print(f"Extract Images: {extractor.extract_images}")
    print(f"Extract Tables: {extractor.extract_tables}")
    print(f"Extract Metadata: {extractor.extract_metadata}")
    print(f"Preserve Layout: {extractor.preserve_layout}")
    print(f"Max Workers: {extractor.max_workers}")
    print(f"Min Text Length: {extractor.min_text_length}")
    
    # Test text processing configuration
    print(f"Clean Text: {extractor.clean_text}")
    print(f"Remove Headers/Footers: {extractor.remove_headers_footers}")
    print(f"Merge Lines: {extractor.merge_lines}")
    print(f"Extract EV Terms: {extractor.extract_ev_terms}")
    
    # Get extraction stats (with empty results for testing)
    stats = extractor.get_extraction_stats([])
    print(f"\nExtraction Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_configuration_updates():
    """Test dynamic configuration updates"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION UPDATES")
    print("="*60)
    
    config = get_config()
    
    # Test setting configuration values
    print("Original deduplication method:", config.get("data_processing.deduplication.method"))
    
    # Update configuration
    config.set("data_processing.deduplication.method", "levenshtein")
    config.set("data_processing.deduplication.semantic_threshold", 0.85)
    
    print("Updated deduplication method:", config.get("data_processing.deduplication.method"))
    print("Updated semantic threshold:", config.get("data_processing.deduplication.semantic_threshold"))
    
    # Test creating new deduplicator with updated config
    deduplicator = Deduplicator()
    print("New deduplicator method:", deduplicator.method)
    print("New deduplicator threshold:", deduplicator.similarity_threshold)

def test_environment_switching():
    """Test environment-specific configuration"""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT SWITCHING")
    print("="*60)
    
    # Test development environment
    dev_config = ConfigManager(environment="development")
    print("Development Environment:")
    print(f"  Debug: {dev_config.get('pipeline.debug')}")
    print(f"  Log Level: {dev_config.get('pipeline.log_level')}")
    print(f"  Web Scraping Rate Limit: {dev_config.get('data_collection.web_scraping.rate_limit')}")
    
    # Test production environment
    prod_config = ConfigManager(environment="production")
    print("\nProduction Environment:")
    print(f"  Debug: {prod_config.get('pipeline.debug')}")
    print(f"  Log Level: {prod_config.get('pipeline.log_level')}")
    print(f"  Error Handling Max Attempts: {prod_config.get('error_handling.retry.max_attempts')}")
    
    # Test testing environment
    test_config = ConfigManager(environment="testing")
    print("\nTesting Environment:")
    print(f"  Debug: {test_config.get('pipeline.debug')}")
    print(f"  Prefect Enabled: {test_config.get('prefect.enabled')}")
    print(f"  Streamlit Enabled: {test_config.get('streamlit.enabled')}")

def test_configuration_validation():
    """Test configuration validation"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION VALIDATION")
    print("="*60)
    
    config = get_config()
    
    # Test validation of different sections
    sections = ["pipeline", "web_scraping", "pdf_extraction", "deduplication", "models", "error_handling"]
    
    for section in sections:
        is_valid = config.validate_specific_config(section)
        print(f"{section}: {'✓ Valid' if is_valid else '✗ Invalid'}")

def test_configuration_export():
    """Test configuration export"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION EXPORT")
    print("="*60)
    
    config = get_config()
    
    # Export as JSON
    json_config = config.export_config("json")
    print("JSON Export (first 500 chars):")
    print(json_config[:500] + "...")
    
    # Export as YAML
    yaml_config = config.export_config("yaml")
    print("\nYAML Export (first 500 chars):")
    print(yaml_config[:500] + "...")

def test_component_configuration_integration():
    """Test how components use configuration"""
    print("\n" + "="*60)
    print("TESTING COMPONENT CONFIGURATION INTEGRATION")
    print("="*60)
    
    # Test deduplicator with different methods
    methods = ["levenshtein", "semantic", "hybrid", "fast_levenshtein"]
    
    for method in methods:
        print(f"\nTesting {method} method:")
        try:
            deduplicator = Deduplicator(method=method)
            print(f"  ✓ Successfully created {method} deduplicator")
            print(f"  Threshold: {deduplicator.similarity_threshold}")
            
            # Test with sample data
            sample_docs = [
                {"id": "doc1", "text": "Electric vehicle charging station information"},
                {"id": "doc2", "text": "EV charging station details and specifications"},
                {"id": "doc3", "text": "Completely different topic about cooking"}
            ]
            
            result = deduplicator.deduplicate(sample_docs)
            print(f"  Deduplication result: {len(result)} documents (from {len(sample_docs)})")
            
        except Exception as e:
            print(f"  ✗ Failed to create {method} deduplicator: {e}")

def test_error_handling_configuration():
    """Test error handling configuration"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING CONFIGURATION")
    print("="*60)
    
    config = get_config()
    error_config = config.get_error_handling_config()
    
    print("Error Handling Configuration:")
    print(f"  Retry Max Attempts: {error_config.get('retry', {}).get('max_attempts')}")
    print(f"  Retry Exponential Backoff: {error_config.get('retry', {}).get('exponential_backoff')}")
    print(f"  Retry Base Delay: {error_config.get('retry', {}).get('base_delay')}s")
    print(f"  Retry Max Delay: {error_config.get('retry', {}).get('max_delay')}s")
    
    print(f"  Circuit Breaker Failure Threshold: {error_config.get('circuit_breaker', {}).get('failure_threshold')}")
    print(f"  Circuit Breaker Recovery Timeout: {error_config.get('circuit_breaker', {}).get('recovery_timeout')}s")
    
    print(f"  Logging Level: {error_config.get('logging', {}).get('level')}")
    print(f"  Logging File: {error_config.get('logging', {}).get('file')}")
    
    print(f"  Monitoring Error Rate Threshold: {error_config.get('monitoring', {}).get('error_rate_threshold')}")
    print(f"  Monitoring Alert on Threshold: {error_config.get('monitoring', {}).get('alert_on_threshold_exceeded')}")

def test_paths_configuration():
    """Test paths configuration"""
    print("\n" + "="*60)
    print("TESTING PATHS CONFIGURATION")
    print("="*60)
    
    config = get_config()
    paths_config = config.get_paths_config()
    
    print("Paths Configuration:")
    print("Data Paths:")
    for key, path in paths_config.get("data", {}).items():
        print(f"  {key}: {path}")
    
    print("\nOutput Paths:")
    for key, path in paths_config.get("outputs", {}).items():
        print(f"  {key}: {path}")
    
    print("\nConfig Paths:")
    for key, path in paths_config.get("config", {}).items():
        print(f"  {key}: {path}")

def main():
    """Main test function"""
    print("CONFIGURATION MANAGEMENT TEST SUITE")
    print("="*60)
    print("This script demonstrates the centralized configuration management system")
    print("and shows how all pipeline components use configuration instead of hardcoded values.")
    
    setup_logging()
    
    try:
        # Run all tests
        test_configuration_loading()
        test_deduplication_configuration()
        test_web_scraping_configuration()
        test_pdf_extraction_configuration()
        test_configuration_updates()
        test_environment_switching()
        test_configuration_validation()
        test_configuration_export()
        test_component_configuration_integration()
        test_error_handling_configuration()
        test_paths_configuration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Benefits of Configuration Management:")
        print("✓ No more hardcoded parameters")
        print("✓ Environment-specific configurations")
        print("✓ Easy parameter tuning without code changes")
        print("✓ Centralized configuration validation")
        print("✓ Dynamic configuration updates")
        print("✓ Configuration export and import")
        print("✓ Consistent configuration across all components")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
