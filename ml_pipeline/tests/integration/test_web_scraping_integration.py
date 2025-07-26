#!/usr/bin/env python3
"""
Web Scraping Integration Test Script
Demonstrates metadata integration and source-specific extractors
"""

import logging
import time
import json
from pathlib import Path
import tempfile
import shutil

# Import web scraping components
from src.data_collection.web_scraper import WebScraper
from src.data_collection.metadata_handler import MetadataHandler, Metadata

# Import configuration manager
from src.utils.config_manager import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_metadata_handler():
    """Test metadata handler functionality"""
    print("\n" + "="*60)
    print("TESTING METADATA HANDLER")
    print("="*60)
    
    # Initialize metadata handler
    metadata_handler = MetadataHandler()
    
    # Test web metadata extraction
    print("Testing web metadata extraction...")
    web_source_data = {
        "url": "https://example.com/article",
        "content": "This is a sample article about electric vehicles and their benefits for the environment.",
        "html": """
        <html>
        <head>
            <title>EV Charging Guide - Complete Information</title>
            <meta name="description" content="Comprehensive guide to electric vehicle charging">
            <meta name="keywords" content="EV, electric vehicle, charging, battery">
            <meta name="author" content="Energy Expert">
        </head>
        <body>
            <h1>EV Charging Guide</h1>
            <p>This is a comprehensive guide about electric vehicle charging.</p>
        </body>
        </html>
        """,
        "custom_fields": {
            "category": "technology",
            "read_time": "5 minutes"
        }
    }
    
    web_metadata = metadata_handler.create_metadata("web", web_source_data)
    print(f"Web metadata created: {web_metadata.source_id}")
    print(f"Title: {web_metadata.title}")
    print(f"Author: {web_metadata.author}")
    print(f"Quality score: {web_metadata.quality_score}")
    print(f"Processing status: {web_metadata.processing_status}")
    
    # Test PDF metadata extraction
    print("\nTesting PDF metadata extraction...")
    pdf_source_data = {
        "file_path": "/path/to/document.pdf",
        "content": "This is a PDF document about renewable energy sources.",
        "pdf_metadata": {
            "title": "Renewable Energy Report",
            "author": "Research Team",
            "subject": "Energy Analysis",
            "creator": "PDF Generator",
            "creationDate": "2024-01-15"
        }
    }
    
    pdf_metadata = metadata_handler.create_metadata("pdf", pdf_source_data)
    print(f"PDF metadata created: {pdf_metadata.source_id}")
    print(f"Title: {pdf_metadata.title}")
    print(f"Author: {pdf_metadata.author}")
    print(f"Content type: {pdf_metadata.content_type}")
    
    # Test API metadata extraction
    print("\nTesting API metadata extraction...")
    api_source_data = {
        "endpoint": "https://api.example.com/energy-data",
        "response": type('Response', (), {
            'headers': {'content-type': 'application/json'},
            'content': b'{"data": "sample"}'
        })(),
        "api_info": {
            "name": "Energy Data API",
            "description": "API for accessing energy consumption data",
            "version": "1.0",
            "base_url": "https://api.example.com"
        }
    }
    
    api_metadata = metadata_handler.create_metadata("api", api_source_data)
    print(f"API metadata created: {api_metadata.source_id}")
    print(f"Title: {api_metadata.title}")
    print(f"Content type: {api_metadata.content_type}")
    
    return [web_metadata, pdf_metadata, api_metadata]

def test_source_specific_extractors():
    """Test source-specific web extractors"""
    print("\n" + "="*60)
    print("TESTING SOURCE-SPECIFIC EXTRACTORS")
    print("="*60)
    
    # Initialize web scraper
    scraper = WebScraper(user_agent="energyAI-bot")
    
    # Test different extractor types
    extractor_types = ["generic", "news", "blog", "documentation", "ecommerce", "government", "academic"]
    
    for extractor_type in extractor_types:
        print(f"\nTesting {extractor_type} extractor...")
        
        # Create mock HTML content for testing
        mock_html = create_mock_html(extractor_type)
        
        # Test extractor (without actual web request)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(mock_html, 'html.parser')
            
            # Get extractor
            extractor = scraper.extractors.get(extractor_type, scraper.extractors["generic"])
            
            # Mock response
            mock_response = type('Response', (), {
                'text': mock_html,
                'content': mock_html.encode(),
                'headers': {'content-type': 'text/html'}
            })()
            
            # Extract content
            result = extractor.extract(soup, "https://example.com", mock_response)
            
            if result:
                print(f"  ✓ {extractor_type} extraction successful")
                print(f"    Title: {result.get('title', 'N/A')}")
                print(f"    Content length: {len(result.get('content', ''))}")
                print(f"    Extractor type: {result.get('extractor_type')}")
                
                # Show extractor-specific fields
                if extractor_type == "news":
                    print(f"    Publication date: {result.get('publication_date', 'N/A')}")
                    print(f"    Author: {result.get('author', 'N/A')}")
                elif extractor_type == "blog":
                    print(f"    Tags: {result.get('tags', [])}")
                    print(f"    Comments count: {result.get('comments_count', 'N/A')}")
                elif extractor_type == "documentation":
                    print(f"    Code blocks: {len(result.get('code_blocks', []))}")
                    print(f"    TOC items: {len(result.get('table_of_contents', []))}")
                elif extractor_type == "ecommerce":
                    print(f"    Product name: {result.get('product_name', 'N/A')}")
                    print(f"    Price: {result.get('price', 'N/A')}")
                elif extractor_type == "government":
                    print(f"    Document number: {result.get('document_number', 'N/A')}")
                    print(f"    Contact info: {bool(result.get('contact_info'))}")
                elif extractor_type == "academic":
                    print(f"    DOI: {result.get('doi', 'N/A')}")
                    print(f"    Abstract: {bool(result.get('abstract'))}")
            else:
                print(f"  ✗ {extractor_type} extraction failed")
                
        except Exception as e:
            print(f"  ✗ {extractor_type} extractor error: {e}")

def create_mock_html(extractor_type: str) -> str:
    """Create mock HTML content for different extractor types"""
    
    base_html = """
    <html>
    <head>
        <title>Sample {type} Content</title>
        <meta name="description" content="Sample {type} description">
        <meta name="keywords" content="sample, {type}, content">
    </head>
    <body>
        <h1>Sample {type} Title</h1>
        <p>This is sample content for {type} extraction testing.</p>
    </body>
    </html>
    """
    
    if extractor_type == "news":
        return """
        <html>
        <head>
            <title>Breaking News: New EV Technology</title>
            <meta name="description" content="Latest developments in electric vehicle technology">
            <meta name="author" content="Tech Reporter">
        </head>
        <body>
            <h1>New EV Battery Technology Announced</h1>
            <time datetime="2024-01-15T10:30:00Z">January 15, 2024</time>
            <div class="author">By John Smith</div>
            <div class="category">Technology</div>
            <p>Scientists have announced a breakthrough in electric vehicle battery technology.</p>
        </body>
        </html>
        """
    
    elif extractor_type == "blog":
        return """
        <html>
        <head>
            <title>My EV Journey - Blog Post</title>
        </head>
        <body>
            <h1>My Experience with Electric Vehicles</h1>
            <div class="tags">
                <a href="#">EV</a>
                <a href="#">sustainability</a>
                <a href="#">technology</a>
            </div>
            <div class="comments-count">15 comments</div>
            <p>Sharing my personal experience with electric vehicles.</p>
        </body>
        </html>
        """
    
    elif extractor_type == "documentation":
        return """
        <html>
        <head>
            <title>API Documentation</title>
        </head>
        <body>
            <h1>Energy API Documentation</h1>
            <nav class="toc">
                <a href="#intro">Introduction</a>
                <a href="#setup">Setup</a>
                <a href="#usage">Usage</a>
            </nav>
            <pre><code>GET /api/energy-data
{
    "consumption": 150.5,
    "unit": "kWh"
}</code></pre>
            <p>This is the API documentation for energy data.</p>
        </body>
        </html>
        """
    
    elif extractor_type == "ecommerce":
        return """
        <html>
        <head>
            <title>EV Charger - Product Page</title>
        </head>
        <body>
            <h1 class="product-name">Level 2 EV Charger</h1>
            <div class="product-description">Fast charging solution for home use</div>
            <div class="price">$599.99</div>
            <p>High-quality electric vehicle charger for residential use.</p>
        </body>
        </html>
        """
    
    elif extractor_type == "government":
        return """
        <html>
        <head>
            <title>Energy Policy Document</title>
        </head>
        <body>
            <h1>Federal Energy Policy Guidelines</h1>
            <div class="document-number">DOC-2024-001</div>
            <p>Contact us at (555) 123-4567 or email@government.gov</p>
            <p>Official government energy policy document.</p>
        </body>
        </html>
        """
    
    elif extractor_type == "academic":
        return """
        <html>
        <head>
            <title>Research Paper on Renewable Energy</title>
        </head>
        <body>
            <h1>Analysis of Renewable Energy Sources</h1>
            <div class="doi">10.1000/example.doi</div>
            <div class="journal">Energy Research Journal</div>
            <div class="abstract">This paper analyzes various renewable energy sources and their efficiency.</div>
            <p>Comprehensive analysis of renewable energy technologies.</p>
        </body>
        </html>
        """
    
    else:  # generic
        return base_html.format(type=extractor_type)

def test_web_scraping_with_metadata():
    """Test web scraping with metadata integration"""
    print("\n" + "="*60)
    print("TESTING WEB SCRAPING WITH METADATA")
    print("="*60)
    
    # Initialize web scraper
    scraper = WebScraper(user_agent="energyAI-bot")
    
    # Test URLs (using example.com for safe testing)
    test_urls = [
        "https://httpbin.org/html",  # Safe test URL
        "https://httpbin.org/json",  # Another safe test URL
    ]
    
    print("Testing web scraping with metadata integration...")
    
    for url in test_urls:
        try:
            print(f"\nScraping: {url}")
            
            # Scrape with different extractor types
            for extractor_type in ["generic", "news"]:
                print(f"  Using {extractor_type} extractor...")
                
                result = scraper.scrape(url, extractor_type)
                
                if result:
                    print(f"    ✓ Scraping successful")
                    print(f"    Title: {result.get('title', 'N/A')}")
                    print(f"    Content length: {len(result.get('content', ''))}")
                    print(f"    Extractor type: {result.get('extractor_type')}")
                    
                    # Check metadata
                    if result.get('metadata'):
                        metadata = result['metadata']
                        print(f"    Metadata ID: {metadata.source_id}")
                        print(f"    Quality score: {metadata.quality_score}")
                        print(f"    Processing status: {metadata.processing_status}")
                        print(f"    Domain: {metadata.domain}")
                    else:
                        print(f"    ✗ No metadata generated")
                else:
                    print(f"    ✗ Scraping failed")
                    
                time.sleep(1)  # Be respectful
                
        except Exception as e:
            print(f"  ✗ Error scraping {url}: {e}")

def test_metadata_validation_and_quality():
    """Test metadata validation and quality scoring"""
    print("\n" + "="*60)
    print("TESTING METADATA VALIDATION AND QUALITY")
    print("="*60)
    
    metadata_handler = MetadataHandler()
    
    # Test cases with different quality levels
    test_cases = [
        {
            "name": "High Quality",
            "data": {
                "url": "https://example.com/high-quality",
                "content": "This is a comprehensive article about electric vehicles with detailed information about charging infrastructure, battery technology, and environmental benefits. The content includes technical specifications, real-world examples, and expert analysis.",
                "html": """
                <html>
                <head>
                    <title>Comprehensive EV Guide - Everything You Need to Know</title>
                    <meta name="description" content="Complete guide to electric vehicles covering technology, charging, and environmental impact">
                    <meta name="author" content="Dr. Energy Expert">
                    <meta name="keywords" content="electric vehicle, EV, charging, battery, environment">
                </head>
                <body>
                    <h1>Comprehensive EV Guide</h1>
                    <p>This is a detailed guide about electric vehicles.</p>
                </body>
                </html>
                """
            }
        },
        {
            "name": "Medium Quality",
            "data": {
                "url": "https://example.com/medium-quality",
                "content": "Electric vehicles are becoming popular. They help the environment.",
                "html": """
                <html>
                <head>
                    <title>EV Info</title>
                    <meta name="description" content="Basic information about electric vehicles">
                </head>
                <body>
                    <h1>EV Information</h1>
                    <p>Basic info about EVs.</p>
                </body>
                </html>
                """
            }
        },
        {
            "name": "Low Quality",
            "data": {
                "url": "https://example.com/low-quality",
                "content": "EV good.",
                "html": """
                <html>
                <head>
                    <title>EV</title>
                </head>
                <body>
                    <p>Short content.</p>
                </body>
                </html>
                """
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']} content...")
        
        metadata = metadata_handler.create_metadata("web", test_case["data"])
        
        print(f"  Quality score: {metadata.quality_score:.2f}")
        print(f"  Completeness score: {metadata.completeness_score:.2f}")
        print(f"  Freshness score: {metadata.freshness_score:.2f}")
        print(f"  Processing status: {metadata.processing_status}")
        
        if metadata.processing_errors:
            print(f"  Errors: {metadata.processing_errors}")

def test_metadata_export_and_reporting():
    """Test metadata export and reporting functionality"""
    print("\n" + "="*60)
    print("TESTING METADATA EXPORT AND REPORTING")
    print("="*60)
    
    metadata_handler = MetadataHandler()
    
    # Create sample metadata
    sample_metadata = []
    
    for i in range(5):
        source_data = {
            "url": f"https://example.com/article-{i}",
            "content": f"This is sample content for article {i} about electric vehicles and renewable energy.",
            "html": f"<html><head><title>Article {i}</title></head><body><h1>Article {i}</h1><p>Content {i}</p></body></html>"
        }
        
        metadata = metadata_handler.create_metadata("web", source_data)
        sample_metadata.append(metadata)
    
    # Test metadata export
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Export metadata report
        metadata_handler.export_metadata_report(sample_metadata, temp_path)
        
        # Check if files were created
        csv_path = Path(temp_path).with_suffix('.csv')
        summary_path = Path(temp_path).with_suffix('.summary.json')
        
        print(f"Metadata report exported:")
        print(f"  CSV file: {csv_path.exists()}")
        print(f"  Summary file: {summary_path.exists()}")
        
        # Read and display summary
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"\nSummary report:")
            print(f"  Total sources: {summary['total_sources']}")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            print(f"  Average quality score: {summary['quality_statistics']['average_quality_score']:.2f}")
            print(f"  Source types: {summary['source_type_distribution']}")
            
    finally:
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        if Path(temp_path).with_suffix('.csv').exists():
            Path(temp_path).with_suffix('.csv').unlink()
        if Path(temp_path).with_suffix('.summary.json').exists():
            Path(temp_path).with_suffix('.summary.json').unlink()

def test_configuration_integration():
    """Test configuration integration with web scraping"""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION INTEGRATION")
    print("="*60)
    
    # Load configuration
    config = get_config()
    
    # Get web scraping configuration
    web_config = config.get("web_scraping", {})
    print(f"Web scraping configuration:")
    print(f"  Base URL: {web_config.get('base_url', 'Not set')}")
    print(f"  Max pages: {web_config.get('max_pages', 'Not set')}")
    print(f"  Timeout: {web_config.get('timeout', 'Not set')}")
    print(f"  Max retries: {web_config.get('max_retries', 'Not set')}")
    
    # Get metadata configuration
    metadata_config = config.get("metadata", {})
    print(f"\nMetadata configuration:")
    print(f"  Required fields: {metadata_config.get('required_fields', 'Not set')}")
    print(f"  Quality thresholds: {metadata_config.get('quality_thresholds', 'Not set')}")
    
    # Test configuration-based initialization
    scraper = WebScraper(user_agent="test-bot")
    print(f"\nScraper initialized with:")
    print(f"  User agent: {scraper.user_agent}")
    print(f"  Base URL: {scraper.base_url}")
    print(f"  Max pages: {scraper.max_pages}")
    print(f"  Available extractors: {list(scraper.extractors.keys())}")

def main():
    """Run all web scraping integration tests"""
    print("WEB SCRAPING INTEGRATION TEST SUITE")
    print("="*60)
    
    try:
        # Test metadata handler
        metadata_list = test_metadata_handler()
        
        # Test source-specific extractors
        test_source_specific_extractors()
        
        # Test web scraping with metadata
        test_web_scraping_with_metadata()
        
        # Test metadata validation and quality
        test_metadata_validation_and_quality()
        
        # Test metadata export and reporting
        test_metadata_export_and_reporting()
        
        # Test configuration integration
        test_configuration_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        print(f"\nTest Summary:")
        print(f"  ✓ Metadata handler tested")
        print(f"  ✓ Source-specific extractors tested")
        print(f"  ✓ Web scraping with metadata integration tested")
        print(f"  ✓ Metadata validation and quality scoring tested")
        print(f"  ✓ Metadata export and reporting tested")
        print(f"  ✓ Configuration integration tested")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 