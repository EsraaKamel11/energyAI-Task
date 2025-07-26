#!/usr/bin/env python3
"""
Test script for PDF layout preservation using unstructured library
Demonstrates advanced PDF extraction with layout awareness
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data_collection.pdf_extractor import PDFExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pdf_layout_preservation():
    """Test PDF layout preservation with sample PDFs"""
    
    # Create output directory
    output_dir = "test_outputs/pdf_extraction"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PDF extractor with advanced settings
    extractor = PDFExtractor(
        output_dir=output_dir,
        strategy="hi_res",  # High-resolution strategy for better layout
        infer_table_structure=True,  # Enable table structure inference
        include_metadata=True  # Include document metadata
    )
    
    # Sample PDF paths (you can replace with your actual PDFs)
    sample_pdfs = [
        # Add your PDF paths here
        # "data/sample_documents/sample_document.pdf",
        # "data/sample_documents/technical_specs.pdf",
    ]
    
    # For demonstration, we'll create a mock test
    logger.info("Testing PDF layout preservation...")
    
    # Test with a sample PDF if available
    test_pdf_path = find_sample_pdf()
    
    if test_pdf_path:
        logger.info(f"Found test PDF: {test_pdf_path}")
        
        try:
            # Extract PDF with layout preservation
            extractor.collect([test_pdf_path])
            
            # Get extraction summary
            summary = extractor.get_extraction_summary([test_pdf_path])
            
            # Display results
            print("\n" + "="*60)
            print("PDF LAYOUT PRESERVATION TEST RESULTS")
            print("="*60)
            
            print(f"📄 PDF File: {os.path.basename(test_pdf_path)}")
            print(f"📊 Total Elements: {summary['total_elements']}")
            print(f"📋 Tables Found: {summary['total_tables']}")
            print(f"📝 Text Blocks: {summary['total_text_blocks']}")
            print(f"📚 Total Words: {summary['total_words']}")
            
            # Load and display detailed results
            output_file = os.path.join(output_dir, f"{extractor._sanitize_filename(test_pdf_path)}.json")
            
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print("\n📋 DETAILED EXTRACTION RESULTS:")
                print("-" * 40)
                
                # Display content breakdown
                content = data.get('content', {})
                print(f"Headers: {len(content.get('headers', []))}")
                print(f"Lists: {len(content.get('lists', []))}")
                print(f"Images: {len(content.get('images', []))}")
                
                # Display table information
                tables = content.get('tables', [])
                if tables:
                    print(f"\n📊 TABLE ANALYSIS:")
                    for i, table in enumerate(tables):
                        print(f"  Table {i+1}:")
                        print(f"    Type: {table.get('table_type', 'unknown')}")
                        print(f"    Cells: {table.get('cell_count', 0)}")
                        if 'shape' in table:
                            print(f"    Shape: {table['shape']}")
                
                # Display layout information
                layout_info = data.get('layout_info', {})
                pages = layout_info.get('pages', {})
                print(f"\n📄 PAGE LAYOUT:")
                for page_num, page_data in pages.items():
                    print(f"  Page {page_num}: {page_data['element_count']} elements")
                
                # Display statistics
                stats = data.get('statistics', {})
                print(f"\n📈 STATISTICS:")
                print(f"  Total Characters: {stats.get('total_characters', 0):,}")
                print(f"  Average Table Size: {stats.get('avg_table_size', 0):.1f} cells")
                
                # Save detailed report
                report_file = os.path.join(output_dir, "extraction_report.json")
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"\n✅ Detailed report saved to: {report_file}")
                
            else:
                print(f"❌ Output file not found: {output_file}")
                
        except Exception as e:
            logger.error(f"Error during PDF extraction: {e}")
            print(f"❌ Extraction failed: {e}")
    
    else:
        print("⚠️  No sample PDF found. Please add a PDF file to test with.")
        print("   You can modify the 'sample_pdfs' list in this script to include your PDF paths.")
        
        # Create a demonstration of the capabilities
        demonstrate_capabilities()

def find_sample_pdf() -> str:
    """Find a sample PDF file for testing"""
    
    # Common locations to look for PDFs
    search_paths = [
        "data/",
        "data/sample_documents/",
        "data/raw/",
        "documents/",
        "samples/",
        "."
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if file.lower().endswith('.pdf'):
                    return os.path.join(search_path, file)
    
    return None

def demonstrate_capabilities():
    """Demonstrate PDF layout preservation capabilities"""
    
    print("\n" + "="*60)
    print("PDF LAYOUT PRESERVATION CAPABILITIES")
    print("="*60)
    
    capabilities = [
        "🔍 High-resolution text extraction with layout awareness",
        "📊 Automatic table structure detection and extraction",
        "📋 Header and footer identification",
        "📝 List and bullet point preservation",
        "🖼️ Image detection and metadata extraction",
        "📄 Page-by-page layout information",
        "📈 Comprehensive statistics and analytics",
        "🔄 Resume capability for interrupted extractions",
        "📋 Metadata extraction (author, title, creation date, etc.)",
        "🧹 Text cleaning and formatting preservation"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print("\n📋 SUPPORTED ELEMENT TYPES:")
    element_types = [
        "Text blocks with formatting",
        "Structured tables with rows/columns",
        "Headers and titles",
        "List items and bullet points",
        "Images and figures",
        "Page breaks and layout markers",
        "Addresses and contact information"
    ]
    
    for element_type in element_types:
        print(f"  • {element_type}")
    
    print("\n⚙️ CONFIGURATION OPTIONS:")
    config_options = [
        "Strategy: 'hi_res' (high-resolution), 'fast' (quick), 'ocr_only' (OCR-based)",
        "Table structure inference: Enable/disable automatic table detection",
        "Metadata inclusion: Extract document metadata",
        "Text cleaning: Remove extra whitespace and normalize text",
        "Error handling: Graceful failure with detailed error reporting"
    ]
    
    for option in config_options:
        print(f"  • {option}")

def create_sample_extraction():
    """Create a sample extraction result for demonstration"""
    
    sample_data = {
        "source": "sample_ev_document.pdf",
        "extraction_strategy": "hi_res",
        "total_elements": 45,
        "content": {
            "text_blocks": [
                {
                    "index": 0,
                    "type": "Title",
                    "text": "Electric Vehicle Charging Guide",
                    "cleaned_text": "Electric Vehicle Charging Guide",
                    "length": 28,
                    "word_count": 4,
                    "has_bold": True,
                    "has_italics": False,
                    "has_numbers": False,
                    "has_special_chars": False
                },
                {
                    "index": 1,
                    "type": "NarrativeText",
                    "text": "This comprehensive guide covers all aspects of electric vehicle charging, including CCS2, CHAdeMO, and Type 2 connectors.",
                    "cleaned_text": "This comprehensive guide covers all aspects of electric vehicle charging, including CCS2, CHAdeMO, and Type 2 connectors.",
                    "length": 134,
                    "word_count": 18,
                    "has_bold": False,
                    "has_italics": False,
                    "has_numbers": False,
                    "has_special_chars": False
                }
            ],
            "tables": [
                {
                    "index": 15,
                    "type": "Table",
                    "table_type": "structured",
                    "rows": [
                        ["CCS2", "350kW", "Europe", "Type 2 + DC"],
                        ["CHAdeMO", "150kW", "Japan", "DC only"],
                        ["Type 2", "22kW", "Europe", "AC only"]
                    ],
                    "columns": ["Connector Type", "Max Power", "Region", "Current Type"],
                    "cell_count": 12,
                    "shape": [3, 4]
                }
            ],
            "headers": [
                {
                    "index": 0,
                    "type": "Title",
                    "text": "Electric Vehicle Charging Guide",
                    "cleaned_text": "Electric Vehicle Charging Guide"
                }
            ],
            "lists": [
                {
                    "index": 5,
                    "type": "ListItem",
                    "text": "CCS2 charging supports up to 350kW",
                    "cleaned_text": "CCS2 charging supports up to 350kW"
                },
                {
                    "index": 6,
                    "type": "ListItem",
                    "text": "OCPP protocol enables smart charging",
                    "cleaned_text": "OCPP protocol enables smart charging"
                }
            ],
            "images": [],
            "metadata": {
                "source": "sample_ev_document.pdf",
                "type": "pdf",
                "filename": "sample_ev_document.pdf",
                "file_size": 1024000
            }
        },
        "layout_info": {
            "pages": {
                "1": {
                    "elements": [
                        {"index": 0, "type": "Title", "text_preview": "Electric Vehicle Charging Guide"},
                        {"index": 1, "type": "NarrativeText", "text_preview": "This comprehensive guide covers all aspects..."}
                    ],
                    "element_count": 2
                }
            },
            "structure": {}
        },
        "statistics": {
            "total_text_blocks": 2,
            "total_tables": 1,
            "total_headers": 1,
            "total_lists": 2,
            "total_images": 0,
            "total_words": 22,
            "total_characters": 162,
            "total_table_cells": 12,
            "avg_table_size": 12.0
        }
    }
    
    # Save sample data
    output_dir = "test_outputs/pdf_extraction"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_file = os.path.join(output_dir, "sample_extraction_demo.json")
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Sample extraction data saved to: {sample_file}")
    print("   This demonstrates the structure and capabilities of the layout preservation system.")

if __name__ == "__main__":
    print("🚀 PDF Layout Preservation Test")
    print("Testing advanced PDF extraction with unstructured library...")
    
    try:
        test_pdf_layout_preservation()
        create_sample_extraction()
        
        print("\n✅ PDF layout preservation test completed!")
        print("\n📋 Next steps:")
        print("   1. Add your PDF files to the data/ directory")
        print("   2. Update the sample_pdfs list in this script")
        print("   3. Run the test again to see real extraction results")
        print("   4. Check the test_outputs/pdf_extraction/ directory for results")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}") 
