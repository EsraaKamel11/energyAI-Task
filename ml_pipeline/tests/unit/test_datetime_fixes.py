#!/usr/bin/env python3
"""
Test script to verify datetime timezone fixes in metadata handler
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from datetime import datetime, timezone
from data_collection.metadata_handler import Metadata, MetadataHandler

def test_datetime_timezone_fixes():
    """Test the datetime timezone fixes"""
    print("Testing datetime timezone fixes...")
    
    # Test 1: Create metadata with naive datetime
    print("\n1. Testing naive datetime handling...")
    naive_dt = datetime(2024, 1, 1, 12, 0, 0)  # Naive datetime
    metadata = Metadata(
        source_id="test_001",
        source_type="web",
        created_at=naive_dt,
        modified_at=naive_dt,
        scraped_at=naive_dt
    )
    
    # Check if __post_init__ made them timezone-aware
    print(f"Created at: {metadata.created_at} (tzinfo: {metadata.created_at.tzinfo})")
    print(f"Modified at: {metadata.modified_at} (tzinfo: {metadata.modified_at.tzinfo})")
    print(f"Scraped at: {metadata.scraped_at} (tzinfo: {metadata.scraped_at.tzinfo})")
    
    assert metadata.created_at.tzinfo is not None, "Created at should be timezone-aware"
    assert metadata.modified_at.tzinfo is not None, "Modified at should be timezone-aware"
    assert metadata.scraped_at.tzinfo is not None, "Scraped at should be timezone-aware"
    print("✓ Naive datetime conversion works")
    
    # Test 2: Test datetime comparison
    print("\n2. Testing datetime comparison...")
    now = datetime.now(timezone.utc)
    days_old = (now - metadata.modified_at).days
    print(f"Days old: {days_old}")
    print("✓ Datetime comparison works")
    
    # Test 3: Test JSON serialization/deserialization
    print("\n3. Testing JSON serialization...")
    metadata_dict = metadata.to_dict()
    print(f"Serialized created_at: {metadata_dict['created_at']}")
    
    # Test deserialization
    handler = MetadataHandler()
    loaded_metadata = handler.load_metadata_from_dict(metadata_dict)
    print(f"Deserialized created_at: {loaded_metadata.created_at} (tzinfo: {loaded_metadata.created_at.tzinfo})")
    assert loaded_metadata.created_at.tzinfo is not None, "Deserialized datetime should be timezone-aware"
    print("✓ JSON serialization/deserialization works")
    
    # Test 4: Test quality score calculation
    print("\n4. Testing quality score calculation...")
    handler._calculate_quality_scores(metadata)
    print(f"Quality score: {metadata.quality_score}")
    print(f"Freshness score: {metadata.freshness_score}")
    print("✓ Quality score calculation works")
    
    print("\n✅ All datetime timezone fixes are working correctly!")

def test_metadata_handler_methods():
    """Test metadata handler methods that were fixed"""
    print("\nTesting metadata handler methods...")
    
    handler = MetadataHandler()
    
    # Test summary report generation
    metadata_list = [
        Metadata(
            source_id="test_001",
            source_type="web",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            scraped_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            quality_score=0.8,
            completeness_score=0.9,
            freshness_score=0.7,
            processing_status="completed"
        ),
        Metadata(
            source_id="test_002",
            source_type="pdf",
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            scraped_at=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            quality_score=0.6,
            completeness_score=0.7,
            freshness_score=0.5,
            processing_status="completed"
        )
    ]
    
    summary = handler._generate_summary_report(metadata_list)
    print(f"Summary report: {summary}")
    
    # Check that processing_times is included
    assert "processing_times" in summary, "Processing times should be included in summary"
    assert "average_days" in summary["processing_times"], "Average days should be calculated"
    print("✓ Summary report generation works")
    
    print("✅ All metadata handler methods are working correctly!")

if __name__ == "__main__":
    try:
        test_datetime_timezone_fixes()
        test_metadata_handler_methods()
        print("\n🎉 All tests passed! Datetime timezone fixes are working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
