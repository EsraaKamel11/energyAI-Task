#!/usr/bin/env python3
"""
Basic functionality test for the ML pipeline
"""

import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config.settings import settings, logger
        print("✓ Config settings imported")
    except Exception as e:
        print(f"✗ Config settings import failed: {e}")
        return False
    
    try:
        from src.data_collection import WebScraper, PDFExtractor
        print("✓ Data collection modules imported")
    except Exception as e:
        print(f"✗ Data collection import failed: {e}")
        return False
    
    try:
        from src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager
        print("✓ Data processing modules imported")
    except Exception as e:
        print(f"✗ Data processing import failed: {e}")
        return False
    
    try:
        from src.training import load_model, LoRATrainer, TrainingLoop
        print("✓ Training modules imported")
    except Exception as e:
        print(f"✗ Training import failed: {e}")
        return False
    
    try:
        from src.evaluation import BenchmarkCreator, MetricsCalculator
        print("✓ Evaluation modules imported")
    except Exception as e:
        print(f"✗ Evaluation import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without requiring external dependencies"""
    print("\nTesting basic functionality...")
    
    try:
        # Test data cleaner
        from src.data_processing import DataCleaner
        cleaner = DataCleaner()
        print("✓ DataCleaner initialized")
        
        # Test quality filter
        from src.data_processing import QualityFilter
        quality_filter = QualityFilter(min_length=10)
        print("✓ QualityFilter initialized")
        
        # Test normalizer
        from src.data_processing import Normalizer
        normalizer = Normalizer(model_name="gpt2")
        print("✓ Normalizer initialized")
        
        # Test storage manager
        from src.data_processing import StorageManager
        storage = StorageManager()
        print("✓ StorageManager initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_sample_data():
    """Test with sample data"""
    print("\nTesting with sample data...")
    
    try:
        import pandas as pd
        
        # Create sample data
        sample_texts = [
            "Electric vehicle charging stations are essential infrastructure.",
            "Level 2 charging uses 240-volt power and provides 10-60 miles per hour.",
            "DC fast charging can provide 60-80% charge in 20-30 minutes."
        ]
        
        sample_data = pd.DataFrame({
            "text": sample_texts,
            "source": ["sample"] * len(sample_texts),
            "timestamp": [pd.Timestamp.now()] * len(sample_texts)
        })
        
        print(f"✓ Created sample data with {len(sample_data)} records")
        
        # Test data cleaning
        from src.data_processing import DataCleaner
        cleaner = DataCleaner()
        cleaned_data = cleaner.process(sample_data, text_column="text")
        print(f"✓ Data cleaning completed, {len(cleaned_data)} records")
        
        # Test quality filtering
        from src.data_processing import QualityFilter
        quality_filter = QualityFilter(min_length=20)
        filtered_data = quality_filter.filter(cleaned_data, text_column="text")
        print(f"✓ Quality filtering completed, {len(filtered_data)} records")
        
        # Test normalization
        from src.data_processing import Normalizer
        normalizer = Normalizer(model_name="gpt2")
        normalized_data = normalizer.normalize(filtered_data, text_column="text")
        print(f"✓ Normalization completed")
        
        # Test storage
        from src.data_processing import StorageManager
        storage = StorageManager()
        storage.save_to_parquet(normalized_data, "test_output/sample_processed.parquet")
        print("✓ Data saved to parquet")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== ML Pipeline Basic Functionality Test ===\n")
    
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Sample Data Test", test_sample_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline should work correctly.")
    else:
        print("⚠ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
