#!/usr/bin/env python3
"""
Simple import test to verify all components can be imported correctly.
Run this from the ml_pipeline directory.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test importing all major components."""
    
    print("Testing imports...")
    
    try:
        print("1. Testing configuration manager...")
        from src.utils.config_manager import ConfigManager
        print("   ✅ ConfigManager imported successfully")
        
        print("2. Testing error handling...")
        from src.utils.error_handling import retry_with_fallback, circuit_breaker
        print("   ✅ Error handling imported successfully")
        
        print("3. Testing error classification...")
        from src.utils.error_classification import classify_and_handle_error
        print("   ✅ Error classification imported successfully")
        
        print("4. Testing memory management...")
        from src.utils.memory_manager import MemoryManager, memory_safe
        print("   ✅ Memory management imported successfully")
        
        print("5. Testing web scraper...")
        from src.data_collection.web_scraper import WebScraper
        print("   ✅ Web scraper imported successfully")
        
        print("6. Testing PDF extractor...")
        from src.data_collection.pdf_extractor import PDFExtractor
        print("   ✅ PDF extractor imported successfully")
        
        print("7. Testing deduplicator...")
        from src.data_processing.deduplication import Deduplicator
        print("   ✅ Deduplicator imported successfully")
        
        print("8. Testing evaluation metrics...")
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        print("   ✅ Evaluation metrics imported successfully")
        
        print("9. Testing metadata handler...")
        from src.data_collection.metadata_handler import MetadataHandler
        print("   ✅ Metadata handler imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    
    print("\nTesting basic functionality...")
    
    try:
        # Import classes for testing
        from src.utils.config_manager import ConfigManager
        from src.data_collection.web_scraper import WebScraper
        from src.data_collection.pdf_extractor import PDFExtractor
        from src.data_processing.deduplication import Deduplicator
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        
        # Test configuration manager
        print("1. Testing configuration manager...")
        config = ConfigManager()
        print(f"   ✅ Config loaded: {config.get('pipeline.name')}")
        
        # Test web scraper initialization
        print("2. Testing web scraper initialization...")
        scraper = WebScraper()
        print("   ✅ Web scraper initialized")
        
        # Test PDF extractor initialization
        print("3. Testing PDF extractor initialization...")
        extractor = PDFExtractor()
        print("   ✅ PDF extractor initialized")
        
        # Test deduplicator initialization
        print("4. Testing deduplicator initialization...")
        deduplicator = Deduplicator()
        print("   ✅ Deduplicator initialized")
        
        # Test evaluation metrics initialization
        print("5. Testing evaluation metrics initialization...")
        evaluator = EvaluationMetrics()
        print("   ✅ Evaluation metrics initialized")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE IMPORT TEST")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED!")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("❌ FUNCTIONALITY TESTS FAILED!")
            print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ IMPORT TESTS FAILED!")
        print("=" * 50) 
