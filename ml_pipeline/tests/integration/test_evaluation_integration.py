#!/usr/bin/env python3
"""
Test script for evaluation integration
Tests that evaluation functions work correctly from the evaluation engine
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evaluation_components_import():
    """Test that modular evaluation components can be imported"""
    print("Testing modular evaluation components import...")
    
    try:
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        from src.evaluation.performance_tester import PerformanceTester
        from src.evaluation.benchmark_creator import BenchmarkCreator
        print("✅ Modular evaluation components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Modular evaluation components import failed: {e}")
        return False

def test_evaluation_components_initialization():
    """Test that evaluation components can be initialized"""
    print("\nTesting evaluation components initialization...")
    
    try:
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        from src.evaluation.performance_tester import PerformanceTester
        from src.evaluation.benchmark_creator import BenchmarkCreator
        
        metrics = EvaluationMetrics()
        tester = PerformanceTester()
        creator = BenchmarkCreator()
        print("✅ Evaluation components initialized successfully")
        
        # Check if basic methods exist
        if hasattr(metrics, 'calculate_basic_metrics'):
            print("✅ calculate_basic_metrics method found")
        else:
            print("✗ calculate_basic_metrics method not found")
            return False
            
        if hasattr(tester, 'test_model_performance'):
            print("✅ test_model_performance method found")
        else:
            print("✗ test_model_performance method not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Evaluation components initialization failed: {e}")
        return False

def test_benchmark_creator_import():
    """Test that benchmark creator can be imported"""
    print("\nTesting benchmark creator import...")
    
    try:
        from src.evaluation.benchmark_creator import BenchmarkCreator
        print("✅ Benchmark creator imported successfully")
        return True
    except Exception as e:
        print(f"✗ Benchmark creator import failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Evaluation Integration")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test evaluation components import
    if test_evaluation_components_import():
        tests_passed += 1
    
    # Test evaluation components initialization
    if test_evaluation_components_initialization():
        tests_passed += 1
    
    # Test benchmark creator import
    if test_benchmark_creator_import():
        tests_passed += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All evaluation integration tests passed!")
        print("\n📋 Next steps:")
        print("1. Run QLoRA training: python qlora_only.py")
        print("2. Run comprehensive evaluation: python run_evaluation.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 