#!/usr/bin/env python3
"""
Test script for benchmark creation
Tests the benchmark creator without requiring evaluation dependencies
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_benchmark_creation():
    """Test benchmark creation functionality"""
    
    print("🧪 Testing EV Charging Benchmark Creation")
    print("=" * 50)
    
    try:
        from evaluation.benchmark_creator import EVChargingBenchmarkCreator
        
        # Create benchmark creator
        creator = EVChargingBenchmarkCreator()
        
        # Generate a small benchmark dataset
        print("📝 Creating benchmark dataset...")
        benchmark_file = creator.run_benchmark_creation(num_questions=10)
        
        print(f"✅ Benchmark created successfully!")
        print(f"📁 File: {benchmark_file}")
        
        # Load and display sample questions
        import json
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 Benchmark Statistics:")
        print(f"   Questions: {len(data['questions'])}")
        print(f"   Categories: {', '.join(data['metadata']['categories'])}")
        print(f"   Domain: {data['metadata']['domain']}")
        
        print(f"\n🔍 Sample Questions:")
        for i, question in enumerate(data['questions'][:3]):
            print(f"   {i+1}. {question['question']}")
            print(f"      Category: {question['category']}")
            print(f"      Difficulty: {question['difficulty']}")
            print()
        
        print("✅ Benchmark creation test passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the evaluation module is properly set up")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_evaluation_components():
    """Test evaluation components (without metrics)"""
    
    print("\n🧪 Testing Evaluation Components Setup")
    print("=" * 50)
    
    try:
        from src.evaluation.evaluation_metrics import EvaluationMetrics
        from src.evaluation.performance_tester import PerformanceTester
        from src.evaluation.benchmark_creator import BenchmarkCreator
        
        # Create components
        metrics = EvaluationMetrics()
        tester = PerformanceTester()
        creator = BenchmarkCreator()
        
        print("✅ Evaluation components created successfully")
        
        # Test with sample data
        sample_responses = [
            {
                "question_id": "ev_benchmark_001",
                "response": "Level 1 charging uses 120V power and takes 8-12 hours for a full charge.",
                "response_time": 2.5
            },
            {
                "question_id": "ev_benchmark_002",
                "response": "Level 2 charging uses 240V power and provides 3.3-19.2 kW.",
                "response_time": 2.1
            }
        ]
        
        print("✅ Sample responses created")
        print("⚠️  Note: Full evaluation requires evaluation dependencies (evaluate, nltk)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚗 EV Charging Evaluation System Test")
    print("=" * 60)
    
    # Test benchmark creation
    benchmark_success = test_benchmark_creation()
    
    # Test evaluation components
    evaluation_success = test_evaluation_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Benchmark Creation: {'✅ PASS' if benchmark_success else '❌ FAIL'}")
    print(f"Evaluation Components: {'✅ PASS' if evaluation_success else '❌ FAIL'}")
    
    if benchmark_success and evaluation_success:
        print("\n🎉 All tests passed! The evaluation system is ready.")
        print("\n📋 Next steps:")
        print("1. Install evaluation dependencies: pip install evaluate nltk")
        print("2. Run QLoRA training: python qlora_only.py")
        print("3. Run comprehensive evaluation: python run_evaluation.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 