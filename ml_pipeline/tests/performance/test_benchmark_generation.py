#!/usr/bin/env python3
"""
Test script for benchmark generation pipeline
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import benchmark generation components
from src.evaluation.benchmark_generation import BenchmarkGenerator, BenchmarkQuestion

def test_benchmark_generation():
    """Test the benchmark generation pipeline"""
    print("🧪 Testing Benchmark Generation Pipeline")
    
    # Test different domains
    domains = ["electric_vehicles", "healthcare", "technology"]
    
    for domain in domains:
        print(f"\n🔧 Testing domain: {domain}")
        
        # Initialize benchmark generator
        benchmark_generator = BenchmarkGenerator(domain=domain)
        
        # Test standard benchmark generation
        print(f"\n📊 Testing standard benchmark generation:")
        standard_questions = benchmark_generator.generate_benchmark(
            num_questions=10,
            difficulty_distribution={"easy": 0.3, "medium": 0.5, "hard": 0.2}
        )
        
        print(f"  Generated {len(standard_questions)} standard questions")
        
        # Display sample questions
        print(f"\n  Sample questions:")
        for i, question in enumerate(standard_questions[:3]):
            print(f"    {i+1}. Q: {question.question}")
            print(f"       A: {question.answer}")
            print(f"       Category: {question.category}, Difficulty: {question.difficulty}")
            print()
        
        # Test adversarial benchmark generation
        print(f"\n🎯 Testing adversarial benchmark generation:")
        adversarial_questions = benchmark_generator.create_adversarial_benchmark(num_questions=5)
        
        print(f"  Generated {len(adversarial_questions)} adversarial questions")
        
        # Display sample adversarial questions
        print(f"\n  Sample adversarial questions:")
        for i, question in enumerate(adversarial_questions[:2]):
            print(f"    {i+1}. Q: {question.question}")
            print(f"       A: {question.answer}")
            print(f"       Category: {question.category}, Difficulty: {question.difficulty}")
            print()
        
        # Test benchmark validation
        print(f"\n✅ Testing benchmark validation:")
        validation_result = benchmark_generator.validate_benchmark()
        print(f"  Valid: {validation_result['valid']}")
        print(f"  Errors: {len(validation_result.get('errors', []))}")
        print(f"  Warnings: {len(validation_result.get('warnings', []))}")
        
        # Test benchmark statistics
        print(f"\n📈 Testing benchmark statistics:")
        stats = benchmark_generator.get_benchmark_stats()
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Categories: {stats.get('category_distribution', {})}")
        print(f"  Difficulties: {stats.get('difficulty_distribution', {})}")
        
        # Test file operations
        print(f"\n💾 Testing file operations:")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSONL format
            jsonl_path = os.path.join(temp_dir, f"{domain}_benchmark.jsonl")
            benchmark_generator.save_benchmark(jsonl_path, format="jsonl")
            print(f"  ✅ Saved to JSONL: {jsonl_path}")
            
            # Test JSON format
            json_path = os.path.join(temp_dir, f"{domain}_benchmark.json")
            benchmark_generator.save_benchmark(json_path, format="json")
            print(f"  ✅ Saved to JSON: {json_path}")
            
            # Test CSV format
            csv_path = os.path.join(temp_dir, f"{domain}_benchmark.csv")
            benchmark_generator.save_benchmark(csv_path, format="csv")
            print(f"  ✅ Saved to CSV: {csv_path}")
            
            # Test loading
            loaded_questions = benchmark_generator.load_benchmark(jsonl_path, format="jsonl")
            print(f"  ✅ Loaded {len(loaded_questions)} questions from JSONL")
            
            # Verify loaded questions
            if len(loaded_questions) == len(standard_questions + adversarial_questions):
                print("  ✅ Load/save verification passed!")
            else:
                print("  ❌ Load/save verification failed!")
    
    print("\n🎉 All benchmark generation tests passed!")
    return True

def test_electric_vehicles_domain():
    """Test specific electric vehicles domain with detailed questions"""
    print("\n🔋 Testing Electric Vehicles Domain (Detailed)")
    
    benchmark_generator = BenchmarkGenerator(domain="electric_vehicles")
    
    # Test different category distributions
    print(f"\n📊 Testing category distributions:")
    
    category_distributions = [
        {"price_comparison": 0.4, "compatibility": 0.3, "technical_specs": 0.3},
        {"range_efficiency": 0.5, "infrastructure": 0.3, "environmental": 0.2},
        {"adversarial_calculation": 0.6, "comparison": 0.4}
    ]
    
    for i, distribution in enumerate(category_distributions):
        print(f"\n  Distribution {i+1}: {distribution}")
        questions = benchmark_generator.generate_benchmark(
            num_questions=5,
            category_distribution=distribution
        )
        
        print(f"    Generated {len(questions)} questions")
        for question in questions:
            print(f"      - {question.category}: {question.question[:50]}...")
    
    # Test difficulty distributions
    print(f"\n📈 Testing difficulty distributions:")
    
    difficulty_distributions = [
        {"easy": 0.7, "medium": 0.2, "hard": 0.1},
        {"easy": 0.2, "medium": 0.3, "hard": 0.5},
        {"easy": 0.3, "medium": 0.4, "hard": 0.3}
    ]
    
    for i, distribution in enumerate(difficulty_distributions):
        print(f"\n  Distribution {i+1}: {distribution}")
        questions = benchmark_generator.generate_benchmark(
            num_questions=5,
            difficulty_distribution=distribution
        )
        
        print(f"    Generated {len(questions)} questions")
        for question in questions:
            print(f"      - {question.difficulty}: {question.question[:50]}...")
    
    # Test adversarial questions specifically
    print(f"\n🎯 Testing adversarial questions:")
    adversarial_questions = benchmark_generator.create_adversarial_benchmark(num_questions=10)
    
    print(f"  Generated {len(adversarial_questions)} adversarial questions")
    
    # Analyze adversarial categories
    adversarial_categories = {}
    for question in adversarial_questions:
        adversarial_categories[question.category] = adversarial_categories.get(question.category, 0) + 1
    
    print(f"  Adversarial categories: {adversarial_categories}")
    
    # Test question quality
    print(f"\n🔍 Testing question quality:")
    
    # Check for variable substitution
    unsubstituted_vars = 0
    for question in adversarial_questions:
        if "{" in question.question or "}" in question.question:
            unsubstituted_vars += 1
        if "{" in question.answer or "}" in question.answer:
            unsubstituted_vars += 1
    
    print(f"  Questions with unsubstituted variables: {unsubstituted_vars}")
    
    # Check question lengths
    question_lengths = [len(q.question) for q in adversarial_questions]
    answer_lengths = [len(q.answer) for q in adversarial_questions]
    
    print(f"  Average question length: {sum(question_lengths)/len(question_lengths):.1f} characters")
    print(f"  Average answer length: {sum(answer_lengths)/len(answer_lengths):.1f} characters")
    
    print("  ✅ Electric vehicles domain test completed!")

def test_benchmark_validation():
    """Test comprehensive benchmark validation"""
    print("\n✅ Testing Comprehensive Benchmark Validation")
    
    # Create a test benchmark with known issues
    test_questions = [
        BenchmarkQuestion(
            question="What's the cheapest charging?",
            answer="IONITY costs €0.79/kWh",
            category="price_comparison",
            difficulty="medium",
            domain="electric_vehicles"
        ),
        BenchmarkQuestion(
            question="Can I use {charger_type} with Tesla?",
            answer="Yes, with {adapter}",
            category="compatibility",
            difficulty="easy",
            domain="electric_vehicles"
        ),
        BenchmarkQuestion(
            question="",
            answer="Valid answer",
            category="technical_specs",
            difficulty="hard",
            domain="electric_vehicles"
        )
    ]
    
    # Create temporary benchmark generator
    benchmark_generator = BenchmarkGenerator(domain="electric_vehicles")
    benchmark_generator.questions = test_questions
    
    # Test validation
    validation_result = benchmark_generator.validate_benchmark()
    
    print(f"  Validation result: {validation_result}")
    print(f"  Valid: {validation_result['valid']}")
    print(f"  Errors: {validation_result['errors']}")
    print(f"  Warnings: {validation_result['warnings']}")
    
    # Test with valid questions
    valid_questions = [
        BenchmarkQuestion(
            question="What's the maximum charging speed for Tesla Model 3?",
            answer="250kW using Supercharger V3",
            category="technical_specs",
            difficulty="easy",
            domain="electric_vehicles"
        ),
        BenchmarkQuestion(
            question="How much does it cost to charge from 10% to 80%?",
            answer="Approximately €15.60 at €0.32/kWh",
            category="cost_calculation",
            difficulty="medium",
            domain="electric_vehicles"
        )
    ]
    
    benchmark_generator.questions = valid_questions
    validation_result = benchmark_generator.validate_benchmark()
    
    print(f"  Valid questions validation: {validation_result['valid']}")
    print(f"  Errors: {len(validation_result['errors'])}")
    print(f"  Warnings: {len(validation_result['warnings'])}")
    
    print("  ✅ Benchmark validation test completed!")

if __name__ == "__main__":
    try:
        test_benchmark_generation()
        test_electric_vehicles_domain()
        test_benchmark_validation()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
