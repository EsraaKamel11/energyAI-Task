#!/usr/bin/env python3
"""
Test Script for Evaluation Metrics

This script demonstrates the comprehensive evaluation metrics functionality
including ROUGE, BLEU, METEOR, and semantic similarity evaluation.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.evaluation.evaluation_metrics import (
    EvaluationMetrics, 
    quick_rouge_evaluation,
    quick_bleu_evaluation,
    quick_meteor_evaluation
)
from src.utils.config_manager import ConfigManager


def create_sample_data():
    """Create sample predictions and references for testing."""
    
    # Sample EV-related content for realistic testing
    predictions = [
        "Electric vehicles are becoming increasingly popular due to their environmental benefits and cost savings.",
        "The Tesla Model 3 offers a range of 350 miles on a single charge and features advanced autopilot capabilities.",
        "Charging infrastructure is expanding rapidly across the country to support the growing EV market.",
        "Battery technology continues to improve, with new lithium-ion cells offering higher energy density.",
        "Government incentives and tax credits are making electric vehicles more affordable for consumers.",
        "Fast charging stations can recharge an EV battery to 80% capacity in just 30 minutes.",
        "The environmental impact of electric vehicles is significantly lower than traditional gasoline cars.",
        "Range anxiety remains a concern for some potential EV buyers, despite improving battery technology.",
        "Electric motors provide instant torque, making EVs excellent for acceleration and performance.",
        "The cost of electricity for charging an EV is typically much lower than the cost of gasoline."
    ]
    
    references = [
        "Electric vehicles are gaining popularity because they help the environment and save money.",
        "Tesla's Model 3 can travel 350 miles per charge and includes sophisticated autopilot features.",
        "The EV charging network is growing quickly nationwide to accommodate more electric vehicles.",
        "Battery improvements are ongoing, with modern lithium-ion batteries providing better energy storage.",
        "Tax incentives and government rebates are reducing the price barrier for electric vehicle adoption.",
        "DC fast chargers can restore 80% of battery life in approximately 30 minutes.",
        "Electric cars have a much smaller environmental footprint compared to conventional vehicles.",
        "Some consumers worry about running out of battery, even though batteries are getting better.",
        "EVs deliver immediate power, providing superior acceleration and driving performance.",
        "Charging an electric vehicle with electricity costs far less than fueling with gasoline."
    ]
    
    return predictions, references


def test_quick_evaluation_functions():
    """Test the quick evaluation convenience functions."""
    print("=" * 60)
    print("Testing Quick Evaluation Functions")
    print("=" * 60)
    
    predictions, references = create_sample_data()
    
    try:
        # Test quick ROUGE evaluation
        print("\n1. Quick ROUGE Evaluation:")
        rouge_scores = quick_rouge_evaluation(predictions, references)
        for metric, score in rouge_scores.items():
            print(f"   {metric}: {score:.4f}")
        
        # Test quick BLEU evaluation
        print("\n2. Quick BLEU Evaluation:")
        bleu_scores = quick_bleu_evaluation(predictions, references)
        for metric, score in bleu_scores.items():
            print(f"   {metric}: {score:.4f}")
        
        # Test quick METEOR evaluation
        print("\n3. Quick METEOR Evaluation:")
        meteor_scores = quick_meteor_evaluation(predictions, references)
        for metric, score in meteor_scores.items():
            print(f"   {metric}: {score:.4f}")
            
    except ImportError as e:
        print(f"   Warning: {e}")
        print("   Install required libraries: pip install evaluate nltk rouge-score")


def test_comprehensive_evaluation():
    """Test the comprehensive evaluation metrics class."""
    print("\n" + "=" * 60)
    print("Testing Comprehensive Evaluation Metrics")
    print("=" * 60)
    
    # Initialize configuration and evaluation metrics
    config = ConfigManager()
    evaluator = EvaluationMetrics(config)
    
    predictions, references = create_sample_data()
    
    print(f"\nSample Data:")
    print(f"   Number of predictions: {len(predictions)}")
    print(f"   Number of references: {len(references)}")
    
    # Test individual metrics
    print("\n1. Individual Metric Evaluation:")
    
    try:
        # ROUGE evaluation
        print("\n   ROUGE Scores:")
        rouge_results = evaluator.evaluate_rouge(predictions, references)
        for metric, score in rouge_results.items():
            print(f"     {metric}: {score:.4f}")
    except Exception as e:
        print(f"     Error in ROUGE evaluation: {e}")
    
    try:
        # BLEU evaluation
        print("\n   BLEU Scores:")
        bleu_results = evaluator.evaluate_bleu(predictions, references)
        for metric, score in bleu_results.items():
            print(f"     {metric}: {score:.4f}")
    except Exception as e:
        print(f"     Error in BLEU evaluation: {e}")
    
    try:
        # METEOR evaluation
        print("\n   METEOR Scores:")
        meteor_results = evaluator.evaluate_meteor(predictions, references)
        for metric, score in meteor_results.items():
            print(f"     {metric}: {score:.4f}")
    except Exception as e:
        print(f"     Error in METEOR evaluation: {e}")
    
    # Semantic similarity evaluation
    print("\n   Semantic Similarity Scores:")
    semantic_results = evaluator.evaluate_semantic_similarity(predictions, references)
    for metric, score in semantic_results.items():
        print(f"     {metric}: {score:.4f}")
    
    # Test comprehensive evaluation
    print("\n2. Comprehensive Evaluation:")
    try:
        comprehensive_results = evaluator.evaluate_comprehensive(
            predictions, 
            references,
            metrics=['rouge', 'bleu', 'meteor', 'semantic_similarity']
        )
        
        for metric_name, results in comprehensive_results.items():
            print(f"\n   {metric_name.upper()}:")
            if isinstance(results, dict) and 'error' not in results:
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.4f}")
                    else:
                        print(f"     {key}: {value}")
            else:
                print(f"     Error: {results}")
                
    except Exception as e:
        print(f"   Error in comprehensive evaluation: {e}")


def test_performance_tracking():
    """Test performance tracking and statistics."""
    print("\n" + "=" * 60)
    print("Testing Performance Tracking")
    print("=" * 60)
    
    config = ConfigManager()
    evaluator = EvaluationMetrics(config)
    
    predictions, references = create_sample_data()
    
    # Run multiple evaluations to gather performance data
    print("\nRunning multiple evaluations for performance tracking...")
    
    for i in range(3):
        print(f"   Evaluation run {i+1}/3")
        
        try:
            evaluator.evaluate_rouge(predictions, references)
            evaluator.evaluate_bleu(predictions, references)
            evaluator.evaluate_meteor(predictions, references)
            evaluator.evaluate_semantic_similarity(predictions, references)
        except Exception as e:
            print(f"     Error in evaluation run {i+1}: {e}")
    
    # Get performance statistics
    print("\nPerformance Statistics:")
    perf_stats = evaluator.get_performance_stats()
    
    for metric, stats in perf_stats.items():
        print(f"\n   {metric.upper()}:")
        print(f"     Mean evaluation time: {stats['mean_time']:.4f}s")
        print(f"     Standard deviation: {stats['std_time']:.4f}s")
        print(f"     Min time: {stats['min_time']:.4f}s")
        print(f"     Max time: {stats['max_time']:.4f}s")
        print(f"     Total evaluations: {stats['total_evaluations']}")


def test_save_load_functionality():
    """Test saving and loading evaluation results."""
    print("\n" + "=" * 60)
    print("Testing Save/Load Functionality")
    print("=" * 60)
    
    config = ConfigManager()
    evaluator = EvaluationMetrics(config)
    
    predictions, references = create_sample_data()
    
    # Run evaluation
    print("\n1. Running evaluation...")
    try:
        evaluator.evaluate_comprehensive(predictions, references)
        print(f"   Generated {len(evaluator.results)} evaluation results")
    except Exception as e:
        print(f"   Error in evaluation: {e}")
        return
    
    # Test JSON save/load
    print("\n2. Testing JSON save/load:")
    json_file = "test_evaluation_results.json"
    
    try:
        evaluator.save_results(json_file, format='json')
        print(f"   Results saved to {json_file}")
        
        # Create new evaluator and load results
        new_evaluator = EvaluationMetrics(config)
        new_evaluator.load_results(json_file, format='json')
        print(f"   Results loaded: {len(new_evaluator.results)} results")
        
        # Clean up
        os.remove(json_file)
        print("   Test file cleaned up")
        
    except Exception as e:
        print(f"   Error in JSON save/load: {e}")
    
    # Test CSV save/load
    print("\n3. Testing CSV save/load:")
    csv_file = "test_evaluation_results.csv"
    
    try:
        evaluator.save_results(csv_file, format='csv')
        print(f"   Results saved to {csv_file}")
        
        # Create new evaluator and load results
        new_evaluator = EvaluationMetrics(config)
        new_evaluator.load_results(csv_file, format='csv')
        print(f"   Results loaded: {len(new_evaluator.results)} results")
        
        # Clean up
        os.remove(csv_file)
        print("   Test file cleaned up")
        
    except Exception as e:
        print(f"   Error in CSV save/load: {e}")


def test_report_generation():
    """Test report generation functionality."""
    print("\n" + "=" * 60)
    print("Testing Report Generation")
    print("=" * 60)
    
    config = ConfigManager()
    evaluator = EvaluationMetrics(config)
    
    predictions, references = create_sample_data()
    
    # Run evaluation
    print("\n1. Running evaluation for report generation...")
    try:
        evaluator.evaluate_comprehensive(predictions, references)
        print(f"   Generated {len(evaluator.results)} evaluation results")
    except Exception as e:
        print(f"   Error in evaluation: {e}")
        return
    
    # Generate report
    print("\n2. Generating evaluation report:")
    try:
        report = evaluator.generate_report()
        print("   Report generated successfully")
        
        # Save report to file
        report_file = "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"   Report saved to {report_file}")
        
        # Display first few lines of report
        print("\n   Report preview:")
        lines = report.split('\n')[:20]
        for line in lines:
            print(f"     {line}")
        
        if len(report.split('\n')) > 20:
            print("     ... (truncated)")
        
        # Clean up
        os.remove(report_file)
        print("   Report file cleaned up")
        
    except Exception as e:
        print(f"   Error in report generation: {e}")


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    config = ConfigManager()
    evaluator = EvaluationMetrics(config)
    
    print("\n1. Testing with empty inputs:")
    try:
        results = evaluator.evaluate_comprehensive([], [])
        print("   Empty inputs handled gracefully")
    except Exception as e:
        print(f"   Error with empty inputs: {e}")
    
    print("\n2. Testing with mismatched lengths:")
    try:
        results = evaluator.evaluate_comprehensive(["text1"], ["ref1", "ref2"])
        print("   Mismatched lengths handled")
    except ValueError as e:
        print(f"   Expected ValueError: {e}")
    except Exception as e:
        print(f"   Unexpected error: {e}")
    
    print("\n3. Testing with None values:")
    try:
        results = evaluator.evaluate_comprehensive([None, "text"], ["ref1", "ref2"])
        print("   None values handled")
    except Exception as e:
        print(f"   Error with None values: {e}")


def main():
    """Main test function."""
    print("Evaluation Metrics Test Suite")
    print("=" * 60)
    print("This script tests the comprehensive evaluation metrics functionality")
    print("including ROUGE, BLEU, METEOR, and semantic similarity evaluation.")
    print("=" * 60)
    
    # Run all tests
    test_quick_evaluation_functions()
    test_comprehensive_evaluation()
    test_performance_tracking()
    test_save_load_functionality()
    test_report_generation()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)
    print("\nKey Features Tested:")
    print("✓ Quick evaluation functions (ROUGE, BLEU, METEOR)")
    print("✓ Comprehensive evaluation with multiple metrics")
    print("✓ Performance tracking and statistics")
    print("✓ Save/load functionality (JSON and CSV)")
    print("✓ Report generation")
    print("✓ Error handling and edge cases")
    print("\nThe evaluation metrics module is ready for integration!")


if __name__ == "__main__":
    main() 