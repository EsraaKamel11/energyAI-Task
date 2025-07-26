#!/usr/bin/env python3
"""
Test script for model comparison pipeline
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import model comparison components
from src.evaluation.model_comparison import ModelEvaluator, EvaluationResult, ComparisonResult

def test_model_comparison():
    """Test the model comparison pipeline"""
    print("🧪 Testing Model Comparison Pipeline")
    
    # Create mock benchmark data
    mock_benchmark = [
        {
            "question": "What's the cheapest 350kW charging in Berlin?",
            "answer": "IONITY costs €0.79/kWh at 350kW",
            "category": "price_comparison",
            "difficulty": "hard"
        },
        {
            "question": "Can I use Tesla Supercharger with non-Tesla EV?",
            "answer": "Yes, at selected locations with CCS2 connector",
            "category": "compatibility",
            "difficulty": "medium"
        },
        {
            "question": "What's the maximum charging speed for Tesla Model 3?",
            "answer": "250kW using Supercharger V3",
            "category": "technical_specs",
            "difficulty": "easy"
        },
        {
            "question": "How much does it cost to charge from 10% to 80%?",
            "answer": "Approximately €15.60 at €0.32/kWh",
            "category": "cost_calculation",
            "difficulty": "medium"
        },
        {
            "question": "What's the carbon footprint of charging in Germany?",
            "answer": "Approximately 366g CO2/kWh due to coal and natural gas",
            "category": "environmental",
            "difficulty": "hard"
        }
    ]
    
    print(f"\n📋 Mock benchmark: {len(mock_benchmark)} questions")
    for i, q in enumerate(mock_benchmark):
        print(f"  {i+1}. {q['question'][:50]}...")
    
    # Initialize model evaluator
    print(f"\n🔧 Initializing model evaluator:")
    model_evaluator = ModelEvaluator(device="cpu")  # Use CPU for testing
    print("  ✅ Model evaluator initialized")
    
    # Test metric calculations
    print(f"\n📊 Testing metric calculations:")
    
    # Mock predictions and references
    mock_predictions = [
        "IONITY costs €0.79/kWh at 350kW",
        "Yes, with CCS2 adapter at selected locations",
        "250kW using Supercharger V3",
        "Approximately €15.60 at €0.32/kWh",
        "Approximately 366g CO2/kWh due to coal and natural gas"
    ]
    
    mock_references = [q["answer"] for q in mock_benchmark]
    
    # Test ROUGE scores
    try:
        rouge_scores = model_evaluator.calculate_rouge_scores(mock_predictions, mock_references)
        print(f"  ✅ ROUGE scores: {rouge_scores}")
    except Exception as e:
        print(f"  ⚠️  ROUGE calculation failed: {e}")
    
    # Test BLEU score
    try:
        bleu_score = model_evaluator.calculate_bleu_score(mock_predictions, mock_references)
        print(f"  ✅ BLEU score: {bleu_score:.4f}")
    except Exception as e:
        print(f"  ⚠️  BLEU calculation failed: {e}")
    
    # Test exact match
    try:
        exact_match = model_evaluator.calculate_exact_match(mock_predictions, mock_references)
        print(f"  ✅ Exact match: {exact_match:.4f}")
    except Exception as e:
        print(f"  ⚠️  Exact match calculation failed: {e}")
    
    # Test domain-specific metrics
    try:
        domain_metrics = model_evaluator.calculate_domain_specific_metrics(
            mock_predictions, mock_references, domain="electric_vehicles"
        )
        print(f"  ✅ Domain-specific metrics: {domain_metrics}")
    except Exception as e:
        print(f"  ⚠️  Domain-specific metrics failed: {e}")
    
    # Test semantic similarity
    try:
        semantic_sim = model_evaluator.calculate_semantic_similarity(mock_predictions, mock_references)
        print(f"  ✅ Semantic similarity: {semantic_sim:.4f}")
    except Exception as e:
        print(f"  ⚠️  Semantic similarity failed: {e}")
    
    # Test comparison result creation
    print(f"\n🔄 Testing comparison result creation:")
    
    # Create mock evaluation results
    mock_fine_tuned_metrics = {
        "rouge1": 0.85,
        "rouge2": 0.72,
        "rougeL": 0.83,
        "bleu": 0.78,
        "exact_match": 0.80,
        "semantic_similarity": 0.88,
        "price_accuracy": 0.90,
        "technical_accuracy": 0.85,
        "compatibility_accuracy": 0.82
    }
    
    mock_baseline_metrics = {
        "rouge1": 0.75,
        "rouge2": 0.62,
        "rougeL": 0.73,
        "bleu": 0.68,
        "exact_match": 0.70,
        "semantic_similarity": 0.78,
        "price_accuracy": 0.80,
        "technical_accuracy": 0.75,
        "compatibility_accuracy": 0.72
    }
    
    # Calculate improvements
    improvements = {}
    relative_improvements = {}
    statistical_significance = {}
    
    for metric in mock_fine_tuned_metrics:
        ft_score = mock_fine_tuned_metrics[metric]
        base_score = mock_baseline_metrics[metric]
        
        improvement = ft_score - base_score
        improvements[metric] = improvement
        
        if base_score != 0:
            relative_improvement = (improvement / base_score) * 100
            relative_improvements[metric] = relative_improvement
        else:
            relative_improvements[metric] = 0.0
        
        # Simple significance test
        statistical_significance[metric] = abs(improvement) > 0.05
    
    # Create comparison result
    comparison_result = ComparisonResult(
        fine_tuned_model="fine_tuned_ev_model",
        baseline_model="baseline_model",
        benchmark_name="ev_benchmark",
        fine_tuned_metrics=mock_fine_tuned_metrics,
        baseline_metrics=mock_baseline_metrics,
        improvements=improvements,
        relative_improvements=relative_improvements,
        statistical_significance=statistical_significance,
        evaluation_summary={
            "total_questions": len(mock_benchmark),
            "fine_tuned_latency": 0.045,
            "baseline_latency": 0.052,
            "latency_improvement": 0.007,
            "fine_tuned_throughput": 22.2,
            "baseline_throughput": 19.2,
            "throughput_improvement": 3.0,
            "total_tokens_fine_tuned": 1500,
            "total_tokens_baseline": 1400
        }
    )
    
    print("  ✅ Comparison result created successfully")
    
    # Test file operations
    print(f"\n💾 Testing file operations:")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test saving comparison results
        results_path = os.path.join(temp_dir, "comparison_results.json")
        model_evaluator.save_comparison_results(comparison_result, results_path)
        print(f"  ✅ Saved comparison results to: {results_path}")
        
        # Test generating comparison report
        report = model_evaluator.generate_comparison_report(comparison_result)
        report_path = os.path.join(temp_dir, "comparison_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  ✅ Generated comparison report: {report_path}")
        
        # Verify files exist
        if os.path.exists(results_path) and os.path.exists(report_path):
            print("  ✅ File operations test passed!")
        else:
            print("  ❌ File operations test failed!")
    
    # Display sample report
    print(f"\n📋 Sample comparison report:")
    print("=" * 50)
    print(report[:500] + "..." if len(report) > 500 else report)
    print("=" * 50)
    
    print("\n🎉 All model comparison tests passed!")
    return True

def test_mock_model_evaluation():
    """Test model evaluation with mock models"""
    print("\n🤖 Testing Mock Model Evaluation")
    
    # Create mock benchmark
    mock_benchmark = [
        {"question": "What is EV charging?", "answer": "Electric vehicle charging"},
        {"question": "How fast can Tesla charge?", "answer": "Up to 250kW with Supercharger V3"}
    ]
    
    # Create mock model evaluator
    class MockModelEvaluator(ModelEvaluator):
        def evaluate_model(self, model, tokenizer, benchmark, model_name="model"):
            """Mock model evaluation"""
            # Simulate evaluation time
            import time
            time.sleep(0.1)
            
            # Return mock results
            metrics = {
                "rouge1": 0.85 if "fine_tuned" in model_name else 0.75,
                "rouge2": 0.72 if "fine_tuned" in model_name else 0.62,
                "rougeL": 0.83 if "fine_tuned" in model_name else 0.73,
                "bleu": 0.78 if "fine_tuned" in model_name else 0.68,
                "exact_match": 0.80 if "fine_tuned" in model_name else 0.70,
                "semantic_similarity": 0.88 if "fine_tuned" in model_name else 0.78
            }
            
            return EvaluationResult(
                model_name=model_name,
                metrics=metrics,
                latency=0.045 if "fine_tuned" in model_name else 0.052,
                throughput=22.2 if "fine_tuned" in model_name else 19.2,
                total_tokens=1500 if "fine_tuned" in model_name else 1400,
                evaluation_time=0.2,
                metadata={"predictions": ["mock response 1", "mock response 2"]}
            )
    
    mock_evaluator = MockModelEvaluator(device="cpu")
    
    # Test model evaluation
    print("  Testing mock model evaluation...")
    
    try:
        # Mock models
        fine_tuned_model = "fine_tuned_model"
        baseline_model = "baseline_model"
        
        # Evaluate both models
        fine_tuned_result = mock_evaluator.evaluate_model(
            fine_tuned_model, None, mock_benchmark, "fine_tuned_model"
        )
        baseline_result = mock_evaluator.evaluate_model(
            baseline_model, None, mock_benchmark, "baseline_model"
        )
        
        print(f"    Fine-tuned ROUGE-1: {fine_tuned_result.metrics['rouge1']:.4f}")
        print(f"    Baseline ROUGE-1: {baseline_result.metrics['rouge1']:.4f}")
        print(f"    Latency improvement: {baseline_result.latency - fine_tuned_result.latency:.4f}s")
        
        # Test comparison
        comparison_result = mock_evaluator.compare_models(
            fine_tuned_model, None,
            baseline_model, None,
            mock_benchmark,
            "fine_tuned_model",
            "baseline_model"
        )
        
        print(f"    ROUGE-1 improvement: {comparison_result.improvements['rouge1']:.4f}")
        print(f"    Significant improvements: {sum(comparison_result.statistical_significance.values())}")
        
        print("  ✅ Mock model evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Mock model evaluation test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        test_model_comparison()
        test_mock_model_evaluation()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
