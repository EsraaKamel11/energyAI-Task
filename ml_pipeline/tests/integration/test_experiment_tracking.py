#!/usr/bin/env python3
"""
Test script for experiment tracking pipeline
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import experiment tracking components
from src.training_lora.experiment_tracker import ExperimentTracker, TrainingConfig, WandbCallback

def test_experiment_tracking():
    """Test the experiment tracking pipeline"""
    print("🧪 Testing Experiment Tracking Pipeline")
    
    # Check if WandB is available
    try:
        import wandb
        wandb_available = True
        print("✅ WandB is available")
    except ImportError:
        wandb_available = False
        print("⚠️  WandB not installed. Running tests without WandB.")
    
    # Create test configuration
    test_config = TrainingConfig(
        base_model="microsoft/DialoGPT-medium",
        domain="electric_vehicles",
        lora_rank=8,
        lora_alpha=16,
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=2
    )
    
    print(f"\n📋 Test configuration:")
    print(f"  Base model: {test_config.base_model}")
    print(f"  Domain: {test_config.domain}")
    print(f"  LoRA rank: {test_config.lora_rank}")
    print(f"  Learning rate: {test_config.learning_rate}")
    print(f"  Batch size: {test_config.batch_size}")
    print(f"  Epochs: {test_config.num_epochs}")
    
    # Test experiment tracker initialization
    print(f"\n🔧 Testing experiment tracker initialization:")
    
    try:
        experiment_tracker = ExperimentTracker(
            project="test-ev-charging",
            config=test_config,
            tags=["test", "lora", "ev-charging"]
        )
        print("  ✅ Experiment tracker initialized successfully")
        
        # Test configuration logging
        print(f"\n📊 Testing configuration logging:")
        experiment_tracker.log_config(test_config)
        print("  ✅ Configuration logged successfully")
        
        # Test dataset info logging
        print(f"\n📈 Testing dataset info logging:")
        dataset_stats = {
            "train_samples": 1000,
            "val_samples": 200,
            "qa_pairs_generated": 500,
            "domain": "electric_vehicles",
            "deduplication_reduction": 25.5
        }
        experiment_tracker.log_dataset_info(dataset_stats)
        print("  ✅ Dataset info logged successfully")
        
        # Test model info logging
        print(f"\n🤖 Testing model info logging:")
        model_info = {
            "base_model": "microsoft/DialoGPT-medium",
            "model_size": 345000000,
            "trainable_params": 8000000,
            "quantization": "4bit"
        }
        experiment_tracker.log_model_info(model_info)
        print("  ✅ Model info logged successfully")
        
        # Test metrics logging
        print(f"\n📊 Testing metrics logging:")
        training_metrics = {
            "loss": 2.5,
            "learning_rate": 1e-4,
            "epoch": 1,
            "step": 100
        }
        experiment_tracker.log_metrics(training_metrics, step=100)
        print("  ✅ Training metrics logged successfully")
        
        # Test evaluation metrics logging
        print(f"\n🎯 Testing evaluation metrics logging:")
        eval_metrics = {
            "eval_loss": 2.1,
            "eval_accuracy": 0.85,
            "eval_rouge": 0.72
        }
        experiment_tracker.log_evaluation_metrics(eval_metrics, step=100)
        print("  ✅ Evaluation metrics logged successfully")
        
        # Test final metrics logging
        print(f"\n🏁 Testing final metrics logging:")
        final_metrics = {
            "final_loss": 1.8,
            "final_accuracy": 0.88,
            "final_rouge": 0.75,
            "latency_avg": 0.045,
            "model_size_mb": 345.0
        }
        experiment_tracker.log_final_metrics(final_metrics)
        print("  ✅ Final metrics logged successfully")
        
        # Test training summary logging
        print(f"\n📋 Testing training summary logging:")
        training_summary = {
            "total_training_steps": 500,
            "final_loss": 1.8,
            "best_eval_loss": 1.7,
            "training_time": 3600,
            "gpu_memory_peak": 8.5
        }
        experiment_tracker.log_training_summary(training_summary)
        print("  ✅ Training summary logged successfully")
        
        # Test artifact creation
        print(f"\n💾 Testing artifact creation:")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test_model.json")
            with open(test_file, 'w') as f:
                json.dump({"model": "test", "version": "1.0"}, f)
            
            experiment_tracker.create_artifact(
                name="test-model-v1",
                type="model",
                description="Test model artifact",
                path=test_file
            )
            print("  ✅ Artifact created successfully")
        
        # Test gradient norms logging
        print(f"\n📈 Testing gradient norms logging:")
        grad_norms = [0.5, 0.6, 0.4, 0.7, 0.3]
        experiment_tracker.log_gradient_norms(grad_norms, step=100)
        print("  ✅ Gradient norms logged successfully")
        
        # Test learning rate logging
        print(f"\n📉 Testing learning rate logging:")
        experiment_tracker.log_learning_rate(1e-4, step=100)
        print("  ✅ Learning rate logged successfully")
        
        # Test memory usage logging
        print(f"\n💾 Testing memory usage logging:")
        memory_stats = {
            "gpu_memory_allocated": 6.2,
            "gpu_memory_reserved": 8.0,
            "cpu_memory_used": 2.1
        }
        experiment_tracker.log_memory_usage(memory_stats, step=100)
        print("  ✅ Memory usage logged successfully")
        
        # Test custom chart logging
        print(f"\n📊 Testing custom chart logging:")
        chart_data = {
            "data": [[1, 2.5], [2, 2.1], [3, 1.8]],
            "columns": ["step", "loss"]
        }
        experiment_tracker.log_custom_chart("training_loss", chart_data)
        print("  ✅ Custom chart logged successfully")
        
        # Test run URL retrieval
        print(f"\n🔗 Testing run URL retrieval:")
        run_url = experiment_tracker.get_run_url()
        if run_url:
            print(f"  ✅ Run URL: {run_url}")
        else:
            print("  ⚠️  Run URL not available")
        
        # Test WandB callback
        print(f"\n🔄 Testing WandB callback:")
        callback = WandbCallback(experiment_tracker)
        print("  ✅ WandB callback created successfully")
        
        # Finish the run
        print(f"\n🏁 Finishing experiment run:")
        experiment_tracker.finish_run()
        print("  ✅ Experiment run finished successfully")
        
    except Exception as e:
        print(f"  ❌ Experiment tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All experiment tracking tests passed!")
    return True

def test_offline_mode():
    """Test experiment tracking in offline mode (without WandB)"""
    print("\n🧪 Testing Offline Mode (No WandB)")
    
    # Create test configuration
    test_config = TrainingConfig(
        base_model="microsoft/DialoGPT-medium",
        domain="electric_vehicles",
        lora_rank=8
    )
    
    try:
        # This should work even without WandB (graceful degradation)
        experiment_tracker = ExperimentTracker(
            project="test-offline",
            config=test_config
        )
        
        # Test that methods don't crash
        experiment_tracker.log_metrics({"test": 1.0})
        experiment_tracker.log_dataset_info({"samples": 100})
        experiment_tracker.log_model_info({"size": 1000000})
        experiment_tracker.finish_run()
        
        print("  ✅ Offline mode works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Offline mode test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_experiment_tracking()
        if success:
            test_offline_mode()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
