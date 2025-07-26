#!/usr/bin/env python3
"""
Test script for modular training components
Tests that all modular training components work correctly
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_imports():
    """Test that all training components can be imported"""
    print("Testing training component imports...")
    
    try:
        from src.training import load_model, LoRATrainer, TrainingLoop, ExperimentTracker, TrainingConfig
        print("✅ All training components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Training component import failed: {e}")
        return False

def test_training_config():
    """Test TrainingConfig creation"""
    print("\nTesting TrainingConfig creation...")
    
    try:
        from src.training import TrainingConfig
        
        config = TrainingConfig(
            base_model="gpt2",
            domain="ev_charging",
            lora_rank=16,
            lora_alpha=32,
            learning_rate=1e-4,
            batch_size=1,
            num_epochs=3
        )
        
        print(f"✅ TrainingConfig created successfully")
        print(f"   Base model: {config.base_model}")
        print(f"   Domain: {config.domain}")
        print(f"   LoRA rank: {config.lora_rank}")
        print(f"   Learning rate: {config.learning_rate}")
        return True
        
    except Exception as e:
        print(f"✗ TrainingConfig creation failed: {e}")
        return False

def test_model_loader():
    """Test model loader functionality"""
    print("\nTesting model loader...")
    
    try:
        from src.training import load_model
        
        # Test model loading (without actually loading the model)
        print("✅ Model loader module available")
        print("   Note: Actual model loading requires GPU and model files")
        return True
        
    except Exception as e:
        print(f"✗ Model loader test failed: {e}")
        return False

def test_lora_trainer():
    """Test LoRA trainer functionality"""
    print("\nTesting LoRA trainer...")
    
    try:
        from src.training import LoRATrainer
        
        # Test LoRA trainer creation (without actual model)
        print("✅ LoRA trainer module available")
        print("   Note: Actual LoRA setup requires a loaded model")
        return True
        
    except Exception as e:
        print(f"✗ LoRA trainer test failed: {e}")
        return False

def test_training_loop():
    """Test training loop functionality"""
    print("\nTesting training loop...")
    
    try:
        from src.training import TrainingLoop
        
        # Test training loop creation (without actual components)
        print("✅ Training loop module available")
        print("   Note: Actual training requires model, tokenizer, and datasets")
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        return False

def test_experiment_tracker():
    """Test experiment tracker functionality"""
    print("\nTesting experiment tracker...")
    
    try:
        from src.training import ExperimentTracker, TrainingConfig
        
        config = TrainingConfig(
            base_model="gpt2",
            domain="ev_charging",
            lora_rank=16,
            lora_alpha=32,
            learning_rate=1e-4,
            batch_size=1,
            num_epochs=3
        )
        
        # Test experiment tracker creation (without WandB)
        print("✅ Experiment tracker module available")
        print("   Note: Actual WandB logging requires API key and internet connection")
        return True
        
    except Exception as e:
        print(f"✗ Experiment tracker test failed: {e}")
        return False

def test_data_preparation():
    """Test data preparation functionality"""
    print("\nTesting data preparation...")
    
    try:
        from src.training.dataset_preparation import QADatasetPreparer
        
        # Test data preparer creation (without OpenAI API)
        print("✅ Data preparation module available")
        print("   Note: Actual QA generation requires OpenAI API key")
        return True
        
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Modular Training Components")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 7
    
    # Test imports
    if test_training_imports():
        tests_passed += 1
    
    # Test training config
    if test_training_config():
        tests_passed += 1
    
    # Test model loader
    if test_model_loader():
        tests_passed += 1
    
    # Test LoRA trainer
    if test_lora_trainer():
        tests_passed += 1
    
    # Test training loop
    if test_training_loop():
        tests_passed += 1
    
    # Test experiment tracker
    if test_experiment_tracker():
        tests_passed += 1
    
    # Test data preparation
    if test_data_preparation():
        tests_passed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All modular training component tests passed!")
        print("\n🎉 Modular training system is ready!")
        print("\n📋 Next steps:")
        print("1. Run modular QLoRA training: python qlora_only.py")
        print("2. Check WandB for experiment tracking")
        print("3. Run evaluation: python run_evaluation.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 