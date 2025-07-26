#!/usr/bin/env python3
"""
Test script for updated main_clean.py with QLoRA integration
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qlora_imports():
    """Test that QLoRA components can be imported"""
    print("Testing QLoRA component imports...")
    
    try:
        # Test modular QLoRA imports
        from training_qlora.model_loader import QLoRAModelLoader
        from training_qlora.lora_config import QLoRAConfigurator
        from training_qlora.data_preparation import QLoRADataPreparer
        from training_qlora.training_loop import QLoRATrainer, QLoRATrainingConfig
        from training_qlora.main_orchestrator import QLoRAOrchestrator
        from training_qlora.experiment_tracker import QLoRAExperimentTracker, create_qlora_experiment_tracker
        
        print("✅ All QLoRA components imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ QLoRA import test failed: {e}")
        return False

def test_main_clean_imports():
    """Test that main_clean.py imports work"""
    print("\nTesting main_clean.py imports...")
    
    try:
        # Test main_clean imports
        from main_clean import create_sample_data, main
        
        print("✅ main_clean.py imports successful")
        return True
        
    except Exception as e:
        print(f"✗ main_clean.py import test failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation"""
    print("\nTesting sample data creation...")
    
    try:
        from main_clean import create_sample_data
        
        sample_data = create_sample_data()
        print(f"✅ Sample data created with {len(sample_data)} documents")
        print(f"   Sample text: {sample_data['text'].iloc[0][:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Sample data creation failed: {e}")
        return False

def test_qlora_orchestrator():
    """Test QLoRA orchestrator initialization"""
    print("\nTesting QLoRA orchestrator...")
    
    try:
        from src.training_qlora.main_orchestrator import QLoRAOrchestrator
        
        orchestrator = QLoRAOrchestrator()
        print("✅ QLoRA orchestrator initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ QLoRA orchestrator test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Updated main_clean.py with QLoRA Integration")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 4
    
    # Test QLoRA imports
    if test_qlora_imports():
        tests_passed += 1
    
    # Test main_clean imports
    if test_main_clean_imports():
        tests_passed += 1
    
    # Test sample data creation
    if test_sample_data_creation():
        tests_passed += 1
    
    # Test QLoRA orchestrator
    if test_qlora_orchestrator():
        tests_passed += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
        print("\n🎉 main_clean.py successfully updated with QLoRA integration!")
        print("\n📋 Usage:")
        print("1. Run the updated pipeline: python main_clean.py")
        print("2. The pipeline will now use modular QLoRA training")
        print("3. Check pipeline_output/ for results")
        print("\n🔧 QLoRA Benefits:")
        print("• 75% less memory usage than LoRA")
        print("• Works on consumer GPUs (8GB+ VRAM)")
        print("• Modular, maintainable architecture")
        print("• Easy to test and debug")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 
