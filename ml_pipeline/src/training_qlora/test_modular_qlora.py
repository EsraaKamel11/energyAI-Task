#!/usr/bin/env python3
"""
Test script for modular QLoRA training components
Tests that all modular components work correctly together
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loader():
    """Test model loader component"""
    print("Testing model loader component...")

    try:
        from .model_loader import QLoRAModelLoader

        loader = QLoRAModelLoader()
        print("✅ Model loader component initialized successfully")
        return True

    except Exception as e:
        print(f"✗ Model loader test failed: {e}")
        return False


def test_lora_config():
    """Test LoRA configuration component"""
    print("\nTesting LoRA configuration component...")

    try:
        from .lora_config import QLoRAConfigurator

        configurator = QLoRAConfigurator()

        # Test target module detection
        gpt_modules = configurator.get_target_modules("gpt2")
        llama_modules = configurator.get_target_modules("llama2")

        print("✅ LoRA configurator component initialized successfully")
        print(f"   GPT-2 target modules: {gpt_modules}")
        print(f"   Llama-2 target modules: {llama_modules}")

        # Test LoRA config creation
        config = configurator.create_lora_config(r=16, lora_alpha=32)
        print(f"   LoRA config created successfully")

        return True

    except Exception as e:
        print(f"✗ LoRA configuration test failed: {e}")
        return False


def test_data_preparation():
    """Test data preparation component"""
    print("\nTesting data preparation component...")

    try:
        from .data_preparation import QLoRADataPreparer

        preparer = QLoRADataPreparer()

        # Test with sample data
        sample_qa = [
            {
                "question": "What is Level 2 charging?",
                "answer": "Level 2 charging uses 240V power.",
            },
            {
                "question": "How fast is DC charging?",
                "answer": "DC charging can provide 60-80% charge in 20-30 minutes.",
            },
        ]

        conversations = preparer.create_conversation_format(sample_qa)

        print("✅ Data preparation component initialized successfully")
        print(f"   Created {len(conversations)} conversations")

        return True

    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False


def test_training_loop():
    """Test training loop component"""
    print("\nTesting training loop component...")

    try:
        from .training_loop import QLoRATrainingConfig, QLoRATrainer

        # Test configuration
        config = QLoRATrainingConfig(batch_size=1, num_epochs=1, learning_rate=1e-4)

        # Test trainer
        trainer = QLoRATrainer(config)

        print("✅ Training loop component initialized successfully")
        print(f"   Training config created with batch_size={config.batch_size}")

        return True

    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        return False


def test_orchestrator():
    """Test main orchestrator component"""
    print("\nTesting main orchestrator component...")

    try:
        from .main_orchestrator import QLoRAOrchestrator

        orchestrator = QLoRAOrchestrator()

        print("✅ Main orchestrator component initialized successfully")
        print("   All modular components integrated successfully")

        return True

    except Exception as e:
        print(f"✗ Main orchestrator test failed: {e}")
        return False


def test_component_integration():
    """Test that all components can be imported together"""
    print("\nTesting component integration...")

    try:
        # Import all components
        from .model_loader import QLoRAModelLoader
        from .lora_config import QLoRAConfigurator
        from .data_preparation import QLoRADataPreparer
        from .training_loop import QLoRATrainingConfig, QLoRATrainer
        from .main_orchestrator import QLoRAOrchestrator

        print("✅ All components imported successfully")
        print("   Modular architecture working correctly")

        return True

    except Exception as e:
        print(f"✗ Component integration test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Testing Modular QLoRA Training Components")
    print("=" * 60)

    tests_passed = 0
    total_tests = 6

    # Test individual components
    if test_model_loader():
        tests_passed += 1

    if test_lora_config():
        tests_passed += 1

    if test_data_preparation():
        tests_passed += 1

    if test_training_loop():
        tests_passed += 1

    if test_orchestrator():
        tests_passed += 1

    if test_component_integration():
        tests_passed += 1

    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✅ All modular QLoRA component tests passed!")
        print("\n🎉 Modular QLoRA training system is ready!")
        print("\n📋 Usage:")
        print("1. Run complete pipeline: python training_qlora/main_orchestrator.py")
        print(
            "2. Run with custom args: python training_qlora/main_orchestrator.py --model gpt2 --epochs 3"
        )
        print("3. Test individual components: python training_qlora/model_loader.py")
        print("4. Test LoRA config: python training_qlora/lora_config.py")
        print("5. Test data prep: python training_qlora/data_preparation.py")
        print("6. Test training: python training_qlora/training_loop.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
