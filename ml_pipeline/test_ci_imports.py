#!/usr/bin/env python3
"""
CI-specific import test for GitHub Actions
This script tests imports in the CI environment and provides detailed error reporting
"""

import sys
import os
from pathlib import Path
import traceback

def test_import_with_details(import_statement, description):
    """Test a specific import statement with detailed error reporting"""
    try:
        exec(import_statement)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description}: ImportError - {e}")
        print(f"  This usually means a missing dependency. Try: pip install {e.name}")
        return False
    except ModuleNotFoundError as e:
        print(f"✗ {description}: ModuleNotFoundError - {e}")
        print(f"  Module not found. Check if the file exists and path is correct.")
        return False
    except Exception as e:
        print(f"✗ {description}: {type(e).__name__} - {e}")
        print(f"  Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Test all imports from main.py in CI environment"""
    print("=== CI Import Test for main.py ===\n")
    
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Add project root to path
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == "ml_pipeline" else current_dir
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "ml_pipeline"))
    
    print(f"Added to Python path: {project_root}")
    print(f"Added to Python path: {project_root / 'ml_pipeline'}")
    
    # Test imports in the order they appear in main.py
    imports_to_test = [
        ("import os", "os module"),
        ("import logging", "logging module"),
        ("import sys", "sys module"),
        ("from pathlib import Path", "pathlib.Path"),
        ("import pandas as pd", "pandas"),
        ("import torch", "torch"),
        ("from huggingface_hub import login", "huggingface_hub"),
        ("from ml_pipeline.config.settings import settings, logger", "ml_pipeline.config.settings"),
        ("from ml_pipeline.src.data_processing import DataCleaner, QualityFilter, Normalizer, StorageManager, MetadataHandler, Deduplicator, QAGenerator, QAGenerationConfig", "ml_pipeline.src.data_processing"),
        ("from ml_pipeline.src.training_qlora.lora_config import QLoRAConfigurator", "ml_pipeline.src.training_qlora.lora_config"),
        ("from ml_pipeline.src.training_qlora.data_preparation import QLoRADataPreparer", "ml_pipeline.src.training_qlora.data_preparation"),
        ("from ml_pipeline.src.training_qlora.main_orchestrator import QLoRAOrchestrator", "ml_pipeline.src.training_qlora.main_orchestrator"),
        ("from ml_pipeline.src.training_qlora.experiment_tracker import QLoRAExperimentTracker, create_qlora_experiment_tracker", "ml_pipeline.src.training_qlora.experiment_tracker"),
        ("from ml_pipeline.src.evaluation import BenchmarkCreator, BenchmarkGenerator, Comparator, ModelEvaluator", "ml_pipeline.src.evaluation"),
        ("from ml_pipeline.src.deployment import ModelRegistry", "ml_pipeline.src.deployment"),
        ("from ml_pipeline.model_configs import get_model_config, list_available_models", "ml_pipeline.model_configs"),
        ("from ml_pipeline.src.deployment.monitored_inference_server import MonitoredInferenceServer", "ml_pipeline.src.deployment.monitored_inference_server"),
        ("import subprocess", "subprocess module"),
        ("import requests", "requests module"),
        ("import time", "time module"),
    ]
    
    results = []
    for import_statement, description in imports_to_test:
        result = test_import_with_details(import_statement, description)
        results.append((description, result))
    
    # Summary
    print("\n=== CI Import Test Summary ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for description, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{description}: {status}")
    
    print(f"\nOverall: {passed}/{total} imports successful")
    
    if passed == total:
        print("🎉 All imports successful! main.py should run without import errors in CI.")
    else:
        print("⚠ Some imports failed. Common CI issues and solutions:")
        print("\n1. Missing dependencies:")
        print("   - Run: pip install -r requirements.txt")
        print("   - Check if all dependencies are listed in requirements.txt")
        
        print("\n2. Path issues:")
        print("   - Ensure the project structure is correct")
        print("   - Check if __init__.py files exist in all directories")
        
        print("\n3. Version conflicts:")
        print("   - Check for conflicting package versions")
        print("   - Try using a virtual environment")
        
        print("\n4. File permissions:")
        print("   - Ensure all Python files are executable")
        print("   - Check file ownership in CI environment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 