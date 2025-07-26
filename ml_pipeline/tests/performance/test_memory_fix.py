#!/usr/bin/env python3
"""
Test script to verify memory fixes work
"""

import os
import sys
import torch
import logging

# Windows multiprocessing fix
if os.name == 'nt':  # Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    print("✅ Windows multiprocessing fix applied")

# Test basic imports
try:
    from transformers import GPT2Tokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print("✅ Transformers imports successful")
except Exception as e:
    print(f"❌ Transformers import failed: {e}")

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    print("✅ PEFT imports successful")
except Exception as e:
    print(f"❌ PEFT import failed: {e}")

try:
    from datasets import Dataset
    print("✅ Datasets import successful")
except Exception as e:
    print(f"❌ Datasets import failed: {e}")

# Test CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  CUDA not available, will use CPU")

# Test memory allocation
try:
    if torch.cuda.is_available():
        # Test small tensor allocation
        test_tensor = torch.randn(100, 100, device='cuda')
        print("✅ GPU memory allocation test successful")
        del test_tensor
        torch.cuda.empty_cache()
    else:
        test_tensor = torch.randn(100, 100)
        print("✅ CPU memory allocation test successful")
        del test_tensor
except Exception as e:
    print(f"❌ Memory allocation test failed: {e}")

print("\n🎉 Memory fix test completed!") 