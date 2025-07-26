#!/usr/bin/env python3
"""
Test script to verify quantized imports work correctly
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

# Test bitsandbytes specifically
try:
    import bitsandbytes as bnb
    print("✅ BitsAndBytes import successful")
except Exception as e:
    print(f"❌ BitsAndBytes import failed: {e}")

# Test CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("⚠️  CUDA not available, will use CPU")

# Test quantization config
try:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("✅ BitsAndBytes config created successfully")
except Exception as e:
    print(f"❌ BitsAndBytes config failed: {e}")

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

print("\n🎉 Quantized imports test completed!")
print("\nNext steps:")
print("1. Run: python ev_charging_qlora_quantized.py")
print("2. After training, evaluate with: python evaluate_ev_charging.py --adapter_dir path/to/adapter") 