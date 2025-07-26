#!/usr/bin/env python3
"""
Test script to verify Llama model access with authentication
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_llama_access():
    """Test access to Llama models"""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ HUGGINGFACE_TOKEN environment variable not set")
        return None, None, None
    
    # Test different Llama models
    models_to_test = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct", 
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    
    for model_name in models_to_test:
        print(f"\nTesting access to {model_name}...")
        try:
            # Test tokenizer loading
            print("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            print(f"  ✅ Tokenizer loaded successfully")
            print(f"  Model max length: {tokenizer.model_max_length}")
            
            # Test model loading (just config, not full model)
            print("  Loading model config...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                token=token,
                torch_dtype="auto",
                device_map="auto"
            )
            print(f"  ✅ Model loaded successfully")
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model_name, tokenizer, model
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {e}")
            continue
    
    print("\n❌ All models failed to load")
    return None, None, None

if __name__ == "__main__":
    print("Testing Llama model access...")
    model_name, tokenizer, model = test_llama_access()
    
    if model_name:
        print(f"\n🎉 Successfully loaded {model_name}")
        print("You can now proceed with the fine-tuning pipeline!")
    else:
        print("\n❌ No models could be loaded. Please check your authentication.") 