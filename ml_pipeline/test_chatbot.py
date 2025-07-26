#!/usr/bin/env python3
"""
Test script to check if chatbot functionality works
"""

import os
import sys
import torch
from pathlib import Path

def test_chatbot_functionality():
    """Test the chatbot model loading and generation"""
    print("🧪 Testing Chatbot Functionality...")
    
    # Check if model exists
    model_path = Path("pipeline_output/qlora_training/final_model")
    if not model_path.exists():
        print("❌ Model not found at:", model_path)
        return False
    
    print("✅ Model found at:", model_path)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Imports successful")
        
        # Test model loading
        print("🤖 Loading model...")
        base_model = "microsoft/DialoGPT-medium"
        
        # Load base model and tokenizer
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print("✅ Base model loaded")
        
        # Load fine-tuned adapter
        fine_tuned_model = PeftModel.from_pretrained(base_model_instance, str(model_path))
        print("✅ Fine-tuned model loaded")
        
        # Test generation
        print("💬 Testing generation...")
        test_prompt = "What is Level 2 charging?"
        
        inputs = tokenizer.encode(test_prompt, return_tensors="pt")
        # Move inputs to the same device as the model
        device = next(fine_tuned_model.parameters()).device
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = fine_tuned_model.generate(
                inputs, 
                max_length=150, 
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response (remove input prompt)
        if test_prompt in response:
            response = response.replace(test_prompt, "").strip()
        
        print(f"✅ Generation successful!")
        print(f"📝 Test prompt: {test_prompt}")
        print(f"🤖 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chatbot_functionality()
    if success:
        print("\n🎉 Chatbot test passed!")
    else:
        print("\n💥 Chatbot test failed!")
        sys.exit(1) 