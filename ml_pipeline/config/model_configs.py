#!/usr/bin/env python3
"""
Model configurations for the EV Charging LLM Pipeline
"""

from typing import Dict, Any, List

# Available model configurations
MODEL_CONFIGS = {
    "dialogpt-medium": {
        "name": "microsoft/DialoGPT-medium",
        "params": "345M",
        "target_modules": ["c_attn", "c_proj", "wte", "wpe"],
        "max_length": 512,
        "description": "DialoGPT-medium (345M parameters, open access, proven to work)"
    },
    "dialogpt-small": {
        "name": "microsoft/DialoGPT-small",
        "params": "117M",
        "target_modules": ["c_attn", "c_proj", "wte", "wpe"],
        "max_length": 512,
        "description": "DialoGPT-small (117M parameters, lightweight option)"
    },
    "gpt2": {
        "name": "gpt2",
        "params": "124M",
        "target_modules": ["c_attn", "c_proj", "wte", "wpe"],
        "max_length": 512,
        "description": "GPT-2 base model (124M parameters, widely used)"
    },
    "llama-2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "params": "7B",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 2048,
        "description": "Llama-2-7B (requires Hugging Face access)"
    },
    "llama-2-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "params": "13B",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_length": 2048,
        "description": "Llama-2-13B (requires Hugging Face access)"
    }
}

def get_model_config(model_key: str) -> Dict[str, Any]:
    """
    Get model configuration by key
    
    Args:
        model_key: Key identifying the model configuration
        
    Returns:
        Dictionary containing model configuration
        
    Raises:
        ValueError: If model_key is not found
    """
    if model_key not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_key}' not found. Available models: {available_models}")
    
    return MODEL_CONFIGS[model_key]

def list_available_models() -> List[str]:
    """
    Get list of available model keys
    
    Returns:
        List of available model keys
    """
    return list(MODEL_CONFIGS.keys())

def get_model_info(model_key: str) -> Dict[str, Any]:
    """
    Get detailed model information
    
    Args:
        model_key: Key identifying the model configuration
        
    Returns:
        Dictionary containing detailed model information
    """
    config = get_model_config(model_key)
    return {
        "key": model_key,
        "name": config["name"],
        "parameters": config["params"],
        "description": config["description"],
        "target_modules": config["target_modules"],
        "max_length": config["max_length"]
    }

def validate_model_config(model_key: str) -> bool:
    """
    Validate if a model configuration is complete
    
    Args:
        model_key: Key identifying the model configuration
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        config = get_model_config(model_key)
        required_fields = ["name", "params", "target_modules", "max_length"]
        return all(field in config for field in required_fields)
    except ValueError:
        return False 