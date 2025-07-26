#!/usr/bin/env python3
"""
QLoRA Model Loader
Handles model loading with quantization and device management
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


class QLoRAModelLoader:
    """Handles loading and preparing models for QLoRA training"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing QLoRA Model Loader on device: {self.device}")

    def load_model_with_quantization(
        self, model_name: str, quantization: str = "4bit"
    ) -> tuple:
        """
        Load model with specified quantization

        Args:
            model_name: Name of the model to load
            quantization: Quantization type ("4bit", "8bit", or "none")

        Returns:
            tuple: (model, tokenizer)
        """
        logger.info(f"Loading model {model_name} with {quantization} quantization")

        # Configure quantization
        quant_config = self._get_quantization_config(quantization)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training if using quantization
        if quantization != "none":
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")

        logger.info(f"✅ Model {model_name} loaded successfully")
        return model, tokenizer

    def _get_quantization_config(self, quantization: str) -> BitsAndBytesConfig:
        """Get quantization configuration"""
        if quantization == "4bit":
            if self.device == "cpu":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None

    def get_model_info(self, model) -> dict:
        """Get model information for logging"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Approximate size in MB
            "device": str(next(model.parameters()).device),
        }


def load_qlora_model(
    model_name: str, quantization: str = "4bit", device: str = None
) -> tuple:
    """
    Convenience function to load a model for QLoRA training

    Args:
        model_name: Name of the model to load
        quantization: Quantization type ("4bit", "8bit", or "none")
        device: Device to load model on

    Returns:
        tuple: (model, tokenizer)
    """
    loader = QLoRAModelLoader(device)
    return loader.load_model_with_quantization(model_name, quantization)


if __name__ == "__main__":
    # Test the model loader
    logging.basicConfig(level=logging.INFO)

    try:
        model, tokenizer = load_qlora_model("gpt2", quantization="4bit")
        loader = QLoRAModelLoader()
        info = loader.get_model_info(model)

        print("✅ Model loader test successful!")
        print(f"Model info: {info}")

    except Exception as e:
        print(f"❌ Model loader test failed: {e}")
        print("Note: This test requires GPU and model files to be available")
