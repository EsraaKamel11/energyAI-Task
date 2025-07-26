#!/usr/bin/env python3
"""
QLoRA Configuration
Handles LoRA configuration and model adaptation
"""

import logging
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


class QLoRAConfigurator:
    """Handles QLoRA configuration and model adaptation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_target_modules(self, model_name: str) -> list:
        """
        Get target modules for LoRA based on model architecture

        Args:
            model_name: Name of the model

        Returns:
            list: Target module names
        """
        model_name_lower = model_name.lower()

        if "gpt" in model_name_lower or "dialogpt" in model_name_lower:
            # GPT models use c_attn and c_proj
            return ["c_attn", "c_proj"]
        elif "llama" in model_name_lower or "mistral" in model_name_lower:
            # Llama and Mistral models use q_proj, k_proj, v_proj, o_proj
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "mpt" in model_name_lower:
            # MPT models use Wqkv and out_proj
            return ["Wqkv", "out_proj"]
        elif "falcon" in model_name_lower:
            # Falcon models use query_key_value and dense
            return ["query_key_value", "dense"]
        else:
            # Default to common attention modules
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def create_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        model_name: str = "gpt2",
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM,
    ) -> LoraConfig:
        """
        Create LoRA configuration

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA (auto-detected if None)
            model_name: Model name for auto-detection
            bias: Bias handling ("none", "all", "lora_only")
            task_type: Task type for LoRA

        Returns:
            LoraConfig: LoRA configuration
        """
        if target_modules is None:
            target_modules = self.get_target_modules(model_name)

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )

        self.logger.info(
            f"Created LoRA config: r={r}, alpha={lora_alpha}, targets={target_modules}"
        )
        return config

    def apply_lora_to_model(self, model, lora_config: LoraConfig) -> object:
        """
        Apply LoRA configuration to model

        Args:
            model: Base model
            lora_config: LoRA configuration

        Returns:
            object: Model with LoRA applied
        """
        self.logger.info("Applying LoRA configuration to model...")

        # Apply LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        model.print_trainable_parameters()

        self.logger.info("✅ LoRA configuration applied successfully")
        return model

    def setup_qlora_model(
        self,
        model,
        model_name: str = "gpt2",
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: str = "none",
    ) -> object:
        """
        Complete QLoRA setup for a model

        Args:
            model: Base model
            model_name: Model name for target module detection
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            bias: Bias handling

        Returns:
            object: Model with QLoRA applied
        """
        # Create LoRA config
        lora_config = self.create_lora_config(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            model_name=model_name,
            bias=bias,
        )

        # Apply to model
        model = self.apply_lora_to_model(model, lora_config)

        return model

    def get_lora_info(self, model) -> dict:
        """
        Get information about LoRA configuration

        Args:
            model: Model with LoRA applied

        Returns:
            dict: LoRA information
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())

        return {
            "trainable_parameters": trainable_params,
            "all_parameters": all_params,
            "trainable_percentage": (
                (trainable_params / all_params) * 100 if all_params > 0 else 0
            ),
        }


def setup_qlora(
    model,
    model_name: str = "gpt2",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> object:
    """
    Convenience function to setup QLoRA on a model

    Args:
        model: Base model
        model_name: Model name
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout

    Returns:
        object: Model with QLoRA applied
    """
    configurator = QLoRAConfigurator()
    return configurator.setup_qlora_model(
        model, model_name, r, lora_alpha, lora_dropout
    )


if __name__ == "__main__":
    # Test the LoRA configurator
    logging.basicConfig(level=logging.INFO)

    try:
        configurator = QLoRAConfigurator()

        # Test target module detection
        gpt_modules = configurator.get_target_modules("gpt2")
        llama_modules = configurator.get_target_modules("llama2")

        print("✅ LoRA configurator test successful!")
        print(f"GPT-2 target modules: {gpt_modules}")
        print(f"Llama-2 target modules: {llama_modules}")

        # Test LoRA config creation
        config = configurator.create_lora_config(r=16, lora_alpha=32)
        print(f"LoRA config created: {config}")

    except Exception as e:
        print(f"❌ LoRA configurator test failed: {e}")
