#!/usr/bin/env python3
"""
QLoRA Training Loop
Handles training configuration and execution
"""

import logging
import torch
from pathlib import Path
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class QLoRATrainingConfig:
    """Configuration for QLoRA training"""

    def __init__(
        self,
        output_dir: str = "outputs",
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 50,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: int = 3,
        mixed_precision: str = "bf16",
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        optim: str = "adamw_bnb_8bit",
        seed: int = 42,
    ):

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        self.mixed_precision = mixed_precision
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.optim = optim
        self.seed = seed

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to TrainingArguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            bf16=(self.mixed_precision == "bf16"),
            fp16=(self.mixed_precision == "fp16"),
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optim,
            seed=self.seed,
            report_to="wandb",  # Enable wandb for experiment tracking
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )


class QLoRATrainer:
    """QLoRA training orchestrator"""

    def __init__(self, config: QLoRATrainingConfig = None):
        self.config = config or QLoRATrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.accelerator = Accelerator(mixed_precision=self.config.mixed_precision)

    def setup_training(self, model, tokenizer, train_dataset, val_dataset) -> Trainer:
        """
        Setup training components

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset

        Returns:
            Trainer: Configured trainer
        """
        self.logger.info("Setting up QLoRA training...")

        # Create training arguments
        training_args = self.config.to_training_arguments()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        self.logger.info("✅ Training setup complete")
        return trainer

    def train(
        self, model, tokenizer, train_dataset, val_dataset, output_dir: str = None
    ) -> dict:
        """
        Execute training

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory for checkpoints

        Returns:
            dict: Training results
        """
        if output_dir:
            self.config.output_dir = output_dir

        # Setup training
        trainer = self.setup_training(model, tokenizer, train_dataset, val_dataset)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        # Train
        self.logger.info("🚀 Starting QLoRA training...")
        train_result = trainer.train()

        # Save final model
        final_output_dir = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(final_output_dir))
        tokenizer.save_pretrained(str(final_output_dir))

        # Get training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        self.logger.info(f"✅ Training completed! Model saved to {final_output_dir}")

        return {
            "train_loss": metrics.get("train_loss", 0),
            "train_runtime": metrics.get("train_runtime", 0),
            "train_samples_per_second": metrics.get("train_samples_per_second", 0),
            "model_path": str(final_output_dir),
        }

    def evaluate(self, trainer: Trainer) -> dict:
        """
        Evaluate the trained model

        Args:
            trainer: Trainer with trained model

        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating trained model...")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        self.logger.info(
            f"✅ Evaluation completed! Loss: {metrics.get('eval_loss', 0):.4f}"
        )

        return metrics


def train_qlora_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: QLoRATrainingConfig = None,
    output_dir: str = None,
) -> dict:
    """
    Convenience function to train a QLoRA model

    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_dir: Output directory

    Returns:
        dict: Training results
    """
    trainer = QLoRATrainer(config)
    return trainer.train(model, tokenizer, train_dataset, val_dataset, output_dir)


if __name__ == "__main__":
    # Test the training loop
    logging.basicConfig(level=logging.INFO)

    try:
        # Test configuration
        config = QLoRATrainingConfig(batch_size=1, num_epochs=1, learning_rate=1e-4)

        training_args = config.to_training_arguments()
        print("✅ Training configuration test successful!")
        print(f"Output dir: {training_args.output_dir}")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")

        # Test trainer setup
        trainer = QLoRATrainer(config)
        print("✅ Trainer setup test successful!")

    except Exception as e:
        print(f"❌ Training loop test failed: {e}")
