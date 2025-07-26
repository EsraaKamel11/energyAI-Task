#!/usr/bin/env python3
"""
QLoRA Main Orchestrator
Orchestrates the complete QLoRA training pipeline using modular components
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import modular components
from .model_loader import load_qlora_model, QLoRAModelLoader
from .lora_config import setup_qlora, QLoRAConfigurator
from .data_preparation import prepare_qlora_data, QLoRADataPreparer
from .training_loop import train_qlora_model, QLoRATrainingConfig, QLoRATrainer
from .experiment_tracker import QLoRAExperimentTracker, create_qlora_experiment_tracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QLoRAOrchestrator:
    """Main orchestrator for QLoRA training pipeline"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_loader = QLoRAModelLoader()
        self.lora_configurator = QLoRAConfigurator()
        self.data_preparer = QLoRADataPreparer()
        self.trainer = QLoRATrainer()
        self.experiment_tracker = None

    def run_training_pipeline(
        self,
        model_name: str = "gpt2",
        data_path: str = "pipeline_output/qa_pairs.jsonl",
        output_dir: str = None,
        quantization: str = "4bit",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        batch_size: int = 1,
        num_epochs: int = 4,
        learning_rate: float = 1e-4,
        system_prompt: str = None,
    ) -> dict:
        """
        Run complete QLoRA training pipeline

        Args:
            model_name: Name of the model to load
            data_path: Path to training data
            output_dir: Output directory
            quantization: Quantization type
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            system_prompt: System prompt for conversations

        Returns:
            dict: Training results
        """
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"qlora_training_{model_name}_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"🚀 Starting QLoRA training pipeline")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Data: {data_path}")
        self.logger.info(f"Output: {output_dir}")

        # Initialize experiment tracking
        self.experiment_tracker = create_qlora_experiment_tracker(
            model_name=model_name,
            quantization=quantization,
            project_name="ev-charging-qlora",
        )

        # Start experiment
        experiment_started = self.experiment_tracker.start_experiment(
            tags=["qlora", "ev-charging", "fine-tuning"]
        )

        if experiment_started:
            self.logger.info("✅ Experiment tracking started with W&B")
        else:
            self.logger.info("⚠️ Experiment tracking disabled (no WANDB_API_KEY)")

        try:
            # Step 1: Load model with quantization
            self.logger.info("📥 Step 1: Loading model with quantization...")
            model, tokenizer = self.model_loader.load_model_with_quantization(
                model_name, quantization
            )

            # Get model info
            model_info = self.model_loader.get_model_info(model)
            self.logger.info(f"Model info: {model_info}")

            # Log model info to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_model_info(model_info)

            # Step 2: Setup QLoRA configuration
            self.logger.info("🎯 Step 2: Setting up QLoRA configuration...")
            model = self.lora_configurator.setup_qlora_model(
                model, model_name, lora_r, lora_alpha, lora_dropout
            )

            # Get LoRA info
            lora_info = self.lora_configurator.get_lora_info(model)
            self.logger.info(f"LoRA info: {lora_info}")

            # Log LoRA configuration
            if self.experiment_tracker:
                self.experiment_tracker.log_training_config(
                    {
                        "lora_r": lora_r,
                        "lora_alpha": lora_alpha,
                        "lora_dropout": lora_dropout,
                        "quantization": quantization,
                    }
                )

            # Step 3: Prepare training data
            self.logger.info("📊 Step 3: Preparing training data...")
            train_dataset, val_dataset = self.data_preparer.prepare_training_data(
                data_path, tokenizer, system_prompt
            )

            # Get dataset stats
            train_stats = self.data_preparer.get_dataset_stats(train_dataset)
            val_stats = self.data_preparer.get_dataset_stats(val_dataset)
            self.logger.info(f"Train stats: {train_stats}")
            self.logger.info(f"Validation stats: {val_stats}")

            # Log dataset info to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_dataset_info(
                    {
                        "train": train_stats,
                        "validation": val_stats,
                        "data_path": data_path,
                    }
                )

            # Step 4: Configure training
            self.logger.info("⚙️ Step 4: Configuring training...")
            training_config = QLoRATrainingConfig(
                output_dir=str(output_path / "checkpoints"),
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                gradient_accumulation_steps=8,
                save_steps=500,
                eval_steps=500,
                logging_steps=10,
            )

            # Step 5: Train the model
            self.logger.info("🚀 Step 5: Starting training...")
            # Create trainer with the training config
            trainer = QLoRATrainer(training_config)
            training_results = trainer.train(
                model,
                tokenizer,
                train_dataset,
                val_dataset,
                str(output_path / "checkpoints"),
            )

            # Step 6: Save final model
            self.logger.info("💾 Step 6: Saving final model...")
            final_model_path = output_path / "final_model"
            model.save_pretrained(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))

            # Step 7: Generate training summary
            self.logger.info("📋 Step 7: Generating training summary...")
            summary = self._generate_training_summary(
                model_info,
                lora_info,
                train_stats,
                val_stats,
                training_results,
                final_model_path,
            )

            # Save summary
            summary_path = output_path / "training_summary.json"
            import json

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"✅ QLoRA training pipeline completed successfully!")
            self.logger.info(f"Final model saved to: {final_model_path}")
            self.logger.info(f"Training summary saved to: {summary_path}")

            # Log final results to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_final_summary(summary)
                self.experiment_tracker.log_artifacts(
                    str(summary_path), "training_summary", "summary"
                )
                self.experiment_tracker.finish_experiment()

                run_url = self.experiment_tracker.get_run_url()
                if run_url:
                    self.logger.info(f"🔗 Experiment tracked at: {run_url}")

            return summary

        except Exception as e:
            self.logger.error(f"❌ QLoRA training pipeline failed: {e}")
            raise

    def _generate_training_summary(
        self,
        model_info,
        lora_info,
        train_stats,
        val_stats,
        training_results,
        model_path,
    ) -> dict:
        """Generate comprehensive training summary"""
        return {
            "training_completed": True,
            "timestamp": datetime.now().isoformat(),
            "model_info": model_info,
            "lora_info": lora_info,
            "dataset_stats": {"train": train_stats, "validation": val_stats},
            "training_results": training_results,
            "model_path": str(model_path),
            "next_steps": [
                "Run evaluation: python run_evaluation.py --model_path <model_path>",
                "Deploy model: python start_server.py --model_path <model_path>",
                "Compare with baseline: python run_evaluation.py --model_path <model_path> --compare_baseline",
            ],
        }


def main():
    """Main function to run QLoRA training"""
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA Training Pipeline")
    parser.add_argument("--model", default="gpt2", help="Model name to load")
    parser.add_argument(
        "--data", default="pipeline_output/qa_pairs.jsonl", help="Path to training data"
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--quantization",
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="Quantization type",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please ensure your pipeline has generated QA data")
        return

    # Run training pipeline
    orchestrator = QLoRAOrchestrator()
    results = orchestrator.run_training_pipeline(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        quantization=args.quantization,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("QLoRA TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Final Loss: {results['training_results']['train_loss']:.4f}")
    print(f"Training Time: {results['training_results']['train_runtime']:.2f}s")
    print(f"Model Path: {results['model_path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
