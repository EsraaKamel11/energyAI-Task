#!/usr/bin/env python3
"""
QLoRA Experiment Tracker
Handles experiment tracking with Weights & Biases
"""

import os
import logging
import wandb
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QLoRAExperimentConfig:
    """Configuration for QLoRA experiment tracking"""

    project_name: str = "ev-charging-qlora"
    experiment_name: str = None
    model_name: str = "gpt2"
    quantization: str = "4bit"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-4
    domain: str = "electric_vehicle_charging"

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"qlora_{self.model_name}_{timestamp}"


class QLoRAExperimentTracker:
    """Experiment tracker for QLoRA training"""

    def __init__(self, config: QLoRAExperimentConfig = None):
        self.config = config or QLoRAExperimentConfig()
        self.wandb_run = None
        self.logger = logging.getLogger(__name__)

    def start_experiment(self, tags: list = None) -> bool:
        """Start a new experiment"""
        try:
            # Check if WANDB_API_KEY is set
            if not os.getenv("WANDB_API_KEY"):
                self.logger.warning(
                    "WANDB_API_KEY not found. Skipping experiment tracking."
                )
                return False

            # Initialize wandb
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=asdict(self.config),
                tags=tags or ["qlora", "ev-charging", "fine-tuning"],
                notes=f"QLoRA fine-tuning for {self.config.domain} domain",
            )

            self.logger.info(f"Started experiment: {self.config.experiment_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start experiment: {e}")
            return False

    def log_training_config(self, training_config: Dict[str, Any]):
        """Log training configuration"""
        if self.wandb_run:
            wandb.config.update(training_config)
            self.logger.info("Logged training configuration")

    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        if self.wandb_run:
            wandb.log({"model_info": model_info})
            self.logger.info("Logged model information")

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        if self.wandb_run:
            wandb.log({"dataset_info": dataset_info})
            self.logger.info("Logged dataset information")

    def log_training_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training metrics"""
        if self.wandb_run:
            wandb.log(metrics, step=step)

    def log_evaluation_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics"""
        if self.wandb_run:
            wandb.log({"evaluation": metrics})
            self.logger.info("Logged evaluation metrics")

    def log_comparison_results(self, comparison_results: Dict[str, Any]):
        """Log model comparison results"""
        if self.wandb_run:
            wandb.log({"comparison": comparison_results})
            self.logger.info("Logged comparison results")

    def log_artifacts(
        self, file_path: str, artifact_name: str, artifact_type: str = "model"
    ):
        """Log artifacts (models, datasets, etc.)"""
        if self.wandb_run:
            try:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    description=f"QLoRA {artifact_type} for {self.config.domain}",
                )
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)
                self.logger.info(f"Logged artifact: {artifact_name}")
            except Exception as e:
                self.logger.error(f"Failed to log artifact: {e}")

    def log_final_summary(self, summary: Dict[str, Any]):
        """Log final experiment summary"""
        if self.wandb_run:
            wandb.log({"final_summary": summary})
            self.logger.info("Logged final experiment summary")

    def finish_experiment(self):
        """Finish the experiment"""
        if self.wandb_run:
            wandb.finish()
            self.logger.info("Finished experiment")

    def get_run_url(self) -> Optional[str]:
        """Get the wandb run URL"""
        if self.wandb_run:
            return self.wandb_run.url
        return None


def create_qlora_experiment_tracker(
    model_name: str = "gpt2",
    quantization: str = "4bit",
    project_name: str = "ev-charging-qlora",
    experiment_name: str = None,
) -> QLoRAExperimentTracker:
    """Convenience function to create experiment tracker"""

    config = QLoRAExperimentConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        model_name=model_name,
        quantization=quantization,
    )

    return QLoRAExperimentTracker(config)


# Test function
if __name__ == "__main__":
    # Test experiment tracker
    tracker = create_qlora_experiment_tracker("gpt2", "4bit")

    if tracker.start_experiment():
        tracker.log_training_config({"batch_size": 1, "epochs": 3})
        tracker.log_model_info({"parameters": "124M", "quantization": "4bit"})
        tracker.log_training_metrics({"loss": 2.5, "accuracy": 0.8})
        tracker.finish_experiment()
        print("✅ Experiment tracker test completed successfully!")
    else:
        print("⚠️ Experiment tracker test skipped (no WANDB_API_KEY)")
