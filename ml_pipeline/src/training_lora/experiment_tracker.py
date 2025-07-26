import wandb
import logging
import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from transformers import TrainerCallback
import torch
import numpy as np
from datetime import datetime

@dataclass
class TrainingConfig:
    """Configuration for training experiment"""
    base_model: str
    domain: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4
    batch_size: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    fp16: bool = True
    dataloader_pin_memory: bool = False

class WandbCallback(TrainerCallback):
    """Custom callback for logging to Weights & Biases"""
    
    def __init__(self, experiment_tracker: 'ExperimentTracker'):
        self.tracker = experiment_tracker
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.logger.info("Starting training - logging to WandB")
        self.tracker.log_config(args)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs:
            # Add step information
            if state.global_step is not None:
                logs["global_step"] = state.global_step
            if state.epoch is not None:
                logs["epoch"] = state.epoch
            
            # Log to WandB
            self.tracker.log_metrics(logs, step=state.global_step)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            self.tracker.log_evaluation_metrics(metrics, step=state.global_step)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when model is saved"""
        if state.best_model_checkpoint:
            self.tracker.log_checkpoint(state.best_model_checkpoint, "best")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.logger.info("Training completed - finalizing WandB run")
        self.tracker.finish_run()

class ExperimentTracker:
    """Enhanced experiment tracker with comprehensive WandB integration"""
    
    def __init__(self, 
                 project: str = "ev-charging-finetune",
                 run_name: str = None,
                 config: Optional[TrainingConfig] = None,
                 tags: Optional[List[str]] = None):
        """
        Initialize experiment tracker
        
        Args:
            project: WandB project name
            run_name: Optional run name (auto-generated if None)
            config: Training configuration
            tags: List of tags for the experiment
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.tags = tags or []
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.domain}_{config.base_model.split('/')[-1]}_{timestamp}"
        
        # Initialize WandB
        try:
            wandb.init(
                project=project,
                name=run_name,
                tags=self.tags,
                config=asdict(config) if config else {},
                reinit=True
            )
            self.logger.info(f"Initialized WandB run: {wandb.run.name} (ID: {wandb.run.id})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_available = False
        else:
            self.wandb_available = True
    
    def log_config(self, args) -> None:
        """Log training arguments and configuration"""
        if not self.wandb_available:
            return
        
        config_dict = {
            "training_args": vars(args),
            "model_config": asdict(self.config) if self.config else {}
        }
        wandb.config.update(config_dict, allow_val_change=True)
        self.logger.info("Logged training configuration to WandB")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB"""
        if not self.wandb_available:
            return
        
        try:
            wandb.log(metrics, step=step)
            if step:
                self.logger.debug(f"Logged metrics at step {step}: {metrics}")
            else:
                self.logger.debug(f"Logged metrics: {metrics}")
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log evaluation metrics with prefix"""
        if not self.wandb_available:
            return
        
        # Add eval prefix to metrics
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        self.log_metrics(eval_metrics, step=step)
        self.logger.info(f"Logged evaluation metrics: {metrics}")
    
    def log_checkpoint(self, checkpoint_path: str, checkpoint_type: str = "regular") -> None:
        """Log model checkpoint to WandB"""
        if not self.wandb_available:
            return
        
        try:
            # Save checkpoint to WandB
            artifact = wandb.Artifact(
                name=f"model-{checkpoint_type}-{wandb.run.id}",
                type="model",
                description=f"{checkpoint_type.capitalize()} model checkpoint"
            )
            artifact.add_dir(checkpoint_path)
            wandb.log_artifact(artifact)
            self.logger.info(f"Logged {checkpoint_type} checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log checkpoint: {e}")
    
    def log_dataset_info(self, dataset_stats: Dict[str, Any]) -> None:
        """Log dataset statistics"""
        if not self.wandb_available:
            return
        
        wandb.log({"dataset_stats": dataset_stats})
        self.logger.info(f"Logged dataset statistics: {dataset_stats}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information"""
        if not self.wandb_available:
            return
        
        wandb.log({"model_info": model_info})
        self.logger.info(f"Logged model information: {model_info}")
    
    def log_training_summary(self, summary: Dict[str, Any]) -> None:
        """Log training summary at the end"""
        if not self.wandb_available:
            return
        
        wandb.log({"training_summary": summary})
        self.logger.info(f"Logged training summary: {summary}")
    
    def log_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log final evaluation metrics"""
        if not self.wandb_available:
            return
        
        # Log final metrics
        wandb.log({"final_metrics": metrics})
        
        # Also log individual metrics for easier tracking
        for key, value in metrics.items():
            wandb.log({f"final_{key}": value})
        
        self.logger.info(f"Logged final metrics: {metrics}")
    
    def log_hyperparameter_sweep(self, sweep_config: Dict[str, Any]) -> None:
        """Log hyperparameter sweep configuration"""
        if not self.wandb_available:
            return
        
        wandb.log({"sweep_config": sweep_config})
        self.logger.info(f"Logged hyperparameter sweep config: {sweep_config}")
    
    def log_gradient_norms(self, grad_norms: List[float], step: int) -> None:
        """Log gradient norms for monitoring training stability"""
        if not self.wandb_available:
            return
        
        wandb.log({
            "gradient_norm": np.mean(grad_norms),
            "gradient_norm_std": np.std(grad_norms),
            "gradient_norm_max": np.max(grad_norms),
            "gradient_norm_min": np.min(grad_norms)
        }, step=step)
    
    def log_learning_rate(self, lr: float, step: int) -> None:
        """Log learning rate schedule"""
        if not self.wandb_available:
            return
        
        wandb.log({"learning_rate": lr}, step=step)
    
    def log_memory_usage(self, memory_stats: Dict[str, Any], step: int) -> None:
        """Log GPU memory usage"""
        if not self.wandb_available:
            return
        
        wandb.log(memory_stats, step=step)
    
    def create_artifact(self, name: str, type: str, description: str, path: str) -> None:
        """Create and log a custom artifact"""
        if not self.wandb_available:
            return
        
        try:
            artifact = wandb.Artifact(name=name, type=type, description=description)
            if os.path.isfile(path):
                artifact.add_file(path)
            elif os.path.isdir(path):
                artifact.add_dir(path)
            wandb.log_artifact(artifact)
            self.logger.info(f"Created artifact: {name} ({type})")
        except Exception as e:
            self.logger.warning(f"Failed to create artifact: {e}")
    
    def finish_run(self) -> None:
        """Finish the WandB run"""
        if not self.wandb_available:
            return
        
        try:
            wandb.finish()
            self.logger.info("Finished WandB run")
        except Exception as e:
            self.logger.warning(f"Failed to finish WandB run: {e}")
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL of the current run"""
        if not self.wandb_available:
            return None
        
        try:
            return wandb.run.get_url()
        except Exception:
            return None
    
    def log_custom_chart(self, chart_name: str, chart_data: Dict[str, Any]) -> None:
        """Log custom charts to WandB"""
        if not self.wandb_available:
            return
        
        try:
            wandb.log({chart_name: wandb.plot_table(
                "wandb/line/v0",
                wandb.Table(data=chart_data["data"], columns=chart_data["columns"])
            )})
        except Exception as e:
            self.logger.warning(f"Failed to log custom chart: {e}") 