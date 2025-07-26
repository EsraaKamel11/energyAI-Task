from pydantic_settings import BaseSettings
from pathlib import Path
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, List

# Try to import yaml, but provide fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

load_dotenv()

class Settings(BaseSettings):
    # Core config fields
    domain: str = "electric vehicle charging stations"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    train_batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 3e-5
    use_gpu: bool = True
    log_level: str = "INFO"
    
    # Additional fields from config.yaml
    data_sources: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    evaluation: Dict[str, Any] = {}
    deployment: Dict[str, Any] = {}
    prompts: Dict[str, Any] = {}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields

# Load YAML config and override defaults
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists() and YAML_AVAILABLE:
    try:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load YAML config: {e}")
        yaml_config = {}
elif not YAML_AVAILABLE:
    logging.warning("YAML not available, using default configuration")
    yaml_config = {}
else:
    yaml_config = {}

settings = Settings(**yaml_config)

# Logging configuration
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ml_pipeline") 
