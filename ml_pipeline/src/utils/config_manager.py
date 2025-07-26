"""
Configuration Manager for ML Pipeline
Handles loading, validation, and access to centralized configuration
"""

import os
import yaml
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import json


class PipelineConfig(BaseModel):
    """Pipeline configuration model with validation"""

    name: str = Field(default="EV_ML_Pipeline")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="production")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")


class WebScrapingConfig(BaseModel):
    """Web scraping configuration"""

    rate_limit: float = Field(default=1.0, ge=0.1)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=10, ge=1, le=60)
    user_agents: List[str] = Field(default_factory=list)
    respect_robots_txt: bool = Field(default=True)
    circuit_breaker: Dict[str, Any] = Field(default_factory=dict)


class PDFExtractionConfig(BaseModel):
    """PDF extraction configuration"""

    strategy: str = Field(default="hi_res")
    infer_table_structure: bool = Field(default=True)
    include_metadata: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=1, le=10)
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    fallback_strategies: List[str] = Field(default_factory=list)
    validation: Dict[str, bool] = Field(default_factory=dict)


class DeduplicationConfig(BaseModel):
    """Deduplication configuration"""

    method: str = Field(default="hybrid")
    semantic_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    levenshtein_threshold: float = Field(default=0.97, ge=0.0, le=1.0)
    fast_levenshtein_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    hybrid: Dict[str, Any] = Field(default_factory=dict)
    faiss_semantic: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model configuration"""

    semantic: Dict[str, Any] = Field(default_factory=dict)
    qa_generation: Dict[str, Any] = Field(default_factory=dict)
    fine_tuning: Dict[str, Any] = Field(default_factory=dict)


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration"""

    retry: Dict[str, Any] = Field(default_factory=dict)
    circuit_breaker: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)


class PathsConfig(BaseModel):
    """File paths configuration"""

    data: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    config: Dict[str, str] = Field(default_factory=dict)


class ConfigManager:
    """Centralized configuration manager"""

    def __init__(
        self, config_path: Optional[str] = None, environment: Optional[str] = None
    ):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file
            environment: Environment to load (production, development, testing)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or "config/pipeline_config.yaml"
        self.environment = environment or os.getenv("PIPELINE_ENV", "production")
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {self.config_path}")
                self._create_default_config()
                return

            with open(config_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            # Apply environment-specific overrides
            self._apply_environment_overrides()

            # Validate configuration
            self._validate_config()

            # Create directories
            self._create_directories()

            self.logger.info(f"Configuration loaded from {self.config_path}")
            self.logger.info(f"Environment: {self.environment}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()

    def _create_default_config(self) -> None:
        """Create default configuration if file doesn't exist"""
        self.logger.info("Creating default configuration")
        self.config = {
            "pipeline": {
                "name": "EV_ML_Pipeline",
                "version": "1.0.0",
                "environment": self.environment,
                "debug": False,
                "log_level": "INFO",
            },
            "data_processing": {
                "deduplication": {
                    "method": "hybrid",
                    "semantic_threshold": 0.92,
                    "levenshtein_threshold": 0.97,
                }
            },
        }

    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific configuration overrides"""
        if (
            "environments" in self.config
            and self.environment in self.config["environments"]
        ):
            env_config = self.config["environments"][self.environment]
            self._merge_config(self.config, env_config)
            self.logger.info(f"Applied {self.environment} environment overrides")

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _validate_config(self) -> None:
        """Validate configuration structure and values"""
        try:
            # Validate pipeline config
            if "pipeline" in self.config:
                PipelineConfig(**self.config["pipeline"])

            # Validate data collection config
            if "data_collection" in self.config:
                if "web_scraping" in self.config["data_collection"]:
                    WebScrapingConfig(**self.config["data_collection"]["web_scraping"])
                if "pdf_extraction" in self.config["data_collection"]:
                    PDFExtractionConfig(
                        **self.config["data_collection"]["pdf_extraction"]
                    )

            # Validate data processing config
            if "data_processing" in self.config:
                if "deduplication" in self.config["data_processing"]:
                    DeduplicationConfig(
                        **self.config["data_processing"]["deduplication"]
                    )

            # Validate models config
            if "models" in self.config:
                ModelConfig(**self.config["models"])

            # Validate error handling config
            if "error_handling" in self.config:
                ErrorHandlingConfig(**self.config["error_handling"])

            # Validate paths config
            if "paths" in self.config:
                PathsConfig(**self.config["paths"])

            self.logger.info("Configuration validation passed")

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def _create_directories(self) -> None:
        """Create necessary directories from configuration"""
        if "paths" in self.config:
            paths = self.config["paths"]

            # Create data directories
            for path_key, path_value in paths.get("data", {}).items():
                Path(path_value).mkdir(parents=True, exist_ok=True)

            # Create output directories
            for path_key, path_value in paths.get("outputs", {}).items():
                Path(path_value).mkdir(parents=True, exist_ok=True)

            # Create config directories
            for path_key, path_value in paths.get("config", {}).items():
                Path(path_value).parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., "data_processing.deduplication.method")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_deduplication_config(self) -> Dict[str, Any]:
        """Get deduplication configuration"""
        return self.get("data_processing.deduplication", {})

    def get_web_scraping_config(self) -> Dict[str, Any]:
        """Get web scraping configuration"""
        return self.get("data_collection.web_scraping", {})

    def get_pdf_extraction_config(self) -> Dict[str, Any]:
        """Get PDF extraction configuration"""
        return self.get("data_collection.pdf_extraction", {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get("models", {})

    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration"""
        return self.get("error_handling", {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.get("paths", {})

    def get_prefect_config(self) -> Dict[str, Any]:
        """Get Prefect configuration"""
        return self.get("prefect", {})

    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit configuration"""
        return self.get("streamlit", {})

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get("api", {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get("database", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get("performance", {})

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get("security", {})

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation

        Args:
            key: Configuration key (e.g., "data_processing.deduplication.method")
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value
        self.logger.info(f"Configuration updated: {key} = {value}")

    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file

        Args:
            path: Path to save configuration (uses default if None)
        """
        save_path = path or self.config_path

        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def reload_config(self) -> None:
        """Reload configuration from file"""
        self.logger.info("Reloading configuration")
        self._load_config()

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        return self.get(f"environments.{self.environment}", {})

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == "testing"

    def get_log_level(self) -> str:
        """Get configured log level"""
        return self.get("pipeline.log_level", "INFO")

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get("pipeline.debug", False)

    def export_config(self, format: str = "json") -> str:
        """
        Export configuration in specified format

        Args:
            format: Export format (json, yaml)

        Returns:
            Configuration as string
        """
        if format.lower() == "json":
            return json.dumps(self.config, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(self.config, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def validate_specific_config(self, config_section: str) -> bool:
        """
        Validate specific configuration section

        Args:
            config_section: Configuration section to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            if config_section == "pipeline":
                PipelineConfig(**self.get("pipeline", {}))
            elif config_section == "web_scraping":
                WebScrapingConfig(**self.get("data_collection.web_scraping", {}))
            elif config_section == "pdf_extraction":
                PDFExtractionConfig(**self.get("data_collection.pdf_extraction", {}))
            elif config_section == "deduplication":
                DeduplicationConfig(**self.get("data_processing.deduplication", {}))
            elif config_section == "models":
                ModelConfig(**self.get("models", {}))
            elif config_section == "error_handling":
                ErrorHandlingConfig(**self.get("error_handling", {}))
            else:
                self.logger.warning(f"Unknown config section: {config_section}")
                return False

            return True

        except Exception as e:
            self.logger.error(
                f"Configuration validation failed for {config_section}: {e}"
            )
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "environment": self.environment,
            "debug": self.is_debug_enabled(),
            "log_level": self.get_log_level(),
            "deduplication_method": self.get("data_processing.deduplication.method"),
            "web_scraping_rate_limit": self.get(
                "data_collection.web_scraping.rate_limit"
            ),
            "pdf_extraction_strategy": self.get(
                "data_collection.pdf_extraction.strategy"
            ),
            "semantic_model": self.get("models.semantic.model_name"),
            "qa_generation_model": self.get("models.qa_generation.model"),
            "prefect_enabled": self.get("prefect.enabled"),
            "streamlit_enabled": self.get("streamlit.enabled"),
            "api_enabled": self.get("api.enabled"),
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager


def reload_config() -> None:
    """Reload global configuration"""
    config_manager.reload_config()
