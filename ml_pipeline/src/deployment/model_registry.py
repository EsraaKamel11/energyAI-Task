import os
import json
import logging
from typing import Dict, Any


class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry.json"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.registry_path = registry_path
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def register(
        self,
        base_model: str,
        adapter_path: str,
        version: str,
        metadata: Dict[str, Any] = None,
    ):
        if base_model not in self.registry:
            self.registry[base_model] = {}
        self.registry[base_model][version] = {
            "adapter_path": adapter_path,
            "metadata": metadata or {},
        }
        self._save_registry()
        self.logger.info(f"Registered {base_model} v{version} at {adapter_path}")

    def get_adapter(self, base_model: str, version: str) -> Dict[str, Any]:
        return self.registry.get(base_model, {}).get(version, {})

    def list_versions(self, base_model: str):
        return list(self.registry.get(base_model, {}).keys())

    def _save_registry(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
