import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseCollector(ABC):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def collect(self, *args, **kwargs) -> None:
        """Collect data and save it with metadata."""
        pass

    def save_data(self, data: Any, metadata: Dict[str, Any], filename: str) -> None:
        import json

        raw_path = os.path.join(self.output_dir, filename)
        try:
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"data": data, "metadata": metadata},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.logger.info(f"Saved data to {raw_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")

    def load_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        import json

        meta_path = os.path.join(self.output_dir, filename)
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            return content.get("metadata", {})
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

    def resume(self, *args, **kwargs) -> None:
        """Resume collection from last saved state."""
        self.logger.info("Resume functionality not implemented.")
