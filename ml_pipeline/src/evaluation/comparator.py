import logging
from typing import Dict


class Comparator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def compare(self, baseline_metrics: Dict, finetuned_metrics: Dict) -> Dict:
        self.logger.info("Comparing baseline and fine-tuned model metrics...")
        comparison = {}
        for key in baseline_metrics:
            base = baseline_metrics.get(key, 0)
            fine = finetuned_metrics.get(key, 0)
            comparison[key] = {
                "baseline": base,
                "finetuned": fine,
                "improvement": fine - base,
            }
        self.logger.info(f"Comparison result: {comparison}")
        return comparison
