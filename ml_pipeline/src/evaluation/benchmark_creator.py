import random
import logging
from typing import List, Dict


class BenchmarkCreator:
    def __init__(self, domain: str = "general"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.domain = domain
        self.prompts = self._get_domain_prompts(domain)

    def _get_domain_prompts(self, domain: str) -> List[str]:
        # Example: extend with more domains as needed
        if domain == "finance":
            return [
                "Explain the concept of compound interest.",
                "What are the risks of stock market investing?",
                "Compare mutual funds and ETFs.",
                "Describe the process of a bank loan application.",
                "What is a credit score and how is it calculated?",
            ]
        # Default/general prompts
        return [
            "Summarize the main idea of the text.",
            "Generate a question and answer based on the passage.",
            "Provide a comparison between two concepts mentioned.",
            "Explain the reasoning behind the main argument.",
            "List key facts from the text.",
        ]

    def create_benchmark(self, n: int = 20) -> List[Dict[str, str]]:
        self.logger.info(
            f"Generating {n} benchmark test cases for domain: {self.domain}"
        )
        return [{"prompt": random.choice(self.prompts)} for _ in range(n)]
