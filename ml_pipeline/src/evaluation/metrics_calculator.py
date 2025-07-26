import evaluate
import logging
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class MetricsCalculator:
    def __init__(self, model_name: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.model = None
        self.tokenizer = None
        if model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        self.logger.info("Calculating ROUGE scores...")
        return self.rouge.compute(predictions=predictions, references=references)

    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        self.logger.info("Calculating BLEU scores...")
        return self.bleu.compute(
            predictions=predictions, references=[[ref] for ref in references]
        )

    def compute_perplexity(self, texts: List[str]) -> float:
        if not self.model or not self.tokenizer:
            self.logger.warning(
                "Model and tokenizer required for perplexity. Returning -1."
            )
            return -1
        self.logger.info("Calculating perplexity...")
        encodings = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        max_length = (
            self.model.config.n_positions
            if hasattr(self.model.config, "n_positions")
            else 1024
        )
        input_ids = encodings.input_ids[:, :max_length]
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def compute_f1(self, predictions: List[str], references: List[str]) -> float:
        self.logger.info("Calculating F1 score...")

        def f1(pred, ref):
            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())
            common = pred_tokens & ref_tokens
            if not common:
                return 0.0
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            return 2 * (precision * recall) / (precision + recall)

        scores = [f1(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores) if scores else 0.0

    def compute_exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        self.logger.info("Calculating Exact Match score...")
        matches = [int(p.strip() == r.strip()) for p, r in zip(predictions, references)]
        return sum(matches) / len(matches) if matches else 0.0

    def compute_custom(self, predictions: List[str], references: List[str]) -> Dict:
        self.logger.info("Calculating custom metrics...")
        return {
            "custom_metric": 0.0,
            "f1": self.compute_f1(predictions, references),
            "exact_match": self.compute_exact_match(predictions, references),
        }
