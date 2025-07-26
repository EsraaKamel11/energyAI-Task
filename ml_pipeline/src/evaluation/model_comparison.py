import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re


@dataclass
class EvaluationResult:
    """Structure for evaluation results"""

    model_name: str
    metrics: Dict[str, float]
    latency: float
    throughput: float
    total_tokens: int
    evaluation_time: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonResult:
    """Structure for model comparison results"""

    fine_tuned_model: str
    baseline_model: str
    benchmark_name: str
    fine_tuned_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    improvements: Dict[str, float]
    relative_improvements: Dict[str, float]
    statistical_significance: Dict[str, bool]
    evaluation_summary: Dict[str, Any]


class ModelEvaluator:
    """Comprehensive model evaluator with multiple metrics"""

    def __init__(self, device: str = "auto"):
        """
        Initialize model evaluator

        Args:
            device: Device to use for evaluation ("auto", "cpu", "cuda")
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bleu_smoothing = SmoothingFunction().method1

        # Load evaluation metrics
        try:
            self.rouge_metric = evaluate.load("rouge")
            self.bleu_metric = evaluate.load("bleu")
            self.meteor_metric = evaluate.load("meteor")
            self.bertscore_metric = evaluate.load("bertscore")
        except Exception as e:
            self.logger.warning(f"Some evaluation metrics not available: {e}")
            # Initialize as None if not available
            self.rouge_metric = None
            self.bleu_metric = None
            self.meteor_metric = None
            self.bertscore_metric = None

    def load_model(self, model_path: str, is_peft: bool = False) -> Tuple[Any, Any]:
        """
        Load model for evaluation

        Args:
            model_path: Path to model or model name
            is_peft: Whether the model is a PEFT model

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            if is_peft:
                # Load base model first
                base_model_name = (
                    "microsoft/DialoGPT-medium"  # Can be made configurable
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name, torch_dtype=torch.float16, device_map=self.device
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                # Load PEFT adapter
                model = PeftModel.from_pretrained(base_model, model_path)
                self.logger.info(f"Loaded PEFT model from {model_path}")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map=self.device
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.logger.info(f"Loaded model from {model_path}")

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def generate_response(
        self,
        model: Any,
        tokenizer: Any,
        question: str,
        max_length: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response from model

        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            question: Input question
            max_length: Maximum response length
            temperature: Generation temperature

        Returns:
            Generated response
        """
        try:
            # Prepare input
            inputs = tokenizer.encode(question, return_tensors="pt").to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return ""

    def calculate_rouge_scores(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            # Use rouge_scorer for more reliable results
            scores = self.rouge_scorer.score_multi(references, predictions)

            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def calculate_bleu_score(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate BLEU score"""
        try:
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                score = sentence_bleu(
                    [ref_tokens], pred_tokens, smoothing_function=self.bleu_smoothing
                )
                scores.append(score)

            return float(np.mean(scores))
        except Exception as e:
            self.logger.warning(f"BLEU calculation failed: {e}")
            return 0.0

    def calculate_exact_match(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate exact match score"""
        try:
            matches = 0
            for pred, ref in zip(predictions, references):
                if pred.strip().lower() == ref.strip().lower():
                    matches += 1

            return matches / len(predictions) if predictions else 0.0
        except Exception as e:
            self.logger.warning(f"Exact match calculation failed: {e}")
            return 0.0

    def calculate_semantic_similarity(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate semantic similarity using BERTScore"""
        try:
            if self.bertscore_metric is None:
                self.logger.warning("BERTScore metric not available")
                return 0.0

            results = self.bertscore_metric.compute(
                predictions=predictions, references=references, lang="en"
            )
            return float(np.mean(results["f1"]))
        except Exception as e:
            self.logger.warning(f"BERTScore calculation failed: {e}")
            return 0.0

    def calculate_domain_specific_metrics(
        self,
        predictions: List[str],
        references: List[str],
        domain: str = "electric_vehicles",
    ) -> Dict[str, float]:
        """Calculate domain-specific metrics"""
        metrics = {}

        if domain == "electric_vehicles":
            # EV-specific metrics
            metrics["price_accuracy"] = self._calculate_price_accuracy(
                predictions, references
            )
            metrics["technical_accuracy"] = self._calculate_technical_accuracy(
                predictions, references
            )
            metrics["compatibility_accuracy"] = self._calculate_compatibility_accuracy(
                predictions, references
            )

        return metrics

    def _calculate_price_accuracy(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate price accuracy for EV domain"""
        try:
            price_pattern = r"€?\d+\.?\d*"
            correct_prices = 0
            total_prices = 0

            for pred, ref in zip(predictions, references):
                pred_prices = re.findall(price_pattern, pred)
                ref_prices = re.findall(price_pattern, ref)

                if ref_prices:
                    total_prices += len(ref_prices)
                    for ref_price in ref_prices:
                        if ref_price in pred_prices:
                            correct_prices += 1

            return correct_prices / total_prices if total_prices > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_technical_accuracy(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate technical accuracy for EV domain"""
        try:
            technical_terms = ["kW", "kWh", "voltage", "charging", "battery", "range"]
            correct_terms = 0
            total_terms = 0

            for pred, ref in zip(predictions, references):
                for term in technical_terms:
                    if term.lower() in ref.lower():
                        total_terms += 1
                        if term.lower() in pred.lower():
                            correct_terms += 1

            return correct_terms / total_terms if total_terms > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_compatibility_accuracy(
        self, predictions: List[str], references: List[str]
    ) -> float:
        """Calculate compatibility accuracy for EV domain"""
        try:
            compatibility_terms = [
                "compatible",
                "adapter",
                "connector",
                "CCS",
                "CHAdeMO",
                "Type 2",
            ]
            correct_terms = 0
            total_terms = 0

            for pred, ref in zip(predictions, references):
                for term in compatibility_terms:
                    if term.lower() in ref.lower():
                        total_terms += 1
                        if term.lower() in pred.lower():
                            correct_terms += 1

            return correct_terms / total_terms if total_terms > 0 else 0.0
        except Exception:
            return 0.0

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        benchmark: List[Dict[str, Any]],
        model_name: str = "model",
    ) -> EvaluationResult:
        """
        Evaluate a single model

        Args:
            model: Loaded model
            tokenizer: Model tokenizer
            benchmark: List of benchmark questions
            model_name: Name of the model for logging

        Returns:
            Evaluation result
        """
        self.logger.info(f"Evaluating {model_name} on {len(benchmark)} questions")

        start_time = time.time()
        predictions = []
        total_tokens = 0

        # Generate responses
        for i, question in enumerate(benchmark):
            question_text = question["question"]

            # Generate response
            response = self.generate_response(model, tokenizer, question_text)
            predictions.append(response)

            # Count tokens
            total_tokens += len(tokenizer.encode(response))

            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(benchmark)} questions")

        evaluation_time = time.time() - start_time

        # Calculate metrics
        references = [q["answer"] for q in benchmark]

        metrics = {}

        # Standard metrics
        metrics.update(self.calculate_rouge_scores(predictions, references))
        metrics["bleu"] = self.calculate_bleu_score(predictions, references)
        metrics["exact_match"] = self.calculate_exact_match(predictions, references)
        metrics["semantic_similarity"] = self.calculate_semantic_similarity(
            predictions, references
        )

        # Domain-specific metrics
        domain_metrics = self.calculate_domain_specific_metrics(predictions, references)
        metrics.update(domain_metrics)

        # Performance metrics
        latency = evaluation_time / len(benchmark)
        throughput = len(benchmark) / evaluation_time

        return EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            latency=latency,
            throughput=throughput,
            total_tokens=total_tokens,
            evaluation_time=evaluation_time,
            metadata={
                "predictions": predictions,
                "references": references,
                "benchmark_size": len(benchmark),
            },
        )

    def compare_models(
        self,
        fine_tuned_model: Any,
        fine_tuned_tokenizer: Any,
        baseline_model: Any,
        baseline_tokenizer: Any,
        benchmark: List[Dict[str, Any]],
        fine_tuned_name: str = "fine_tuned",
        baseline_name: str = "baseline",
    ) -> ComparisonResult:
        """
        Compare fine-tuned model against baseline

        Args:
            fine_tuned_model: Fine-tuned model
            fine_tuned_tokenizer: Fine-tuned model tokenizer
            baseline_model: Baseline model
            baseline_tokenizer: Baseline model tokenizer
            benchmark: Benchmark dataset
            fine_tuned_name: Name of fine-tuned model
            baseline_name: Name of baseline model

        Returns:
            Comparison result
        """
        self.logger.info(f"Comparing {fine_tuned_name} vs {baseline_name}")

        # Evaluate both models
        fine_tuned_result = self.evaluate_model(
            fine_tuned_model, fine_tuned_tokenizer, benchmark, fine_tuned_name
        )

        baseline_result = self.evaluate_model(
            baseline_model, baseline_tokenizer, benchmark, baseline_name
        )

        # Calculate improvements
        improvements = {}
        relative_improvements = {}

        for metric in fine_tuned_result.metrics:
            ft_score = fine_tuned_result.metrics[metric]
            base_score = baseline_result.metrics[metric]

            improvement = ft_score - base_score
            improvements[metric] = improvement

            if base_score != 0:
                relative_improvement = (improvement / base_score) * 100
                relative_improvements[metric] = relative_improvement
            else:
                relative_improvements[metric] = 0.0

        # Calculate statistical significance (simple t-test)
        statistical_significance = {}
        for metric in fine_tuned_result.metrics:
            # This is a simplified significance test
            # In practice, you'd want to use proper statistical tests
            ft_score = fine_tuned_result.metrics[metric]
            base_score = baseline_result.metrics[metric]
            improvement = abs(ft_score - base_score)

            # Consider significant if improvement > 5%
            # Convert numpy.bool_ to Python bool for JSON serialization
            statistical_significance[metric] = bool(improvement > 0.05)

        # Create evaluation summary
        evaluation_summary = {
            "total_questions": len(benchmark),
            "fine_tuned_latency": fine_tuned_result.latency,
            "baseline_latency": baseline_result.latency,
            "latency_improvement": baseline_result.latency - fine_tuned_result.latency,
            "fine_tuned_throughput": fine_tuned_result.throughput,
            "baseline_throughput": baseline_result.throughput,
            "throughput_improvement": fine_tuned_result.throughput
            - baseline_result.throughput,
            "total_tokens_fine_tuned": fine_tuned_result.total_tokens,
            "total_tokens_baseline": baseline_result.total_tokens,
        }

        return ComparisonResult(
            fine_tuned_model=fine_tuned_name,
            baseline_model=baseline_name,
            benchmark_name="benchmark",
            fine_tuned_metrics=fine_tuned_result.metrics,
            baseline_metrics=baseline_result.metrics,
            improvements=improvements,
            relative_improvements=relative_improvements,
            statistical_significance=statistical_significance,
            evaluation_summary=evaluation_summary,
        )

    def save_comparison_results(
        self, result: ComparisonResult, output_path: str
    ) -> None:
        """Save comparison results to file"""
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary for JSON serialization
        result_dict = {
            "fine_tuned_model": result.fine_tuned_model,
            "baseline_model": result.baseline_model,
            "benchmark_name": result.benchmark_name,
            "fine_tuned_metrics": result.fine_tuned_metrics,
            "baseline_metrics": result.baseline_metrics,
            "improvements": result.improvements,
            "relative_improvements": result.relative_improvements,
            "statistical_significance": result.statistical_significance,
            "evaluation_summary": result.evaluation_summary,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved comparison results to {output_path}")

    def generate_comparison_report(self, result: ComparisonResult) -> str:
        """Generate a human-readable comparison report"""
        report = f"""
# Model Comparison Report

## Models Compared
- **Fine-tuned Model**: {result.fine_tuned_model}
- **Baseline Model**: {result.baseline_model}
- **Benchmark**: {result.benchmark_name}

## Performance Metrics

### Standard Metrics
"""

        for metric in result.fine_tuned_metrics:
            if metric in [
                "rouge1",
                "rouge2",
                "rougeL",
                "bleu",
                "exact_match",
                "semantic_similarity",
            ]:
                ft_score = result.fine_tuned_metrics[metric]
                base_score = result.baseline_metrics[metric]
                improvement = result.improvements[metric]
                relative_imp = result.relative_improvements[metric]
                significant = result.statistical_significance[metric]

                report += f"""
**{metric.upper()}**
- Fine-tuned: {ft_score:.4f}
- Baseline: {base_score:.4f}
- Improvement: {improvement:.4f} ({relative_imp:+.1f}%)
- Significant: {'Yes' if significant else 'No'}
"""

        report += f"""
### Domain-Specific Metrics
"""

        for metric in result.fine_tuned_metrics:
            if metric not in [
                "rouge1",
                "rouge2",
                "rougeL",
                "bleu",
                "exact_match",
                "semantic_similarity",
            ]:
                ft_score = result.fine_tuned_metrics[metric]
                base_score = result.baseline_metrics[metric]
                improvement = result.improvements[metric]
                relative_imp = result.relative_improvements[metric]
                significant = result.statistical_significance[metric]

                report += f"""
**{metric.replace('_', ' ').title()}**
- Fine-tuned: {ft_score:.4f}
- Baseline: {base_score:.4f}
- Improvement: {improvement:.4f} ({relative_imp:+.1f}%)
- Significant: {'Yes' if significant else 'No'}
"""

        report += f"""
## Performance Summary
- **Total Questions**: {result.evaluation_summary['total_questions']}
- **Fine-tuned Latency**: {result.evaluation_summary['fine_tuned_latency']:.4f}s
- **Baseline Latency**: {result.evaluation_summary['baseline_latency']:.4f}s
- **Latency Improvement**: {result.evaluation_summary['latency_improvement']:.4f}s
- **Fine-tuned Throughput**: {result.evaluation_summary['fine_tuned_throughput']:.2f} q/s
- **Baseline Throughput**: {result.evaluation_summary['baseline_throughput']:.2f} q/s
- **Throughput Improvement**: {result.evaluation_summary['throughput_improvement']:.2f} q/s

## Overall Assessment
"""

        # Calculate overall improvement
        avg_improvement = float(np.mean(list(result.improvements.values())))
        significant_metrics = sum(result.statistical_significance.values())
        total_metrics = len(result.statistical_significance)

        report += f"""
- **Average Improvement**: {avg_improvement:.4f}
- **Significant Improvements**: {significant_metrics}/{total_metrics} metrics
- **Overall Performance**: {'Improved' if avg_improvement > 0 else 'Degraded'}
"""

        return report
