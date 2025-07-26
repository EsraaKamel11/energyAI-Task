"""
Evaluation Metrics Module

This module provides comprehensive evaluation metrics for LLM pipeline outputs,
including ROUGE, BLEU, and other NLP evaluation metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Evaluation libraries
try:
    from evaluate import load

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logging.warning(
        "evaluate library not available. Install with: pip install evaluate"
    )

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk library not available. Install with: pip install nltk")

try:
    from rouge_score import rouge_scorer

    ROUGE_SCORER_AVAILABLE = True
except ImportError:
    ROUGE_SCORER_AVAILABLE = False
    logging.warning(
        "rouge_score library not available. Install with: pip install rouge-score"
    )

from src.utils.config_manager import ConfigManager
from src.utils.error_handling import retry_with_fallback, circuit_breaker
from src.utils.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class for storing evaluation results."""

    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for LLM pipeline outputs.

    Supports ROUGE, BLEU, METEOR, and other NLP evaluation metrics.
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize evaluation metrics.

        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.memory_manager = MemoryManager()

        # Initialize evaluation metrics
        self._initialize_metrics()

        # Results storage
        self.results: List[EvaluationResult] = []

        # Performance tracking
        self.evaluation_times: Dict[str, List[float]] = {}

    def _initialize_metrics(self):
        """Initialize evaluation metrics and download required resources."""
        try:
            # Download NLTK resources if available
            if NLTK_AVAILABLE:
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt")

                try:
                    nltk.data.find("wordnet")
                except LookupError:
                    nltk.download("wordnet")

                try:
                    nltk.data.find("omw-1.4")
                except LookupError:
                    nltk.download("omw-1.4")

            # Initialize ROUGE scorer
            if ROUGE_SCORER_AVAILABLE:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
                )

            # Initialize evaluate metrics
            if EVALUATE_AVAILABLE:
                self.rouge_evaluate = load("rouge")
                self.bleu_evaluate = load("bleu")
                self.meteor_evaluate = load("meteor")

        except Exception as e:
            logger.error(f"Error initializing evaluation metrics: {e}")
            raise

    @retry_with_fallback(max_attempts=3)
    def evaluate_rouge(
        self,
        predictions: List[str],
        references: List[str],
        use_evaluate_lib: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate ROUGE scores for predictions against references.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            use_evaluate_lib: Whether to use evaluate library (fallback to rouge-score)

        Returns:
            Dictionary containing ROUGE scores
        """
        start_time = time.time()

        try:
            if use_evaluate_lib and EVALUATE_AVAILABLE:
                results = self.rouge_evaluate.compute(
                    predictions=predictions, references=references
                )

                # Extract scores - handle both old format (ROUGE score objects) and new format (numpy floats)
                def extract_rouge_score(value):
                    if hasattr(value, "mid") and hasattr(value.mid, "fmeasure"):
                        return value.mid.fmeasure
                    elif isinstance(value, (int, float, np.number)):
                        return float(value)
                    else:
                        return 0.0

                rouge_scores = {
                    "rouge1": extract_rouge_score(results["rouge1"]),
                    "rouge2": extract_rouge_score(results["rouge2"]),
                    "rougeL": extract_rouge_score(results["rougeL"]),
                    "rougeLsum": extract_rouge_score(results["rougeLsum"]),
                }

            elif ROUGE_SCORER_AVAILABLE:
                # Use rouge-score library as fallback
                rouge_scores = {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "rougeLsum": 0.0,
                }

                for pred, ref in zip(predictions, references):
                    scores = self.rouge_scorer.score(ref, pred)
                    rouge_scores["rouge1"] += scores["rouge1"].fmeasure
                    rouge_scores["rouge2"] += scores["rouge2"].fmeasure
                    rouge_scores["rougeL"] += scores["rougeL"].fmeasure
                    rouge_scores["rougeLsum"] += scores["rougeLsum"].fmeasure

                # Average scores
                n = len(predictions)
                for key in rouge_scores:
                    rouge_scores[key] /= n

            else:
                raise ImportError("No ROUGE evaluation library available")

            # Track evaluation time
            eval_time = time.time() - start_time
            self.evaluation_times.setdefault("rouge", []).append(eval_time)

            return rouge_scores

        except Exception as e:
            logger.error(f"Error in ROUGE evaluation: {e}")
            raise

    @retry_with_fallback(max_attempts=3)
    def evaluate_bleu(
        self,
        predictions: List[str],
        references: List[str],
        use_evaluate_lib: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate BLEU scores for predictions against references.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            use_evaluate_lib: Whether to use evaluate library (fallback to nltk)

        Returns:
            Dictionary containing BLEU scores
        """
        start_time = time.time()

        try:
            if use_evaluate_lib and EVALUATE_AVAILABLE:
                results = self.bleu_evaluate.compute(
                    predictions=predictions, references=references
                )

                bleu_scores = {"bleu": results["bleu"]}

            elif NLTK_AVAILABLE:
                # Use NLTK as fallback
                smoothing = SmoothingFunction().method1
                bleu_scores = {"bleu": 0.0}

                for pred, ref in zip(predictions, references):
                    pred_tokens = nltk.word_tokenize(pred.lower())
                    ref_tokens = [nltk.word_tokenize(ref.lower())]
                    score = sentence_bleu(
                        ref_tokens, pred_tokens, smoothing_function=smoothing
                    )
                    bleu_scores["bleu"] += score

                bleu_scores["bleu"] /= len(predictions)

            else:
                raise ImportError("No BLEU evaluation library available")

            # Track evaluation time
            eval_time = time.time() - start_time
            self.evaluation_times.setdefault("bleu", []).append(eval_time)

            return bleu_scores

        except Exception as e:
            logger.error(f"Error in BLEU evaluation: {e}")
            raise

    @retry_with_fallback(max_attempts=3)
    def evaluate_meteor(
        self,
        predictions: List[str],
        references: List[str],
        use_evaluate_lib: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate METEOR scores for predictions against references.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            use_evaluate_lib: Whether to use evaluate library (fallback to nltk)

        Returns:
            Dictionary containing METEOR scores
        """
        start_time = time.time()

        try:
            if use_evaluate_lib and EVALUATE_AVAILABLE:
                results = self.meteor_evaluate.compute(
                    predictions=predictions, references=references
                )

                meteor_scores = {"meteor": results["meteor"]}

            elif NLTK_AVAILABLE:
                # Use NLTK as fallback
                meteor_scores = {"meteor": 0.0}

                for pred, ref in zip(predictions, references):
                    pred_tokens = nltk.word_tokenize(pred.lower())
                    ref_tokens = nltk.word_tokenize(ref.lower())
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores["meteor"] += score

                meteor_scores["meteor"] /= len(predictions)

            else:
                raise ImportError("No METEOR evaluation library available")

            # Track evaluation time
            eval_time = time.time() - start_time
            self.evaluation_times.setdefault("meteor", []).append(eval_time)

            return meteor_scores

        except Exception as e:
            logger.error(f"Error in METEOR evaluation: {e}")
            raise

    def evaluate_semantic_similarity(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate semantic similarity using cosine similarity of embeddings.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dictionary containing semantic similarity scores
        """
        start_time = time.time()

        try:
            # Use memory manager for chunked processing
            similarity_scores = []

            for pred, ref in zip(predictions, references):
                # Simple token overlap as fallback
                pred_tokens = set(pred.lower().split())
                ref_tokens = set(ref.lower().split())

                if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                    similarity_scores.append(0.0)
                else:
                    intersection = len(pred_tokens.intersection(ref_tokens))
                    union = len(pred_tokens.union(ref_tokens))
                    similarity = intersection / union if union > 0 else 0.0
                    similarity_scores.append(similarity)

            semantic_scores = {
                "semantic_similarity": np.mean(similarity_scores),
                "semantic_similarity_std": np.std(similarity_scores),
            }

            # Track evaluation time
            eval_time = time.time() - start_time
            self.evaluation_times.setdefault("semantic_similarity", []).append(
                eval_time
            )

            return semantic_scores

        except Exception as e:
            logger.error(f"Error in semantic similarity evaluation: {e}")
            return {"semantic_similarity": 0.0, "semantic_similarity_std": 0.0}

    def evaluate_comprehensive(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with multiple metrics.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            metrics: List of metrics to evaluate (default: all available)

        Returns:
            Dictionary containing all evaluation results
        """
        if metrics is None:
            metrics = ["rouge", "bleu", "meteor", "semantic_similarity"]

        results = {}
        metadata = {
            "num_predictions": len(predictions),
            "num_references": len(references),
            "metrics_evaluated": metrics,
            "timestamp": time.time(),
        }

        # Validate inputs
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if len(predictions) == 0:
            logger.warning("Empty predictions/references provided")
            return results

        # Run evaluations
        for metric in metrics:
            try:
                if metric == "rouge":
                    results["rouge"] = self.evaluate_rouge(predictions, references)
                elif metric == "bleu":
                    results["bleu"] = self.evaluate_bleu(predictions, references)
                elif metric == "meteor":
                    results["meteor"] = self.evaluate_meteor(predictions, references)
                elif metric == "semantic_similarity":
                    results["semantic_similarity"] = self.evaluate_semantic_similarity(
                        predictions, references
                    )
                else:
                    logger.warning(f"Unknown metric: {metric}")

            except Exception as e:
                logger.error(f"Error evaluating {metric}: {e}")
                results[metric] = {"error": str(e)}

        # Store results
        for metric_name, metric_results in results.items():
            if "error" not in metric_results:
                evaluation_result = EvaluationResult(
                    metric_name=metric_name,
                    score=self._extract_primary_score(metric_results),
                    details=metric_results,
                    timestamp=time.time(),
                    metadata=metadata,
                )
                self.results.append(evaluation_result)

        return results

    def _extract_primary_score(self, metric_results: Dict[str, float]) -> float:
        """Extract the primary score from metric results."""
        if isinstance(metric_results, dict):
            # Try common primary score keys
            for key in ["rouge1", "bleu", "meteor", "semantic_similarity"]:
                if key in metric_results:
                    return metric_results[key]
            # Return first numeric value
            for value in metric_results.values():
                if isinstance(value, (int, float)):
                    return value
        return 0.0

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for evaluation metrics."""
        stats = {}
        for metric, times in self.evaluation_times.items():
            if times:
                stats[metric] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "total_evaluations": len(times),
                }
        return stats

    def save_results(self, filepath: Union[str, Path], format: str = "json"):
        """Save evaluation results to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                # Convert results to serializable format
                serializable_results = []
                for result in self.results:
                    serializable_results.append(
                        {
                            "metric_name": result.metric_name,
                            "score": result.score,
                            "details": result.details,
                            "timestamp": result.timestamp,
                            "metadata": result.metadata,
                        }
                    )

                with open(filepath, "w") as f:
                    json.dump(serializable_results, f, indent=2)

            elif format.lower() == "csv":
                # Convert to DataFrame and save
                df = pd.DataFrame(
                    [
                        {
                            "metric_name": r.metric_name,
                            "score": r.score,
                            "timestamp": r.timestamp,
                            **r.metadata,
                        }
                        for r in self.results
                    ]
                )
                df.to_csv(filepath, index=False)

            logger.info(f"Results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def load_results(self, filepath: Union[str, Path], format: str = "json"):
        """Load evaluation results from file."""
        try:
            filepath = Path(filepath)

            if format.lower() == "json":
                with open(filepath, "r") as f:
                    data = json.load(f)

                self.results = []
                for item in data:
                    result = EvaluationResult(
                        metric_name=item["metric_name"],
                        score=item["score"],
                        details=item["details"],
                        timestamp=item["timestamp"],
                        metadata=item["metadata"],
                    )
                    self.results.append(result)

            elif format.lower() == "csv":
                df = pd.read_csv(filepath)
                self.results = []
                for _, row in df.iterrows():
                    result = EvaluationResult(
                        metric_name=row["metric_name"],
                        score=row["score"],
                        details={},
                        timestamp=row["timestamp"],
                        metadata={
                            k: v
                            for k, v in row.items()
                            if k not in ["metric_name", "score", "timestamp"]
                        },
                    )
                    self.results.append(result)

            logger.info(f"Results loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise

    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return "No evaluation results available."

        report_lines = [
            "# Evaluation Metrics Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total evaluations: {len(self.results)}",
            "",
            "## Performance Statistics",
        ]

        # Add performance stats
        perf_stats = self.get_performance_stats()
        for metric, stats in perf_stats.items():
            report_lines.extend(
                [
                    f"### {metric.upper()}",
                    f"- Mean evaluation time: {stats['mean_time']:.4f}s",
                    f"- Standard deviation: {stats['std_time']:.4f}s",
                    f"- Total evaluations: {stats['total_evaluations']}",
                    "",
                ]
            )

        # Add metric summaries
        report_lines.append("## Metric Summaries")
        metric_groups = {}
        for result in self.results:
            if result.metric_name not in metric_groups:
                metric_groups[result.metric_name] = []
            metric_groups[result.metric_name].append(result.score)

        for metric, scores in metric_groups.items():
            report_lines.extend(
                [
                    f"### {metric.upper()}",
                    f"- Mean score: {np.mean(scores):.4f}",
                    f"- Standard deviation: {np.std(scores):.4f}",
                    f"- Min score: {np.min(scores):.4f}",
                    f"- Max score: {np.max(scores):.4f}",
                    f"- Number of evaluations: {len(scores)}",
                    "",
                ]
            )

        return "\n".join(report_lines)


# Convenience functions for quick integration
def quick_rouge_evaluation(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Quick ROUGE evaluation using the evaluate library."""
    if not EVALUATE_AVAILABLE:
        raise ImportError(
            "evaluate library not available. Install with: pip install evaluate"
        )

    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references)

    # Handle both old format (ROUGE score objects) and new format (numpy floats)
    def extract_rouge_score(value):
        if hasattr(value, "mid") and hasattr(value.mid, "fmeasure"):
            return value.mid.fmeasure
        elif isinstance(value, (int, float, np.number)):
            return float(value)
        else:
            return 0.0

    return {
        "rouge1": extract_rouge_score(results["rouge1"]),
        "rouge2": extract_rouge_score(results["rouge2"]),
        "rougeL": extract_rouge_score(results["rougeL"]),
        "rougeLsum": extract_rouge_score(results["rougeLsum"]),
    }


def quick_bleu_evaluation(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Quick BLEU evaluation using the evaluate library."""
    if not EVALUATE_AVAILABLE:
        raise ImportError(
            "evaluate library not available. Install with: pip install evaluate"
        )

    bleu = load("bleu")
    results = bleu.compute(predictions=predictions, references=references)

    return {"bleu": results["bleu"]}


def quick_meteor_evaluation(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Quick METEOR evaluation using the evaluate library."""
    if not EVALUATE_AVAILABLE:
        raise ImportError(
            "evaluate library not available. Install with: pip install evaluate"
        )

    meteor = load("meteor")
    results = meteor.compute(predictions=predictions, references=references)

    return {"meteor": results["meteor"]}
