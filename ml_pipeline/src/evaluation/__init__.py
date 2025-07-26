"""
Evaluation Module

This module provides comprehensive evaluation metrics for LLM pipeline outputs,
including ROUGE, BLEU, METEOR, and semantic similarity evaluation.
"""

from .evaluation_metrics import (
    EvaluationMetrics,
    EvaluationResult,
    quick_rouge_evaluation,
    quick_bleu_evaluation,
    quick_meteor_evaluation,
)
from .benchmark_creator import BenchmarkCreator
from .benchmark_generation import BenchmarkGenerator, BenchmarkQuestion
from .metrics_calculator import MetricsCalculator
from .performance_tester import PerformanceTester
from .comparator import Comparator
from .model_comparison import ModelEvaluator, ComparisonResult

__all__ = [
    "EvaluationMetrics",
    "EvaluationResult",
    "quick_rouge_evaluation",
    "quick_bleu_evaluation",
    "quick_meteor_evaluation",
    "BenchmarkCreator",
    "BenchmarkGenerator",
    "BenchmarkQuestion",
    "MetricsCalculator",
    "PerformanceTester",
    "Comparator",
    "ModelEvaluator",
    "ComparisonResult",
]

__version__ = "1.0.0"
__author__ = "ML Pipeline Team"
__description__ = "Comprehensive evaluation metrics for LLM pipeline outputs"
