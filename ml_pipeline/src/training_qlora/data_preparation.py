#!/usr/bin/env python3
"""
QLoRA Data Preparation
Handles data loading, preprocessing, and tokenization for QLoRA training
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class QLoRADataPreparer:
    """Handles data preparation for QLoRA training"""

    def __init__(self, max_length: int = 1024):
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

    def load_qa_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load QA data from JSONL file

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List[Dict]: List of QA pairs
        """
        self.logger.info(f"Loading QA data from {jsonl_path}")

        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))

        self.logger.info(f"Loaded {len(data)} QA pairs")
        return data

    def create_conversation_format(
        self, qa_pairs: List[Dict[str, Any]], system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Convert QA pairs to conversation format for training

        Args:
            qa_pairs: List of QA pairs
            system_prompt: System prompt to prepend

        Returns:
            List[Dict]: List of conversations
        """
        if system_prompt is None:
            system_prompt = (
                "EV Assistant: I'm here to help you with electric vehicle charging information. "
                "I can provide guidance on charging stations, connectors, and best practices.\n\n"
            )

        conversations = []
        for qa in qa_pairs:
            conversation = (
                f"{system_prompt}"
                f"User: {qa['question']}\n\n"
                f"EV Assistant: {qa['answer']}"
            )
            conversations.append({"text": conversation})

        self.logger.info(f"Created {len(conversations)} conversations")
        return conversations

    def create_dataset(
        self,
        conversations: List[Dict[str, str]],
        test_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[Dataset, Dataset]:
        """
        Create train/validation datasets

        Args:
            conversations: List of conversations
            test_size: Fraction of data for validation
            seed: Random seed

        Returns:
            Tuple[Dataset, Dataset]: Train and validation datasets
        """
        dataset = Dataset.from_list(conversations)
        train_test = dataset.train_test_split(test_size=test_size, seed=seed)

        train_dataset = train_test["train"]
        val_dataset = train_test["test"]

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def tokenize_dataset(
        self, dataset: Dataset, tokenizer, max_length: int = None
    ) -> Dataset:
        """
        Tokenize dataset for training

        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Dataset: Tokenized dataset
        """
        if max_length is None:
            max_length = self.max_length

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        self.logger.info(f"Tokenized dataset with max_length={max_length}")
        return tokenized_dataset

    def prepare_training_data(
        self,
        jsonl_path: str,
        tokenizer,
        system_prompt: str = None,
        test_size: float = 0.1,
        max_length: int = None,
    ) -> Tuple[Dataset, Dataset]:
        """
        Complete data preparation pipeline

        Args:
            jsonl_path: Path to JSONL file
            tokenizer: Tokenizer to use
            system_prompt: System prompt
            test_size: Validation split size
            max_length: Maximum sequence length

        Returns:
            Tuple[Dataset, Dataset]: Tokenized train and validation datasets
        """
        # Load data
        qa_pairs = self.load_qa_data(jsonl_path)

        # Create conversations
        conversations = self.create_conversation_format(qa_pairs, system_prompt)

        # Create datasets
        train_dataset, val_dataset = self.create_dataset(conversations, test_size)

        # Tokenize datasets
        train_tokenized = self.tokenize_dataset(train_dataset, tokenizer, max_length)
        val_tokenized = self.tokenize_dataset(val_dataset, tokenizer, max_length)

        return train_tokenized, val_tokenized

    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get statistics about the dataset

        Args:
            dataset: Dataset to analyze

        Returns:
            Dict: Dataset statistics
        """
        # Check if dataset has 'text' column (before tokenization) or 'input_ids' (after tokenization)
        if "text" in dataset.column_names:
            # Before tokenization - get text lengths
            text_lengths = [len(item["text"]) for item in dataset]
            return {
                "num_samples": len(dataset),
                "avg_text_length": (
                    sum(text_lengths) / len(text_lengths) if text_lengths else 0
                ),
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "total_chars": sum(text_lengths),
            }
        elif "input_ids" in dataset.column_names:
            # After tokenization - get token lengths
            token_lengths = [len(item["input_ids"]) for item in dataset]
            return {
                "num_samples": len(dataset),
                "avg_token_length": (
                    sum(token_lengths) / len(token_lengths) if token_lengths else 0
                ),
                "min_token_length": min(token_lengths) if token_lengths else 0,
                "max_token_length": max(token_lengths) if token_lengths else 0,
                "total_tokens": sum(token_lengths),
            }
        else:
            # Fallback for unknown dataset format
            return {
                "num_samples": len(dataset),
                "columns": dataset.column_names,
                "note": "Dataset format not recognized",
            }

    def save_processed_data(
        self, train_dataset: Dataset, val_dataset: Dataset, output_dir: str
    ) -> None:
        """
        Save processed datasets

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSONL
        train_dataset.to_json(output_path / "train.jsonl")
        val_dataset.to_json(output_path / "validation.jsonl")

        # Save as Parquet
        train_dataset.to_parquet(output_path / "train.parquet")
        val_dataset.to_parquet(output_path / "validation.parquet")

        self.logger.info(f"Saved processed data to {output_path}")


def prepare_qlora_data(
    jsonl_path: str,
    tokenizer,
    system_prompt: str = None,
    test_size: float = 0.1,
    max_length: int = 1024,
) -> Tuple[Dataset, Dataset]:
    """
    Convenience function to prepare data for QLoRA training

    Args:
        jsonl_path: Path to JSONL file
        tokenizer: Tokenizer to use
        system_prompt: System prompt
        test_size: Validation split size
        max_length: Maximum sequence length

    Returns:
        Tuple[Dataset, Dataset]: Tokenized train and validation datasets
    """
    preparer = QLoRADataPreparer(max_length)
    return preparer.prepare_training_data(
        jsonl_path, tokenizer, system_prompt, test_size, max_length
    )


if __name__ == "__main__":
    # Test the data preparer
    logging.basicConfig(level=logging.INFO)

    try:
        preparer = QLoRADataPreparer()

        # Test with sample data
        sample_qa = [
            {
                "question": "What is Level 2 charging?",
                "answer": "Level 2 charging uses 240V power.",
            },
            {
                "question": "How fast is DC charging?",
                "answer": "DC charging can provide 60-80% charge in 20-30 minutes.",
            },
        ]

        # Test conversation format
        conversations = preparer.create_conversation_format(sample_qa)
        print("✅ Data preparer test successful!")
        print(f"Created {len(conversations)} conversations")
        print(f"Sample conversation: {conversations[0]['text'][:100]}...")

    except Exception as e:
        print(f"❌ Data preparer test failed: {e}")
