import pandas as pd
import logging
from transformers import AutoTokenizer


class Normalizer:
    def __init__(self, model_name: str = "gpt2"):
        self.logger = logging.getLogger(self.__class__.__name__)
        # Load tokenizer (authentication handled by huggingface_hub.login())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Get model's maximum sequence length
        self.max_length = getattr(self.tokenizer, "model_max_length", 1024)
        self.logger.info(f"Using max sequence length: {self.max_length}")

    def normalize(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        self.logger.info("Normalizing and tokenizing text...")
        df["normalized_text"] = df[text_column].astype(str).map(self._normalize_text)
        df["tokens"] = df["normalized_text"].map(self._tokenize_with_truncation)
        return df

    def _normalize_text(self, text: str) -> str:
        return text.strip().replace("\r", "").replace("\n", " ")

    def _tokenize_with_truncation(self, text: str) -> list:
        """Tokenize text with proper truncation to avoid sequence length errors"""
        try:
            # Use the tokenizer's built-in truncation
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            return tokens
        except Exception as e:
            self.logger.warning(
                f"Tokenization error for text (length: {len(text)}): {e}"
            )
            # Fallback: truncate text and retry
            truncated_text = text[: self.max_length * 4]  # Rough estimate
            return self.tokenizer.encode(
                truncated_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )
