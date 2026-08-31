"""Tokenizer Utilities for Qwen2.5.

Handles tokenizer instantiation, encoding, token count calculation,
and distribution statistics.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class QwenTokenizerWrapper:
    """Wrapper around HuggingFace AutoTokenizer for Qwen2.5."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def count_tokens(self, text: str) -> int:
        """Return token count for the given text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def batch_count_tokens(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        encodings = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        return [len(ids) for ids in encodings["input_ids"]]


def compute_token_statistics(lengths: List[int]) -> Dict[str, Any]:
    """Calculate summary statistics and percentiles for token lengths."""
    arr = np.array(lengths)
    return {
        "count": int(len(arr)),
        "min": int(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }
