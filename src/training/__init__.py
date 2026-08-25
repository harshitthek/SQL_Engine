"""Training package for QLoRA fine-tuning, mid-training evaluation, and adapter merging."""

from .eval_metrics import compute_exact_match, normalize_sql, run_dev_evaluation
from .merge_model import compute_file_sha256, merge_lora_and_save
from .train_qlora import MidTrainingExactMatchCallback, train_qlora

__all__ = [
    "compute_exact_match",
    "normalize_sql",
    "run_dev_evaluation",
    "compute_file_sha256",
    "merge_lora_and_save",
    "MidTrainingExactMatchCallback",
    "train_qlora",
]
