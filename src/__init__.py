"""Main package exports for Text-to-SQL system.

Imports are lazy to avoid crashing in deployment environments
where ML training dependencies (peft, trl, bitsandbytes) are
not installed.
"""


def __getattr__(name: str):
    """Lazy-load subpackage symbols on first access."""
    _training_symbols = {
        "compute_exact_match",
        "compute_file_sha256",
        "merge_lora_and_save",
        "normalize_sql",
        "run_dev_evaluation",
        "train_qlora",
    }
    _inference_symbols = {
        "Text2SQLEngine",
        "get_default_model_path",
    }

    if name in _training_symbols:
        from src import training

        return getattr(training, name)
    if name in _inference_symbols:
        from src import inference

        return getattr(inference, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "compute_exact_match",
    "compute_file_sha256",
    "merge_lora_and_save",
    "normalize_sql",
    "run_dev_evaluation",
    "train_qlora",
    "Text2SQLEngine",
    "get_default_model_path",
]
