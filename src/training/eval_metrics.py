"""Evaluation Metrics for Text-to-SQL.

Implements Exact Match (EM) metric calculation, Execution Accuracy (EX),
and evaluation on Spider validation examples.
"""

from typing import Any, Dict, List, Optional
import torch

from src.evaluation.evaluator import (
    EvalItemResult,
    ExecutionResult,
    SQLEvaluator,
    SandboxViolationError,
    compare_result_sets,
    compute_exact_match,
    execute_query,
    get_db_path,
    normalize_for_em,
    normalize_sql,
    sort_select_columns,
    validate_sandbox,
)


def run_dev_evaluation(
    model: Any,
    tokenizer: Any,
    dev_dataset: Any,
    max_samples: int = 500,
    device: Optional[str] = None,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Run greedy inference on dev dataset and compute Exact Match.

    Designed for mid-training evaluation.
    """
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    num_samples = min(max_samples, len(dev_dataset))
    eval_slice = dev_dataset.select(range(num_samples))

    prompts = eval_slice["prompt"]
    gold_sqls = eval_slice["sql"]

    predictions = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for prompt_len, out_seq in zip(inputs["input_ids"], outputs):
                new_tokens = out_seq[len(prompt_len):]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                predictions.append(pred_text)

    metrics = compute_exact_match(predictions, gold_sqls)
    metrics["num_evaluated"] = num_samples
    model.train()
    return metrics
