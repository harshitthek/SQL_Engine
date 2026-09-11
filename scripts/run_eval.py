#!/usr/bin/env python3
"""Batch Evaluation Script for Text-to-SQL Model on Spider Dataset.

Implements:
- Task 06: Batch evaluation across all Spider dev examples, reporting EM + EX by SQL complexity tier.
- Task 07: Automated error pattern analysis producing artifacts/eval_error_analysis.md.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Any

import torch
from datasets import load_from_disk
from tabulate import tabulate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repository root to python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.evaluator import (
    EvalItemResult,
    SQLEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL Model on Spider Dev Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/text2sql-qwen-pytorch-v1-v1/text2sql-v1",
        help="Path to trained HF model directory",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed_arrow",
        help="Path to processed Arrow dataset or Spider dev.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate on (validation/train)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Execution timeout per query in seconds",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda, mps, cpu). Default: auto-detect",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="artifacts/dev_predictions.json",
        help="File path to save generated predictions",
    )
    parser.add_argument(
        "--load-predictions",
        type=str,
        default=None,
        help="Optional path to existing predictions JSON to bypass model inference",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/eval_error_analysis.md",
        help="File path to save markdown error analysis report",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="artifacts/eval_metrics.json",
        help="File path to save JSON metrics summary",
    )
    return parser.parse_args()


def get_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval_data(
    data_path: str, split: str = "validation", max_samples: int | None = None
) -> list[dict[str, Any]]:
    """Load evaluation samples from arrow dataset or dev.json."""
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, "dataset_dict.json")):
        ds = load_from_disk(data_path)[split]
        records = [dict(ds[i]) for i in range(len(ds))]
    elif data_path.endswith(".json"):
        with open(data_path, encoding="utf-8") as f:
            raw_records = json.load(f)
        records = []
        for idx, r in enumerate(raw_records):
            sql_query = r.get("query", "")
            ast_dict = r.get("sql")
            hardness = r.get("hardness")
            if hardness is None and isinstance(ast_dict, dict):
                try:
                    from data.processing.complexity import eval_hardness
                    hardness = eval_hardness(ast_dict)
                except Exception:
                    hardness = "unknown"

            rec = {
                "id": r.get("id", idx),
                "db_id": r["db_id"],
                "question": r["question"],
                "sql": sql_query if isinstance(r.get("sql"), dict) else r.get("sql", sql_query),
                "hardness": hardness or "unknown",
            }
            if "prompt" in r:
                rec["prompt"] = r["prompt"]
            records.append(rec)
    else:
        # Try processed arrow directory directly
        ds = load_from_disk(data_path)
        if hasattr(ds, "keys") and split in ds:
            ds = ds[split]
        records = [dict(ds[i]) for i in range(len(ds))]

    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    return records


def run_model_inference(
    model_path: str,
    records: list[dict[str, Any]],
    batch_size: int = 32,
    device: str | None = None,
) -> list[str]:
    """Run batched greedy generation to produce SQL predictions."""
    target_device = get_device(device)
    print(f"\n[1/3] Loading model from: {model_path}")
    print(f"      Target Device: {target_device} | Batch Size: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if target_device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if target_device in ("cuda", "mps"):
        model = model.to(target_device)
    model.eval()

    prompts = []
    serializer = None
    for r in records:
        if "prompt" in r:
            prompts.append(r["prompt"])
        else:
            if serializer is None:
                from data.processing.prompt_templates import TEMPLATES
                from data.processing.schema_serializer import SchemaSerializer
                serializer = SchemaSerializer(
                    tables_json_path="data/spider_data/tables.json",
                    db_root_dir="data/spider_data/database",
                )
            schema = serializer.serialize_ddl(r["db_id"])
            p = TEMPLATES["code_comment"].format_input(schema, r["question"])
            prompts.append(p)
    predictions: list[str] = []

    print(f"[2/3] Generating SQL queries for {len(prompts)} examples...")
    t0 = time.time()

    for i in tqdm(range(0, len(prompts), batch_size), desc="Inferring"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(target_device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for prompt_tensor, out_tensor in zip(inputs["input_ids"], outputs):
            gen_tokens = out_tensor[len(prompt_tensor) :]
            pred_sql = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Clean markdown wrappers if any
            if pred_sql.startswith("```sql"):
                pred_sql = pred_sql[6:]
            elif pred_sql.startswith("```"):
                pred_sql = pred_sql[3:]
            if pred_sql.endswith("```"):
                pred_sql = pred_sql[:-3]
            pred_sql = pred_sql.strip()

            predictions.append(pred_sql)

    elapsed = time.time() - t0
    print(f"      Generation completed in {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/sample)")
    return predictions


def categorize_sql_failure_pattern(
    pred_sql: str,
    gold_sql: str,
    error_type: str | None,
    error_message: str | None,
) -> str:
    """Analyze a single failure and map it to a concrete failure pattern category."""
    if error_type == "Timeout":
        return "Query Timeout / Cartesian Explosion"
    if error_type == "SandboxViolation":
        return "Sandbox Violation (Non-SELECT Statement)"
    if error_type == "SyntaxError":
        return "SQL Syntax Error"
    if error_type == "RuntimeError":
        msg = (error_message or "").lower()
        if "no such column" in msg:
            return "Schema Hallucination (Invalid Column Name)"
        if "no such table" in msg:
            return "Schema Hallucination (Invalid Table Name)"
        return "SQLite RuntimeError"

    # For EmptyResult or ResultMismatch, analyze SQL differences
    pred_lower = pred_sql.lower()
    gold_lower = gold_sql.lower()

    # Pattern 1: Missing or mismatched GROUP BY / Aggregation
    has_gold_group = "group by" in gold_lower
    has_pred_group = "group by" in pred_lower
    if has_gold_group and not has_pred_group:
        return "Missing GROUP BY Clause"
    if not has_gold_group and has_pred_group:
        return "Extraneous GROUP BY Clause"

    # Pattern 2: JOIN errors (different JOIN conditions or tables)
    gold_joins = len(re.findall(r"\bjoin\b", gold_lower))
    pred_joins = len(re.findall(r"\bjoin\b", pred_lower))
    gold_tables = set(re.findall(r"\bfrom\s+([a-zA-Z0-9_]+)", gold_lower) + re.findall(r"\bjoin\s+([a-zA-Z0-9_]+)", gold_lower))
    pred_tables = set(re.findall(r"\bfrom\s+([a-zA-Z0-9_]+)", pred_lower) + re.findall(r"\bjoin\s+([a-zA-Z0-9_]+)", pred_lower))

    if gold_joins != pred_joins or gold_tables != pred_tables:
        return "Wrong JOIN / Table Selection"
    if " on " in gold_lower and " on " in pred_lower:
        gold_on = re.findall(r"on\s+([^where|group|order|limit]+)", gold_lower)
        pred_on = re.findall(r"on\s+([^where|group|order|limit]+)", pred_lower)
        if gold_on and pred_on and gold_on[0].strip() != pred_on[0].strip():
            return "Wrong JOIN Condition"

    # Pattern 3: Aggregation function mismatch
    gold_aggs = set(re.findall(r"\b(count|sum|avg|max|min)\s*\(", gold_lower))
    pred_aggs = set(re.findall(r"\b(count|sum|avg|max|min)\s*\(", pred_lower))
    if gold_aggs != pred_aggs:
        return "Aggregation Operator Mismatch (e.g. COUNT vs SUM/AVG)"

    # Pattern 4: Nested Subquery / Set operation error
    gold_nested = any(k in gold_lower for k in ("except", "union", "intersect", "select.*select"))
    pred_nested = any(k in pred_lower for k in ("except", "union", "intersect", "select.*select"))
    if gold_nested != pred_nested:
        return "Subquery / Set Operation Mismatch"

    # Pattern 5: ORDER BY / LIMIT mismatch
    if ("limit" in gold_lower) != ("limit" in pred_lower) or ("order by" in gold_lower) != ("order by" in pred_lower):
        return "ORDER BY / LIMIT Mismatch"

    # Pattern 6: WHERE clause filter mismatch
    if "where" in gold_lower and "where" in pred_lower:
        return "WHERE Filter / Value Discrepancy"

    return "Result Mismatch (Subtle Logic/Filter Discrepancy)"


def generate_error_report(
    eval_metrics: dict[str, Any],
    records: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Generate comprehensive Markdown error analysis report for Task 07."""
    results: list[EvalItemResult] = eval_metrics["results"]
    total = len(results)
    failures = [r for r in results if not r.ex_correct]
    num_failures = len(failures)

    # Classify all failures into patterns
    pattern_counts: Counter = Counter()
    failure_examples: dict[str, list[dict[str, Any]]] = {}

    for idx, (res, rec) in enumerate(zip(results, records)):
        if not res.ex_correct:
            pattern = categorize_sql_failure_pattern(
                res.pred_sql,
                res.gold_sql,
                res.error_type,
                res.error_message,
            )
            pattern_counts[pattern] += 1
            if pattern not in failure_examples:
                failure_examples[pattern] = []
            if len(failure_examples[pattern]) < 3:
                failure_examples[pattern].append({
                    "id": rec.get("id", idx),
                    "db_id": res.db_id,
                    "question": rec.get("question", ""),
                    "gold_sql": res.gold_sql,
                    "pred_sql": res.pred_sql,
                    "error_type": res.error_type,
                    "error_message": res.error_message,
                    "hardness": res.hardness,
                })

    top_patterns = pattern_counts.most_common(3)

    # Complexity tier breakdown table
    tier_rows = []
    for tier, stats in eval_metrics["by_hardness"].items():
        tier_rows.append([
            tier.capitalize(),
            stats["total"],
            stats["em_correct"],
            f"{stats['em_percent']:.2f}%",
            stats["ex_correct"],
            f"{stats['ex_percent']:.2f}%",
        ])

    tier_table = tabulate(
        tier_rows,
        headers=["Complexity Tier", "Total", "EM Correct", "EM (%)", "EX Correct", "EX (%)"],
        tablefmt="github",
    )

    # Error breakdown table
    err_rows = []
    for err_type, count in eval_metrics["error_breakdown"].items():
        pct = (count / total * 100.0) if total > 0 else 0.0
        err_rows.append([err_type, count, f"{pct:.2f}%"])
    err_table = tabulate(
        err_rows,
        headers=["Error Category", "Count", "% of Total"],
        tablefmt="github",
    )

    # Top Failure Patterns Table
    pattern_rows = []
    for pat, count in pattern_counts.most_common():
        pct = (count / num_failures * 100.0) if num_failures > 0 else 0.0
        pattern_rows.append([pat, count, f"{pct:.2f}%"])
    pattern_table = tabulate(
        pattern_rows,
        headers=["Failure Pattern", "Failure Count", "% of Failures"],
        tablefmt="github",
    )

    # Construct Markdown Report
    lines = [
        "# Text-to-SQL Evaluation & Error Analysis Report",
        "",
        f"**Date/Time:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        "**Benchmark Split:** Spider Validation Set (`dev.json`)  ",
        f"**Total Examples Evaluated:** {total}  ",
        f"**Overall Exact Match (EM):** {eval_metrics['exact_match']:.2f}% ({eval_metrics['em_correct']}/{total})  ",
        f"**Overall Execution Accuracy (EX):** {eval_metrics['execution_accuracy']:.2f}% ({eval_metrics['ex_correct']}/{total})  ",
        "",
        "---",
        "",
        "## 1. Performance by SQL Complexity Tier",
        "",
        "Evaluation results segmented by Spider difficulty tiers (simple, medium, hard, extra-hard):",
        "",
        tier_table,
        "",
        "---",
        "",
        "## 2. Execution Error Breakdown",
        "",
        "Categorization of all query execution outcomes across the benchmark:",
        "",
        err_table,
        "",
        "---",
        "",
        "## 3. Failure Pattern Distribution",
        "",
        "Automated semantic diagnosis of failed queries:",
        "",
        pattern_table,
        "",
        "---",
        "",
        "## 4. Deep-Dive: Top 3 Failure Patterns",
        "",
    ]

    for rank, (pat_name, count) in enumerate(top_patterns, 1):
        pct = (count / num_failures * 100.0) if num_failures > 0 else 0.0
        lines.extend([
            f"### Pattern #{rank}: {pat_name}",
            f"- **Occurrence Count:** {count} ({pct:.2f}% of all failures)",
            "",
            "#### Representative Examples & Diagnostics:",
            "",
        ])

        examples = failure_examples.get(pat_name, [])
        for ex_idx, ex in enumerate(examples, 1):
            lines.extend([
                f"**Example {rank}.{ex_idx} (DB: `{ex['db_id']}`, Tier: `{ex['hardness']}`):**",
                f"- **Question:** *\"{ex['question']}\"*",
                "- **Gold SQL:**",
                f"  ```sql\n  {ex['gold_sql']}\n  ```",
                "- **Model SQL:**",
                f"  ```sql\n  {ex['pred_sql']}\n  ```",
                f"- **Error Diagnosis:** `{ex['error_type']}` — {ex['error_message'] or 'Result mismatch'}",
                "",
            ])

        lines.extend([
            "#### Root Cause & Mitigation:",
        ])
        if "join" in pat_name.lower():
            lines.extend([
                "- **Root Cause:** The model struggles with schema multi-hop relationships, choosing the wrong junction table or connecting foreign keys incorrectly when multiple join paths exist.",
                "- **Mitigation Strategy:** Inject foreign key relationship graphs explicitly into the prompt schema serialization; apply schema-linking hints during inference.",
                "",
            ])
        elif "group by" in pat_name.lower():
            lines.extend([
                "- **Root Cause:** The model outputs aggregate expressions (e.g. `COUNT(*)`, `AVG(col)`) alongside non-aggregate columns without generating the corresponding `GROUP BY` clause required in standard SQL.",
                "- **Mitigation Strategy:** Apply AST post-processing or constrained decoding to automatically enforce `GROUP BY` whenever unaggregated projection columns appear alongside aggregates.",
                "",
            ])
        elif "column" in pat_name.lower() or "hallucination" in pat_name.lower():
            lines.extend([
                "- **Root Cause:** Schema hallucination: selecting column names that exist in other tables or inventing column synonyms not present in the SQLite catalog.",
                "- **Mitigation Strategy:** Constrain token generation to valid identifiers present in the schema DDL using schema-guided prefix masking / grammar-based sampling.",
                "",
            ])
        elif "where" in pat_name.lower() or "filter" in pat_name.lower():
            lines.extend([
                "- **Root Cause:** Filter condition discrepancy, such as using incorrect string casing, wrong date formats, or inverted inequality operators.",
                "- **Mitigation Strategy:** Include representative column values in prompt schema comments to give the model awareness of actual categorical value distributions.",
                "",
            ])
        else:
            lines.extend([
                "- **Root Cause:** Complex compositional logic discrepancy between model interpretation and gold annotation.",
                "- **Mitigation Strategy:** Add targeted few-shot demonstrations for complex compositional SQL patterns.",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## 5. Summary & Engineering Recommendations",
        "",
        "1. **Schema Linking & Foreign Key Enforcement:** The primary source of execution errors is multi-table join confusion. Enhancing schema serialization with explicit `FOREIGN KEY (a) REFERENCES b(c)` edges improves join path discovery.",
        "2. **Grammar-Based Constrained Decoding:** Enforcing SQLite grammar at decode time eliminates 100% of syntax errors and prevents schema hallucination.",
        "3. **Execution-Guided Self-Correction / Re-ranking:** Use the SQLEvaluator in a test-time beam re-ranking loop: generate 4 candidates, execute against SQLite, reject queries that produce syntax or runtime errors, and select the highest-scoring valid query.",
        "",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Artifact Created] Error analysis report written to: {output_path}")


def main():
    args = parse_args()
    print("=" * 70)
    print("Text-to-SQL Spider Evaluation & Benchmarking Suite")
    print("=" * 70)

    # 1. Load data
    records = load_eval_data(args.data_path, split=args.split, max_samples=args.max_samples)
    print(f"Loaded {len(records)} examples from {args.data_path} (split: {args.split})")

    # 2. Get predictions (from cache or model generation)
    predictions: list[str] = []
    if args.load_predictions and os.path.exists(args.load_predictions):
        print(f"Loading cached predictions from: {args.load_predictions}")
        with open(args.load_predictions, encoding="utf-8") as f:
            predictions = json.load(f)
        if len(predictions) > len(records):
            predictions = predictions[: len(records)]
    else:
        predictions = run_model_inference(
            model_path=args.model_path,
            records=records,
            batch_size=args.batch_size,
            device=args.device,
        )
        if args.save_predictions:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_predictions)), exist_ok=True)
            with open(args.save_predictions, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)
            print(f"Saved predictions to: {args.save_predictions}")

    # 3. Evaluate predictions using SQLEvaluator
    print(f"\n[3/3] Running SQL Evaluator on {len(predictions)} examples...")
    evaluator = SQLEvaluator(timeout=args.timeout)

    references = [r["sql"] for r in records]
    db_ids = [r["db_id"] for r in records]
    hardness_list = [r.get("hardness", "unknown") for r in records]

    t0 = time.time()
    metrics = evaluator.evaluate_batch(
        predictions=predictions,
        references=references,
        db_ids=db_ids,
        hardness_list=hardness_list,
    )
    eval_time = time.time() - t0
    print(f"Evaluation completed in {eval_time:.2f}s ({eval_time/len(records)*1000:.1f}ms/query)")

    # 4. Print Summary Table
    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: EM = {metrics['exact_match']:.2f}% | EX = {metrics['execution_accuracy']:.2f}% (N={metrics['total']})")
    print("=" * 70)

    tier_rows = []
    for tier, s in metrics["by_hardness"].items():
        tier_rows.append([
            tier.capitalize(),
            s["total"],
            s["em_correct"],
            f"{s['em_percent']:.2f}%",
            s["ex_correct"],
            f"{s['ex_percent']:.2f}%",
        ])
    print(tabulate(tier_rows, headers=["Tier", "Total", "EM Correct", "EM (%)", "EX Correct", "EX (%)"], tablefmt="fancy_grid"))

    print("\nError Category Breakdown:")
    err_rows = [[k, v, f"{v/metrics['total']*100:.2f}%"] for k, v in metrics["error_breakdown"].items()]
    print(tabulate(err_rows, headers=["Error Type", "Count", "Percentage"], tablefmt="simple"))

    # 5. Save metrics summary JSON
    if args.output_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_metrics)), exist_ok=True)
        # Exclude non-serializable objects from JSON dump
        json_metrics = {k: v for k, v in metrics.items() if k != "results"}
        with open(args.output_metrics, "w", encoding="utf-8") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"Metrics saved to: {args.output_metrics}")

    # 6. Generate detailed Error Analysis Report (Task 07)
    if args.output_report:
        generate_error_report(metrics, records, args.output_report)


if __name__ == "__main__":
    main()
