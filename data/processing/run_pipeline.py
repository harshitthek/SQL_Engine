"""Unified Data Processing Pipeline Runner for Tasks 02–09 (Spider Benchmark)."""

import argparse
import json
import os
import sys

# Ensure SQL_Engine root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.processing.complexity import eval_hardness
from data.processing.prompt_templates import TEMPLATES
from data.processing.schema_serializer import SchemaSerializer
from data.processing.sample_rows_study import SampleRowsStudy
from data.processing.tokenizer_utils import QwenTokenizerWrapper
from data.processing.dataset_builder import DatasetBuilder
from data.processing.sanity_checker import SanityChecker
from tabulate import tabulate


def run_task_02():
    print("\n" + "=" * 60)
    print("TASK 02: Auditing Dataset Complexity Tiers")
    print("=" * 60)
    tables_path = os.path.join(REPO_ROOT, "data/spider_data/tables.json")
    train_path = os.path.join(REPO_ROOT, "data/spider_data/train_spider.json")
    dev_path = os.path.join(REPO_ROOT, "data/spider_data/dev.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    train_tiers = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
    dev_tiers = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}

    for ex in train_data:
        h = eval_hardness(ex["sql"])
        train_tiers[h] = train_tiers.get(h, 0) + 1

    for ex in dev_data:
        h = eval_hardness(ex["sql"])
        dev_tiers[h] = dev_tiers.get(h, 0) + 1

    table = [
        ["easy", train_tiers["easy"], dev_tiers["easy"]],
        ["medium", train_tiers["medium"], dev_tiers["medium"]],
        ["hard", train_tiers["hard"], dev_tiers["hard"]],
        ["extra", train_tiers["extra"], dev_tiers["extra"]],
        ["Total", len(train_data), len(dev_data)],
    ]
    print(tabulate(table, headers=["Tier", "Train Count", "Dev Count"], tablefmt="github"))

    res = {"train": train_tiers, "dev": dev_tiers, "total_train": len(train_data), "total_dev": len(dev_data)}
    os.makedirs(os.path.join(REPO_ROOT, "data/reports"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "data/reports/audit_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    # Also save to data/ for backward compatibility
    with open(os.path.join(REPO_ROOT, "data/audit_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Saved audit results to data/reports/audit_results.json")


def run_task_03():
    print("\n" + "=" * 60)
    print("TASK 03: Testing Instruction Prompt Templates")
    print("=" * 60)
    schema = "CREATE TABLE department (\n  Department_ID NUMBER PRIMARY KEY,\n  Name TEXT\n);"
    question = "List all department names."
    for name, tmpl in TEMPLATES.items():
        print(f"\n--- [Template: {name}] ---")
        prompt = tmpl.format_prompt(schema, question)
        print(prompt[:200] + ("..." if len(prompt) > 200 else ""))


def run_task_04():
    print("\n" + "=" * 60)
    print("TASK 04: Testing Schema Serializer (DDL Extraction)")
    print("=" * 60)
    serializer = SchemaSerializer(
        tables_json_path=os.path.join(REPO_ROOT, "data/spider_data/tables.json"),
        db_root_dir=os.path.join(REPO_ROOT, "data/spider_data/database"),
    )
    db_ids = serializer.get_db_ids()
    print(f"Loaded {len(db_ids)} databases from Spider tables.json.")
    sample_db = "department_management" if "department_management" in db_ids else db_ids[0]
    ddl = serializer.serialize_ddl(sample_db)
    print(f"\n--- [Serialized Schema DDL for '{sample_db}'] ---")
    print(ddl)


def run_task_05():
    print("\n" + "=" * 60)
    print("TASK 05: Empirical Study on Sample Rows (3 Rows Per Table)")
    print("=" * 60)
    study = SampleRowsStudy(
        tables_json_path=os.path.join(REPO_ROOT, "data/spider_data/tables.json"),
        db_root_dir=os.path.join(REPO_ROOT, "data/spider_data/database"),
    )
    results = study.run_study_on_databases()
    study.print_summary_table(results)
    os.makedirs(os.path.join(REPO_ROOT, "data/reports"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "data/reports/sample_rows_study_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(REPO_ROOT, "data/sample_rows_study_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved sample rows study results to data/reports/sample_rows_study_results.json")


def run_task_06():
    print("\n" + "=" * 60)
    print("TASK 06: Tokenize with Qwen2.5 & Plot Distribution")
    print("=" * 60)
    builder = DatasetBuilder(
        tables_json_path=os.path.join(REPO_ROOT, "data/spider_data/tables.json"),
        db_root_dir=os.path.join(REPO_ROOT, "data/spider_data/database"),
    )
    train_paths = [
        os.path.join(REPO_ROOT, "data/spider_data/train_spider.json"),
        os.path.join(REPO_ROOT, "data/spider_data/train_others.json"),
    ]
    records, _ = builder.process_split(train_paths, max_seq_length=999999)
    lengths = [r["token_length"] for r in records]

    tokenizer_wrapper = builder.tokenizer_wrapper
    stats = tokenizer_wrapper.analyze_token_lengths(lengths)
    print("\nToken Length Percentiles:")
    for k, v in stats.items():
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

    os.makedirs(os.path.join(REPO_ROOT, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "data/reports"), exist_ok=True)
    fig_path = os.path.join(REPO_ROOT, "artifacts/token_distribution.png")
    tokenizer_wrapper.plot_token_distribution(lengths, cutoff_95=stats["p95"], output_path=fig_path)

    with open(os.path.join(REPO_ROOT, "data/reports/token_distribution_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(REPO_ROOT, "data/token_distribution_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved distribution plot to {fig_path}")


def run_task_07_08(max_seq_length=1024):
    print("\n" + "=" * 60)
    print(f"TASKS 07 & 08: Filter by max_seq_length={max_seq_length} & Export Arrow Dataset")
    print("=" * 60)
    builder = DatasetBuilder(
        tables_json_path=os.path.join(REPO_ROOT, "data/spider_data/tables.json"),
        db_root_dir=os.path.join(REPO_ROOT, "data/spider_data/database"),
    )
    train_paths = [
        os.path.join(REPO_ROOT, "data/spider_data/train_spider.json"),
        os.path.join(REPO_ROOT, "data/spider_data/train_others.json"),
    ]
    dev_paths = [os.path.join(REPO_ROOT, "data/spider_data/dev.json")]

    ds_dict, stats = builder.build_and_export(
        train_paths=train_paths,
        dev_paths=dev_paths,
        output_dir=os.path.join(REPO_ROOT, "data/processed_arrow"),
        max_seq_length=max_seq_length,
    )
    builder.log_drop_statistics(stats)
    os.makedirs(os.path.join(REPO_ROOT, "data/reports"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "data/reports/filter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(REPO_ROOT, "data/filter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved Arrow dataset to data/processed_arrow/ and data/processed/")


def run_task_09():
    print("\n" + "=" * 60)
    print("TASK 09: Data Sanity Check with 20 Rendered Prompts")
    print("=" * 60)
    from datasets import load_from_disk
    dataset = load_from_disk(os.path.join(REPO_ROOT, "data/processed_arrow"))
    train_records = [ex for ex in dataset["train"]]

    checker = SanityChecker(
        tables_json_path=os.path.join(REPO_ROOT, "data/spider_data/tables.json"),
        db_root_dir=os.path.join(REPO_ROOT, "data/spider_data/database"),
    )
    report_path = os.path.join(REPO_ROOT, "artifacts/data_sanity_report.md")
    checker.sample_and_render(train_records, num_samples=20, output_file=report_path)
    print(f"Exported 20 sample sanity report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Spider Text-to-SQL Data Processing Pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "02", "03", "04", "05", "06", "07_08", "09"],
        help="Task to run (default: all)",
    )
    args = parser.parse_args()

    if args.task in ("all", "02"):
        run_task_02()
    if args.task in ("all", "03"):
        run_task_03()
    if args.task in ("all", "04"):
        run_task_04()
    if args.task in ("all", "05"):
        run_task_05()
    if args.task in ("all", "06"):
        run_task_06()
    if args.task in ("all", "07_08"):
        run_task_07_08()
    if args.task in ("all", "09"):
        run_task_09()

    print("\n" + "=" * 60)
    print("ALL REQUESTED DATA PROCESSING TASKS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
