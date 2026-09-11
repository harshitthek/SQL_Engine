"""Dataset Builder for Text-to-SQL.

Applies prompt templates, filters examples exceeding max_seq_length,
logs drop statistics per complexity tier, builds HuggingFace Dataset objects,
and exports them to disk in Apache Arrow format.
"""

import json
import os
from collections import Counter
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from tabulate import tabulate

try:
    from data.processing.complexity import eval_hardness
    from data.processing.prompt_templates import TEMPLATES, PromptTemplate
    from data.processing.schema_serializer import SchemaSerializer
    from data.processing.tokenizer_utils import QwenTokenizerWrapper
except ImportError:
    from .complexity import eval_hardness
    from .prompt_templates import TEMPLATES, PromptTemplate
    from .schema_serializer import SchemaSerializer
    from .tokenizer_utils import QwenTokenizerWrapper


class DatasetBuilder:
    """Builds, filters, and serializes HuggingFace Arrow datasets."""

    def __init__(
        self,
        tables_json_path: str = "data/spider_data/tables.json",
        db_root_dir: str = "data/spider_data/database",
        template_name: str = "markdown",
        tokenizer_wrapper: QwenTokenizerWrapper | None = None,
    ):
        self.serializer = SchemaSerializer(tables_json_path, db_root_dir)
        self.template: PromptTemplate = TEMPLATES[template_name]
        self.tokenizer_wrapper = tokenizer_wrapper or QwenTokenizerWrapper()

    def process_split(
        self,
        data_paths: list[str],
        max_seq_length: int,
        include_sample_rows: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process raw JSON examples: serialize, compute tokens, filter by max_seq_length."""
        raw_examples = []
        for path in data_paths:
            with open(path, encoding="utf-8") as f:
                raw_examples.extend(json.load(f))

        kept_records = []
        dropped_records = []

        # Cache serialized schemas to avoid recomputing for every example
        schema_cache: dict[str, str] = {}

        for idx, ex in enumerate(raw_examples):
            db_id = ex["db_id"]
            question = ex["question"]
            sql_query = ex["query"]
            sql_ast = ex.get("sql", {})

            # Hardness classification
            hardness = eval_hardness(sql_ast)

            # Retrieve or serialize schema
            cache_key = f"{db_id}_{include_sample_rows}"
            if cache_key not in schema_cache:
                schema_cache[cache_key] = self.serializer.serialize_ddl(
                    db_id, include_sample_rows=include_sample_rows
                )
            schema_text = schema_cache[cache_key]

            # Format full training sequence and input prompt
            full_prompt = self.template.format_full(schema_text, question, sql_query)
            input_prompt = self.template.format_input(schema_text, question)

            # Tokenize
            token_count = self.tokenizer_wrapper.count_tokens(full_prompt)

            record = {
                "id": idx,
                "db_id": db_id,
                "question": question,
                "sql": sql_query,
                "hardness": hardness,
                "prompt": input_prompt,
                "full_text": full_prompt,
                "token_length": token_count,
            }

            if token_count > max_seq_length:
                dropped_records.append(record)
            else:
                kept_records.append(record)

        stats = {
            "total_input": len(raw_examples),
            "kept_count": len(kept_records),
            "dropped_count": len(dropped_records),
            "dropped_pct": (
                (len(dropped_records) / len(raw_examples) * 100)
                if raw_examples
                else 0.0
            ),
            "dropped_by_tier": dict(
                Counter(r["hardness"] for r in dropped_records)
            ),
            "kept_by_tier": dict(Counter(r["hardness"] for r in kept_records)),
            "dropped_by_db": dict(
                Counter(r["db_id"] for r in dropped_records)
            ),
        }

        return kept_records, stats

    def build_dataset_dict(
        self,
        train_records: list[dict[str, Any]],
        val_records: list[dict[str, Any]],
    ) -> DatasetDict:
        """Convert record dictionaries to HuggingFace DatasetDict."""
        train_dataset = Dataset.from_list(train_records)
        val_dataset = Dataset.from_list(val_records)
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    def save_arrow_dataset(
        self, dataset_dict: DatasetDict, output_dir: str
    ) -> None:
        """Serialize DatasetDict to disk in Apache Arrow format."""
        os.makedirs(output_dir, exist_ok=True)
        dataset_dict.save_to_disk(output_dir)

    @staticmethod
    def verify_saved_dataset(output_dir: str) -> DatasetDict:
        """Load and verify the saved Arrow dataset."""
        ds = load_from_disk(output_dir)
        return ds

    @staticmethod
    def format_drop_stats_table(stats: dict[str, Any], split_name: str) -> str:
        """Format dropping summary as a table."""
        tiers = ["simple", "medium", "hard", "extra-hard"]
        table_data = []
        for t in tiers:
            kept = stats["kept_by_tier"].get(t, 0)
            dropped = stats["dropped_by_tier"].get(t, 0)
            total = kept + dropped
            pct_drop = (dropped / total * 100) if total > 0 else 0.0
            table_data.append([t, total, kept, dropped, f"{pct_drop:.2f}%"])

        tot_kept = stats["kept_count"]
        tot_dropped = stats["dropped_count"]
        tot_all = stats["total_input"]
        overall_pct = (tot_dropped / tot_all * 100) if tot_all > 0 else 0.0
        table_data.append(["TOTAL", tot_all, tot_kept, tot_dropped, f"{overall_pct:.2f}%"])

        headers = ["Complexity Tier", "Total Examples", "Kept", "Dropped", "Drop Rate"]
        title = f"### Filtering Results ({split_name})\n"
        return title + tabulate(table_data, headers=headers, tablefmt="github")
