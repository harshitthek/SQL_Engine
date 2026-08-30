"""Empirical Study on Including Sample Rows in Prompts.

Investigates token count inflation and practical trade-offs of including
3 sample rows per table in the text-to-SQL prompt.
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
from tabulate import tabulate

try:
    from data.processing.prompt_templates import TEMPLATES
    from data.processing.schema_serializer import SchemaSerializer
    from data.processing.tokenizer_utils import QwenTokenizerWrapper
except ImportError:
    from .prompt_templates import TEMPLATES
    from .schema_serializer import SchemaSerializer
    from .tokenizer_utils import QwenTokenizerWrapper


class SampleRowsStudy:
    """Evaluates token count impact of sample rows across databases."""

    def __init__(
        self,
        tables_json_path: str = "data/spider_data/tables.json",
        db_root_dir: str = "data/spider_data/database",
        tokenizer_wrapper: Optional[QwenTokenizerWrapper] = None,
    ):
        self.serializer = SchemaSerializer(tables_json_path, db_root_dir)
        self.tokenizer_wrapper = tokenizer_wrapper or QwenTokenizerWrapper()
        self.template = TEMPLATES["markdown"]

    def run_study_on_databases(self) -> Dict[str, Any]:
        """Compare token count for pure schema vs schema + 3 sample rows per DB."""
        db_ids = self.serializer.get_db_ids()
        results = []

        for db_id in db_ids:
            # Schema only
            ddl_only = self.serializer.serialize_ddl(db_id, include_sample_rows=False)
            tokens_ddl = self.tokenizer_wrapper.count_tokens(ddl_only)

            # Schema with 3 sample rows
            ddl_with_rows = self.serializer.serialize_ddl(
                db_id, include_sample_rows=True, num_sample_rows=3
            )
            tokens_with_rows = self.tokenizer_wrapper.count_tokens(ddl_with_rows)

            inflation = tokens_with_rows - tokens_ddl
            pct_increase = (
                (inflation / tokens_ddl * 100) if tokens_ddl > 0 else 0.0
            )

            num_tables = len(
                self.serializer.schemas[db_id].get("table_names_original", [])
            )

            results.append(
                {
                    "db_id": db_id,
                    "num_tables": num_tables,
                    "tokens_ddl": tokens_ddl,
                    "tokens_with_rows": tokens_with_rows,
                    "inflation": inflation,
                    "pct_increase": pct_increase,
                }
            )

        ddl_tokens = [r["tokens_ddl"] for r in results]
        with_rows_tokens = [r["tokens_with_rows"] for r in results]

        summary = {
            "num_dbs": len(db_ids),
            "ddl_stats": {
                "mean": float(np.mean(ddl_tokens)),
                "median": float(np.median(ddl_tokens)),
                "p95": float(np.percentile(ddl_tokens, 95)),
                "max": int(np.max(ddl_tokens)),
            },
            "with_rows_stats": {
                "mean": float(np.mean(with_rows_tokens)),
                "median": float(np.median(with_rows_tokens)),
                "p95": float(np.percentile(with_rows_tokens, 95)),
                "max": int(np.max(with_rows_tokens)),
            },
            "per_db_results": sorted(
                results, key=lambda x: x["inflation"], reverse=True
            ),
        }
        return summary

    def format_summary_table(self, summary: Dict[str, Any]) -> str:
        """Format the study summary into a clean markdown table."""
        d = summary["ddl_stats"]
        w = summary["with_rows_stats"]

        table_data = [
            ["Mean Tokens", f"{d['mean']:.1f}", f"{w['mean']:.1f}", f"+{w['mean'] - d['mean']:.1f} ({(w['mean']/d['mean'] - 1)*100:.1f}%)"],
            ["Median Tokens", f"{d['median']:.1f}", f"{w['median']:.1f}", f"+{w['median'] - d['median']:.1f} ({(w['median']/d['median'] - 1)*100:.1f}%)"],
            ["95th Percentile", f"{d['p95']:.1f}", f"{w['p95']:.1f}", f"+{w['p95'] - d['p95']:.1f} ({(w['p95']/d['p95'] - 1)*100:.1f}%)"],
            ["Max Tokens", f"{d['max']}", f"{w['max']}", f"+{w['max'] - d['max']} ({(w['max']/d['max'] - 1)*100:.1f}%)"],
        ]
        headers = ["Metric", "Schema Only", "Schema + 3 Rows", "Token Overhead"]
        return tabulate(table_data, headers=headers, tablefmt="github")
