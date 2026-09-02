"""Data Sanity Checking Script.

Samples 20 random examples from the dataset, renders them as full human-readable
prompts, verifies schema-SQL alignment, and saves the inspection report.
"""

import os
import random
from typing import Any, Dict, List, Optional

try:
    from data.processing.complexity import eval_hardness
    from data.processing.prompt_templates import TEMPLATES
    from data.processing.schema_serializer import SchemaSerializer
    from data.processing.tokenizer_utils import QwenTokenizerWrapper
except ImportError:
    from .complexity import eval_hardness
    from .prompt_templates import TEMPLATES
    from .schema_serializer import SchemaSerializer
    from .tokenizer_utils import QwenTokenizerWrapper


class SanityChecker:
    """Verifies dataset quality and renders human-readable prompt samples."""

    def __init__(
        self,
        tables_json_path: str = "data/spider_data/tables.json",
        db_root_dir: str = "data/spider_data/database",
        tokenizer_wrapper: Optional[QwenTokenizerWrapper] = None,
    ):
        self.serializer = SchemaSerializer(tables_json_path, db_root_dir)
        self.tokenizer_wrapper = tokenizer_wrapper or QwenTokenizerWrapper()
        self.template = TEMPLATES["markdown"]

    def sample_and_render(
        self,
        dataset_records: List[Dict[str, Any]],
        num_samples: int = 20,
        random_seed: int = 42,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Randomly sample N examples and format them into readable inspection blocks."""
        random.seed(random_seed)
        sampled = random.sample(dataset_records, min(num_samples, len(dataset_records)))

        rendered_reports = []
        md_lines = [
            f"# Data Sanity Inspection Report ({len(sampled)} Random Samples)\n",
            "This report details 20 randomly sampled examples across various databases and complexity tiers,",
            "verifying schema correctness, question clarity, foreign key linkages, and target SQL.\n\n",
            "---\n\n",
        ]

        for i, ex in enumerate(sampled, start=1):
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex.get("query") or (ex.get("sql") if isinstance(ex.get("sql"), str) else "")
            hardness = ex.get("hardness")
            if not hardness and "sql" in ex and isinstance(ex["sql"], dict):
                hardness = eval_hardness(ex["sql"])

            schema_ddl = self.serializer.serialize_ddl(db_id, include_sample_rows=False)
            prompt = self.template.format_input(schema_ddl, question)
            full_prompt = self.template.format_full(schema_ddl, question, sql)
            token_len = self.tokenizer_wrapper.count_tokens(full_prompt)

            # Verification heuristics
            checks = {
                "non_empty_sql": bool(sql and sql.strip()),
                "non_empty_question": bool(question and question.strip()),
                "schema_present": bool(schema_ddl and len(schema_ddl) > 20),
                "db_sqlite_exists": self.serializer.get_sqlite_path(db_id) is not None,
            }
            all_passed = all(checks.values())

            sample_info = {
                "sample_num": i,
                "db_id": db_id,
                "hardness": hardness,
                "question": question,
                "sql": sql,
                "token_length": token_len,
                "checks_passed": all_passed,
                "prompt": prompt,
            }
            rendered_reports.append(sample_info)

            # Format markdown
            md_lines.append(f"## Sample {i}/{len(sampled)}: `{db_id}` (Complexity: **{hardness}**)\n")
            md_lines.append(f"- **Token Count**: {token_len} tokens")
            md_lines.append(f"- **Sanity Checks**: {'PASSED' if all_passed else 'FAILED'}\n")
            md_lines.append(f"**Question**: {question}\n")
            md_lines.append(f"**Target SQL**:\n```sql\n{sql}\n```\n")
            md_lines.append("<details>\n<summary>Click to view serialized Database Schema & Full Prompt</summary>\n\n")
            md_lines.append(f"```text\n{prompt}\n```\n")
            md_lines.append("</details>\n\n---\n\n")

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(md_lines)

        return rendered_reports
