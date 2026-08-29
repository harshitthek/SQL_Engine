"""Database Schema Serializer.

Converts Spider tables.json into clean, human-readable text (SQLite DDL)
including column types, primary keys, and foreign key relationships.
Also supports extracting and embedding 3 sample rows per table from SQLite databases.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class SchemaSerializer:
    """Serializes Spider database schemas into text representations."""

    def __init__(self, tables_json_path: str, db_root_dir: Optional[str] = None):
        """Initialize the serializer with path to tables.json and optional db directory."""
        self.tables_json_path = tables_json_path
        self.db_root_dir = db_root_dir
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_tables()

    def _load_tables(self) -> None:
        """Load schemas from tables.json and index by db_id."""
        with open(self.tables_json_path, "r", encoding="utf-8") as f:
            tables_data = json.load(f)
        for entry in tables_data:
            self.schemas[entry["db_id"]] = entry

    def get_db_ids(self) -> List[str]:
        """Return list of all database IDs."""
        return list(self.schemas.keys())

    def get_sqlite_path(self, db_id: str) -> Optional[str]:
        """Get path to the sqlite file for a given db_id."""
        if not self.db_root_dir:
            return None
        candidate = os.path.join(self.db_root_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            return candidate
        return None

    def get_sample_rows(
        self, db_id: str, table_name: str, num_rows: int = 3
    ) -> List[Tuple[Any, ...]]:
        """Fetch sample rows for a table from its SQLite database."""
        sqlite_path = self.get_sqlite_path(db_id)
        if not sqlite_path or not os.path.exists(sqlite_path):
            return []

        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            # Escape quotes in table name
            escaped_name = table_name.replace('"', '""')
            cursor.execute(f'SELECT * FROM "{escaped_name}" LIMIT {num_rows};')
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception:
            return []

    def serialize_ddl(
        self, db_id: str, include_sample_rows: bool = False, num_sample_rows: int = 3
    ) -> str:
        """Serialize a database schema into standard SQLite DDL format.

        Includes column types, PRIMARY KEY constraints, and FOREIGN KEY constraints.
        Optionally appends sample rows beneath each table definition.
        """
        if db_id not in self.schemas:
            raise KeyError(f"Database ID '{db_id}' not found in tables.json")

        data = self.schemas[db_id]
        table_names = data.get("table_names_original", data.get("table_names", []))
        col_names = data.get("column_names_original", data.get("column_names", []))
        col_types = data.get("column_types", [])
        primary_keys = set(data.get("primary_keys", []))
        foreign_keys = data.get("foreign_keys", [])

        # Map table_index -> list of (col_index, col_name, col_type, is_pk)
        table_cols: Dict[int, List[Dict[str, Any]]] = {
            t_idx: [] for t_idx in range(len(table_names))
        }
        for col_idx, (tbl_idx, name) in enumerate(col_names):
            if tbl_idx == -1:  # special '*' column
                continue
            t_type = col_types[col_idx].upper() if col_idx < len(col_types) else "TEXT"
            is_pk = col_idx in primary_keys
            table_cols[tbl_idx].append(
                {
                    "col_idx": col_idx,
                    "name": name,
                    "type": t_type,
                    "is_pk": is_pk,
                }
            )

        # Map source_table_idx -> list of (src_col_name, target_table_name, target_col_name)
        fk_by_table: Dict[int, List[Tuple[str, str, str]]] = {
            t_idx: [] for t_idx in range(len(table_names))
        }
        for src_col_idx, tgt_col_idx in foreign_keys:
            if src_col_idx >= len(col_names) or tgt_col_idx >= len(col_names):
                continue
            src_tbl_idx, src_col_name = col_names[src_col_idx]
            tgt_tbl_idx, tgt_col_name = col_names[tgt_col_idx]
            i
# [WIP: table serialization]
