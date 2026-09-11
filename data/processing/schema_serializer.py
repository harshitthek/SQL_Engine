"""Database Schema Serializer.

Converts Spider tables.json into clean, human-readable text (SQLite DDL)
including column types, primary keys, and foreign key relationships.
Also supports extracting and embedding 3 sample rows per table from SQLite databases.
"""

import json
import os
import sqlite3
from typing import Any


class SchemaSerializer:
    """Serializes Spider database schemas into text representations."""

    def __init__(self, tables_json_path: str, db_root_dir: str | None = None):
        """Initialize the serializer with path to tables.json and optional db directory."""
        self.tables_json_path = tables_json_path
        self.db_root_dir = db_root_dir
        self.schemas: dict[str, dict[str, Any]] = {}
        self._load_tables()

    def _load_tables(self) -> None:
        """Load schemas from tables.json and index by db_id."""
        with open(self.tables_json_path, encoding="utf-8") as f:
            tables_data = json.load(f)
        for entry in tables_data:
            self.schemas[entry["db_id"]] = entry

    def get_db_ids(self) -> list[str]:
        """Return list of all database IDs."""
        return list(self.schemas.keys())

    def get_sqlite_path(self, db_id: str) -> str | None:
        """Get path to the sqlite file for a given db_id."""
        if not self.db_root_dir:
            return None
        candidate = os.path.join(self.db_root_dir, db_id, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            return candidate
        return None

    def get_sample_rows(
        self, db_id: str, table_name: str, num_rows: int = 3
    ) -> list[tuple[Any, ...]]:
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
        table_cols: dict[int, list[dict[str, Any]]] = {
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
        fk_by_table: dict[int, list[tuple[str, str, str]]] = {
            t_idx: [] for t_idx in range(len(table_names))
        }
        for src_col_idx, tgt_col_idx in foreign_keys:
            if src_col_idx >= len(col_names) or tgt_col_idx >= len(col_names):
                continue
            src_tbl_idx, src_col_name = col_names[src_col_idx]
            tgt_tbl_idx, tgt_col_name = col_names[tgt_col_idx]
            if src_tbl_idx >= 0 and tgt_tbl_idx >= 0:
                tgt_tbl_name = table_names[tgt_tbl_idx]
                fk_by_table[src_tbl_idx].append(
                    (src_col_name, tgt_tbl_name, tgt_col_name)
                )

        # Build DDL string
        statements = []
        for t_idx, t_name in enumerate(table_names):
            lines = []
            cols = table_cols.get(t_idx, [])
            pk_cols = [c["name"] for c in cols if c["is_pk"]]

            # Format columns
            for c in cols:
                # If single PK, can put inline; if multiple or standard, put column definition
                c_line = f"  `{c['name']}` {c['type']}"
                if c["is_pk"] and len(pk_cols) == 1:
                    c_line += " PRIMARY KEY"
                lines.append(c_line)

            # Composite PK constraint if multiple
            if len(pk_cols) > 1:
                pk_str = ", ".join([f"`{name}`" for name in pk_cols])
                lines.append(f"  PRIMARY KEY ({pk_str})")

            # Foreign keys
            fks = fk_by_table.get(t_idx, [])
            for src_c, tgt_t, tgt_c in fks:
                lines.append(
                    f"  FOREIGN KEY (`{src_c}`) REFERENCES `{tgt_t}`(`{tgt_c}`)"
                )

            ddl = f"CREATE TABLE `{t_name}` (\n" + ",\n".join(lines) + "\n);"

            # Add sample rows if requested
            if include_sample_rows:
                sample_rows = self.get_sample_rows(
                    db_id, t_name, num_rows=num_sample_rows
                )
                if sample_rows:
                    col_headers = [c["name"] for c in cols]
                    header_str = " | ".join(col_headers)
                    row_strs = []
                    for row in sample_rows:
                        # Limit string length to avoid extreme values
                        row_strs.append(
                            " | ".join(
                                [
                                    str(val)[:50] if val is not None else "NULL"
                                    for val in row
                                ]
                            )
                        )
                    sample_block = (
                        f"\n/*\n3 sample rows from `{t_name}`:\n{header_str}\n"
                        + "\n".join(row_strs)
                        + "\n*/"
                    )
                    ddl += sample_block

            statements.append(ddl)

        return "\n\n".join(statements)

    def serialize_compact(self, db_id: str) -> str:
        """Serialize schema into a compact text format:

        Table: table_name (col1: TYPE [PK], col2: TYPE, ...)
        Foreign keys: table.col -> other.col
        """
        if db_id not in self.schemas:
            raise KeyError(f"Database ID '{db_id}' not found in tables.json")

        data = self.schemas[db_id]
        table_names = data.get("table_names_original", data.get("table_names", []))
        col_names = data.get("column_names_original", data.get("column_names", []))
        col_types = data.get("column_types", [])
        primary_keys = set(data.get("primary_keys", []))
        foreign_keys = data.get("foreign_keys", [])

        table_lines = []
        for t_idx, t_name in enumerate(table_names):
            cols_desc = []
            for col_idx, (tbl_idx, name) in enumerate(col_names):
                if tbl_idx == t_idx:
                    t_type = col_types[col_idx] if col_idx < len(col_types) else "text"
                    pk_marker = " [PK]" if col_idx in primary_keys else ""
                    cols_desc.append(f"{name}: {t_type}{pk_marker}")
            table_lines.append(f"Table `{t_name}`: " + ", ".join(cols_desc))

        fk_lines = []
        for src_col_idx, tgt_col_idx in foreign_keys:
            if src_col_idx < len(col_names) and tgt_col_idx < len(col_names):
                src_tbl, src_col = col_names[src_col_idx]
                tgt_tbl, tgt_col = col_names[tgt_col_idx]
                if src_tbl >= 0 and tgt_tbl >= 0:
                    fk_lines.append(
                        f"{table_names[src_tbl]}.{src_col} -> {table_names[tgt_tbl]}.{tgt_col}"
                    )

        output = "\n".join(table_lines)
        if fk_lines:
            output += "\nForeign Keys:\n" + "\n".join(fk_lines)
        return output
