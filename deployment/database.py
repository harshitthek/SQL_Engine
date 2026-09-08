from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional
from urllib.parse import quote_plus

from sqlalchemy import (
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


@dataclass
class DatabaseConfig:

    db_type: str

    database: str

    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None

    url: Optional[str] = None


def adapt_sql_dialect(sql: str, dialect: str = "sqlite") -> str:
    """
    Dynamically adapt SQL queries across database dialects (PostgreSQL, MySQL, SQLite).

    Transforms dialect-specific idioms:
    - Automatically maps SQLite metadata tables (sqlite_master, sqlite_schema) to
      information_schema.tables with table_schema = 'public' and table_name when targeting PostgreSQL.
    - Replaces SQLite-style double quotes on string literals (e.g. type = "table") with single quotes.
    - Harmonizes functions: IFNULL -> COALESCE, strftime -> TO_CHAR / EXTRACT.
    """
    if not sql or not isinstance(sql, str):
        return sql

    target_dialect = (dialect or "sqlite").strip().lower()

    if target_dialect in ("postgresql", "postgres", "psql", "supabase"):
        adapted = sql

        # 1. Convert SQLite-style double-quoted string literals to single quotes
        # Comparisons: = "table", != "val", <> "val", <= "val", etc.
        adapted = re.sub(r'([=!<>]=?|<>)\s*"([^"]+)"', r"\1 '\2'", adapted)
        # LIKE / ILIKE
        adapted = re.sub(r'\b(LIKE|ILIKE|NOT\s+LIKE)\s+"([^"]+)"', r"\1 '\2'", adapted, flags=re.IGNORECASE)
        # BETWEEN
        adapted = re.sub(r'\bBETWEEN\s+"([^"]+)"\s+AND\s+"([^"]+)"', r"BETWEEN '\1' AND '\2'", adapted, flags=re.IGNORECASE)
        # IN (...) lists with double-quoted literals
        def _fix_in_quotes(match: re.Match) -> str:
            in_content = match.group(1)
            fixed = re.sub(r'"([^"]+)"', r"'\1'", in_content)
            return f"IN ({fixed})"
        adapted = re.sub(r'\bIN\s*\(([^)]+)\)', _fix_in_quotes, adapted, flags=re.IGNORECASE)
        # Standalone metadata literals if double-quoted
        adapted = re.sub(r'"(table|view|index)"', r"'\1'", adapted, flags=re.IGNORECASE)

        # 2. Harmonize IFNULL -> COALESCE
        adapted = re.sub(r'\bIFNULL\b', 'COALESCE', adapted, flags=re.IGNORECASE)

        # 3. Harmonize strftime -> EXTRACT or TO_CHAR
        # Case A: strftime('%Y', col) = 2024 -> EXTRACT(YEAR FROM col) = 2024
        adapted = re.sub(
            r"\bstrftime\s*\(\s*['\"]%Y['\"]\s*,\s*([^)]+)\)\s*([=!<>]=?|<>)\s*(\d+)\b",
            r"EXTRACT(YEAR FROM \1) \2 \3",
            adapted,
            flags=re.IGNORECASE,
        )
        # Case B: General strftime(fmt, col) -> TO_CHAR(col, translated_fmt)
        def _translate_strftime(match: re.Match) -> str:
            raw_fmt = match.group(1).strip("'\"")
            col_expr = match.group(2).strip()
            fmt_mapping = {
                "%Y": "YYYY",
                "%m": "MM",
                "%d": "DD",
                "%H": "HH24",
                "%M": "MI",
                "%S": "SS",
                "%j": "DDD",
                "%w": "D",
            }
            pg_fmt = raw_fmt
            for k, v in fmt_mapping.items():
                pg_fmt = pg_fmt.replace(k, v)
            return f"TO_CHAR({col_expr}, '{pg_fmt}')"

        adapted = re.sub(
            r"\bstrftime\s*\(\s*(['\"][^'\"]+['\"])\s*,\s*([^)]+)\)",
            _translate_strftime,
            adapted,
            flags=re.IGNORECASE,
        )

        # 4. Map sqlite_master / sqlite_schema -> information_schema.tables
        if re.search(r'\b(sqlite_master|sqlite_schema)\b', adapted, flags=re.IGNORECASE):
            adapted = re.sub(r'\b(sqlite_master|sqlite_schema)\b', 'information_schema.tables', adapted, flags=re.IGNORECASE)

            # Map type = 'table' / 'view'
            adapted = re.sub(
                r"\btype\s*=\s*['\"]table['\"]",
                "table_schema = 'public' AND table_type = 'BASE TABLE'",
                adapted,
                flags=re.IGNORECASE,
            )
            adapted = re.sub(
                r"\btype\s*=\s*['\"]view['\"]",
                "table_schema = 'public' AND table_type = 'VIEW'",
                adapted,
                flags=re.IGNORECASE,
            )
            adapted = re.sub(
                r"\btype\s*IN\s*\(\s*['\"]table['\"]\s*,\s*['\"]view['\"]\s*\)",
                "table_schema = 'public' AND table_type IN ('BASE TABLE', 'VIEW')",
                adapted,
                flags=re.IGNORECASE,
            )

            # Ensure table_schema = 'public' is present
            if "table_schema" not in adapted.lower():
                where_match = re.search(r'\bWHERE\b', adapted, flags=re.IGNORECASE)
                if where_match:
                    idx = where_match.end()
                    adapted = adapted[:idx] + " table_schema = 'public' AND" + adapted[idx:]
                else:
                    clause_match = re.search(r'\b(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b|;', adapted, flags=re.IGNORECASE)
                    if clause_match:
                        idx = clause_match.start()
                        matched_str = clause_match.group(0)
                        if matched_str == ";":
                            adapted = adapted[:idx].rstrip() + " WHERE table_schema = 'public';"
                        else:
                            adapted = adapted[:idx].rstrip() + " WHERE table_schema = 'public' " + adapted[idx:].lstrip()
                    else:
                        adapted = adapted.rstrip() + " WHERE table_schema = 'public'"

            # Map column names: name / tbl_name -> table_name
            adapted = re.sub(r'\b(?<!table_)name\b', 'table_name', adapted, flags=re.IGNORECASE)
            adapted = re.sub(r'\btbl_name\b', 'table_name', adapted, flags=re.IGNORECASE)
            adapted = re.sub(r'\b(?<!table_)type\b', 'table_type', adapted, flags=re.IGNORECASE)

        return adapted

    elif target_dialect in ("mysql", "mariadb"):
        adapted = sql
        adapted = re.sub(r'([=!<>]=?|<>)\s*"([^"]+)"', r"\1 '\2'", adapted)
        adapted = re.sub(r'\b(LIKE|ILIKE|NOT\s+LIKE)\s+"([^"]+)"', r"\1 '\2'", adapted, flags=re.IGNORECASE)
        adapted = re.sub(r'\bBETWEEN\s+"([^"]+)"\s+AND\s+"([^"]+)"', r"BETWEEN '\1' AND '\2'", adapted, flags=re.IGNORECASE)
        def _fix_in_quotes_mysql(match: re.Match) -> str:
            in_content = match.group(1)
            fixed = re.sub(r'"([^"]+)"', r"'\1'", in_content)
            return f"IN ({fixed})"
        adapted = re.sub(r'\bIN\s*\(([^)]+)\)', _fix_in_quotes_mysql, adapted, flags=re.IGNORECASE)
        adapted = re.sub(r'"(table|view|index)"', r"'\1'", adapted, flags=re.IGNORECASE)
        if re.search(r'\b(sqlite_master|sqlite_schema)\b', adapted, flags=re.IGNORECASE):
            adapted = re.sub(r'\b(sqlite_master|sqlite_schema)\b', 'information_schema.tables', adapted, flags=re.IGNORECASE)
            adapted = re.sub(
                r"\btype\s*=\s*['\"]table['\"]",
                "table_schema = DATABASE() AND table_type = 'BASE TABLE'",
                adapted,
                flags=re.IGNORECASE
# [WIP: database engine]
