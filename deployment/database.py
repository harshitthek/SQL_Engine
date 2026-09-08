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
                flags=re.IGNORECASE,
            )
            adapted = re.sub(
                r"\btype\s*=\s*['\"]view['\"]",
                "table_schema = DATABASE() AND table_type = 'VIEW'",
                adapted,
                flags=re.IGNORECASE,
            )
            adapted = re.sub(
                r"\btype\s*IN\s*\(\s*['\"]table['\"]\s*,\s*['\"]view['\"]\s*\)",
                "table_schema = DATABASE() AND table_type IN ('BASE TABLE', 'VIEW')",
                adapted,
                flags=re.IGNORECASE,
            )
            if "table_schema" not in adapted.lower():
                where_match = re.search(r'\bWHERE\b', adapted, flags=re.IGNORECASE)
                if where_match:
                    idx = where_match.end()
                    adapted = adapted[:idx] + " table_schema = DATABASE() AND" + adapted[idx:]
                else:
                    clause_match = re.search(r'\b(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b|;', adapted, flags=re.IGNORECASE)
                    if clause_match:
                        idx = clause_match.start()
                        matched_str = clause_match.group(0)
                        if matched_str == ";":
                            adapted = adapted[:idx].rstrip() + " WHERE table_schema = DATABASE();"
                        else:
                            adapted = adapted[:idx].rstrip() + " WHERE table_schema = DATABASE() " + adapted[idx:].lstrip()
                    else:
                        adapted = adapted.rstrip() + " WHERE table_schema = DATABASE()"
            adapted = re.sub(r'\b(?<!table_)name\b', 'table_name', adapted, flags=re.IGNORECASE)
            adapted = re.sub(r'\btbl_name\b', 'table_name', adapted, flags=re.IGNORECASE)
            adapted = re.sub(r'\b(?<!table_)type\b', 'table_type', adapted, flags=re.IGNORECASE)

        return adapted

    return sql


class DatabaseManager:

    SUPPORTED_DATABASES = {
        "sqlite",
        "postgresql",
        "mysql",
        "supabase",
    }

    READ_ONLY_KEYWORDS = {
        "SELECT",
        "WITH",
        "EXPLAIN",
    }

    BLOCKED_KEYWORDS = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
        "ATTACH",
        "DETACH",
        "VACUUM",
        "PRAGMA",
    }

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: Optional[Engine] = None

    def _build_connection_url(self) -> str:

        if self.config.url:
            if self.config.db_type and self.config.db_type.strip().lower() == "supabase":
                if "sslmode=" in self.config.url:
                    return re.sub(r"sslmode=[^&]*", "sslmode=require", self.config.url)
                delimiter = "&" if "?" in self.config.url else "?"
                return f"{self.config.url}{delimiter}sslmode=require"
            return self.config.url

        db_type = (self.config.db_type or "").strip().lower()

        if db_type not in self.SUPPORTED_DATABASES:
            raise ValueError(
                f"Unsupported database type: {self.config.db_type}. "
                f"Supported types: {sorted(self.SUPPORTED_DATABASES)}"
            )

        if db_type == "sqlite":
            return f"sqlite:///{self.config.database}"

        if db_type in ("postgresql", "mysql"):
            if not all(
                [
                    self.config.host,
                    self.config.port,
                    self.config.username,
                    self.config.password is not None,
                ]
            ):
                raise ValueError(
                    "host, port, username and password are required "
                    f"for {db_type}."
                )

        if db_type == "supabase":
            if not self.config.host or not str(self.config.host).strip() or self.config.password is None:
                raise ValueError(
                    "host, port, username and password are required "
                    f"for {db_type}."
                )

            port = self.config.port if self.config.port is not None else 5432
            database = (
                self.config.database.strip().lstrip("/")
                if (self.config.database and self.config.database.strip())
                else "postgres"
            )
            username = (
                self.config.username.strip()
                if (self.config.username and self.config.username.strip())
                else "postgres"
            )

            if "sslmode=" in database:
                database_with_ssl = re.sub(r"sslmode=[^&]*", "sslmode=require", database)
            else:
                sep = "&" if "?" in database else "?"
                database_with_ssl = f"{database}{sep}sslmode=require"

            encoded_username = quote_plus(username)
            encoded_password = quote_plus(str(self.config.password))

            return (
                "postgresql://"
                f"{encoded_username}:"
                f"{encoded_password}@"
                f"{self.config.host}:"
                f"{port}/"
                f"{database_with_ssl}"
            )

        if db_type == "postgresql":
            return (
                "postgresql://"
                f"{self.config.username}:"
                f"{self.config.password}@"
                f"{self.config.host}:"
                f"{self.config.port}/"
                f"{self.config.database}"
            )

        if db_type == "mysql":
            return (
                "mysql+pymysql://"
                f"{self.config.username}:"
                f"{self.config.password}@"
                f"{self.config.host}:"
                f"{self.config.port}/"
                f"{self.config.database}"
            )

        raise ValueError(
            f"Unsupported database type: {db_type}"
        )

    def connect(self) -> Engine:

        if self.engine is not None:
            return self.engine

        connection_url = self._build_connection_url()

        try:
            self.engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

            # Immediately verify connectivity.
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))

            return self.engine

        except (SQLAlchemyError, ImportError, ModuleNotFoundError) as exc:
            self.engine = None

            raise ConnectionError(
                f"Failed to connect to database: {exc}"
            ) from exc

    def test_connection(self) -> bool:

        try:
            engine = self.connect()
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except (ConnectionError, SQLAlchemyError, ImportError, ModuleNotFoundError, Exception):
            self.engine = None
            return False

    def _get_target_schema(self) -> Optional[str]:
        if self.config.db_type and self.config.db_type.strip().lower() in ("supabase", "postgresql", "postgres", "psql"):
            return "public"
        return None

    def get_table_names(self) -> list[str]:

        engine = self.connect()
        inspector = inspect(engine)
        target_schema = self._get_target_schema()
        kwargs = {"schema": target_schema} if target_schema else {}

        return inspector.get_table_names(**kwargs)

    def get_schema(self) -> str:

        engine = self.connect()
        inspector = inspect(engine)
        target_schema = self._get_target_schema()
        kwargs = {"schema": target_schema} if target_schema else {}

        tables = inspector.get_table_names(**kwargs)

        if not tables:
            return "-- Database contains no tables."

        schema_parts: list[str] = []

        for table_name in tables:

            columns = inspector.get_columns(table_name, **kwargs)
            primary_key = inspector.get_pk_constraint(table_name, **kwargs)
            foreign_keys = inspector.get_foreign_keys(table_name, **kwargs)

            pk_columns = set(
                primary_key.get("constrained_columns") or []
            )

            column_lines: list[str] = []

            for column in columns:

                column_name = column["name"]
                column_type = str(column["type"])

                line = f"    {column_name} {column_type}"

                if column_name in pk_columns:
                    line += " PRIMARY KEY"

                if not column.get("nullable", True):
                    line += " NOT NULL"

                column_lines.append(line)

            table_block = (
                f"CREATE TABLE {table_name} (\n"
                + ",\n".join(column_lines)
                + "\n);"
            )

            schema_parts.append(table_block)

            # Foreign-key relationships
            for fk in foreign_keys:

                constrained_columns = fk.get(
                    "constrained_columns",
                    [],
                )

                referred_table = fk.get(
                    "referred_table"
                )

                referred_columns = fk.get(
                    "referred_columns",
                    [],
                )

                referred_schema = fk.get(
                    "referred_schema"
                )

                ref_target = (
                    f"{referred_schema}.{referred_table}"
                    if (referred_schema and referred_schema != target_schema)
                    else referred_table
                )

                for source_column, target_column in zip(
                    constrained_columns,
                    referred_columns,
                ):
                    schema_parts.append(
                        f"-- FOREIGN KEY: "
                        f"{table_name}.{source_column} "
                        f"REFERENCES "
                        f"{ref_target}.{target_column}"
                    )

        return "\n\n".join(schema_parts)

    def get_schema_dict(self) -> dict[str, Any]:

        engine = self.connect()
        inspector = inspect(engine)
        target_schema = self._get_target_schema()
        kwargs = {"schema": target_schema} if target_schema else {}

        result: dict[str, Any] = {}

        for table_name in inspector.get_table_names(**kwargs):

            columns = inspector.get_columns(table_name, **kwargs)
            primary_key = inspector.get_pk_constraint(table_name, **kwargs)
            foreign_keys = inspector.get_foreign_keys(table_name, **kwargs)

            result[table_name] = {
                "columns": [
                    {
                        "name": column["name"],
                        "type": str(column["type"]),
                        "nullable": column.get(
                            "nullable",
                            True,
                        ),
                    }
                    for column in columns
                ],
                "primary_key": (
                    primary_key.get(
                        "constrained_columns",
                        [],
                    )
                ),
                "foreign_keys": foreign_keys,
            }

        return result

    @classmethod
    def validate_sql(cls, sql: str) -> None:

        if not sql or not sql.strip():
            raise ValueError(
                "SQL query cannot be empty."
            )

        # Strip line comments (-- ...) and block comments (/* ... */)
        cleaned_sql