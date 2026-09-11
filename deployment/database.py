from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
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

    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None

    url: str | None = None


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

    if target_dialect in ("postgresql", "postgres", "psql", "supabase", "supabase_api", "supabase (api)", "supabase-api"):
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
        "supabase_api",
        "supabase (api)",
        "supabase-api",
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
        self.engine: Engine | None = None

    @property
    def is_api_mode(self) -> bool:
        """Return True if connecting via Supabase HTTP/REST/Management API rather than direct PostgreSQL."""
        return bool(
            self.config.db_type
            and self.config.db_type.strip().lower()
            in ("supabase_api", "supabase (api)", "supabase-api")
        )

    def _parse_supabase_api_info(self) -> tuple[str, str, str]:
        """
        Extracts (project_ref, base_url, api_key_or_token) from configuration.
        """
        raw_target = (self.config.database or self.config.url or "").strip()
        token = str(self.config.password or "").strip()

        if "://" in raw_target:
            from urllib.parse import urlparse
            parsed = urlparse(raw_target)
            host = parsed.hostname or ""
            ref = host.split(".")[0] if "." in host else host
            base_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        elif "." in raw_target:
            ref = raw_target.split(".")[0]
            base_url = f"https://{raw_target}".rstrip("/")
        else:
            ref = raw_target
            base_url = f"https://{ref}.supabase.co" if ref else ""

        return ref, base_url, token

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

        if db_type in ("supabase_api", "supabase (api)", "supabase-api"):
            _, base_url, _ = self._parse_supabase_api_info()
            return base_url or "https://supabase.co"

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

    def connect(self) -> Engine | None:

        if self.is_api_mode:
            if not self.test_connection():
                ref, base_url, _ = self._parse_supabase_api_info()
                raise ConnectionError(
                    f"Failed to connect to Supabase API at '{base_url}' (ref: '{ref}'). "
                    "Please verify your Project URL/Ref and API Key or Personal Access Token."
                )
            return None

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

        if self.is_api_mode:
            import requests
            try:
                ref, base_url, token = self._parse_supabase_api_info()
                if not ref or not token:
                    return False
                if token.startswith("sbp_"):
                    # Personal Access Token: test management API
                    resp = requests.get(
                        f"https://api.supabase.com/v1/projects/{ref}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=5.0,
                    )
                    return resp.status_code == 200
                else:
                    # PostgREST API: ping root OpenAPI endpoint
                    resp = requests.get(
                        f"{base_url}/rest/v1/",
                        headers={"apikey": token, "Authorization": f"Bearer {token}"},
                        timeout=5.0,
                    )
                    return resp.status_code == 200
            except Exception:
                return False

        try:
            engine = self.connect()
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except (ConnectionError, SQLAlchemyError, ImportError, ModuleNotFoundError, Exception):
            self.engine = None
            return False

    def _get_target_schema(self) -> str | None:
        if self.config.db_type and self.config.db_type.strip().lower() in ("supabase", "postgresql", "postgres", "psql", "supabase_api", "supabase (api)", "supabase-api"):
            return "public"
        return None

    def get_table_names(self) -> list[str]:

        if self.is_api_mode:
            return self._get_table_names_api()

        engine = self.connect()
        inspector = inspect(engine)
        target_schema = self._get_target_schema()
        kwargs = {"schema": target_schema} if target_schema else {}

        return inspector.get_table_names(**kwargs)

    def get_schema(self) -> str:

        if self.is_api_mode:
            return self._get_schema_api()

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

        if self.is_api_mode:
            return self._get_schema_dict_api()

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
        cleaned_sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
        cleaned_sql = re.sub(r"/\*.*?\*/", "", cleaned_sql, flags=re.DOTALL).strip()

        if not cleaned_sql:
            raise ValueError(
                "SQL query cannot be empty."
            )

        # Strip string literals to safely inspect structure and keywords without false positives
        # (e.g. semicolons or keywords like 'update'/'delete' inside string literals)
        code_without_literals = re.sub(r"'(?:''|[^'])*'", "''", cleaned_sql)
        code_without_literals = re.sub(r'"(?:""|[^"])*"', '""', code_without_literals)

        statements = [
            statement.strip()
            for statement in code_without_literals.split(";")
            if statement.strip()
        ]

        if len(statements) > 1:
            raise ValueError(
                "Multiple SQL statements are not allowed."
            )

        normalized = code_without_literals.upper()

        first_token = (
            normalized.split()[0]
            if normalized.split()
            else ""
        )

        if first_token not in cls.READ_ONLY_KEYWORDS:
            raise ValueError(
                "Only SELECT, WITH and EXPLAIN queries "
                "are allowed."
            )

        for keyword in cls.BLOCKED_KEYWORDS:
            if re.search(r"\b" + keyword + r"\b", normalized):
                raise ValueError(
                    f"Blocked SQL operation detected: {keyword}"
                )

    def execute_query(
        self,
        sql: str,
        max_rows: int = 1000,
    ) -> dict[str, Any]:

        effective_dialect = self.config.db_type
        if self.is_api_mode:
            effective_dialect = "postgresql"
        elif self.engine is not None and hasattr(self.engine, "dialect") and hasattr(self.engine.dialect, "name"):
            effective_dialect = self.engine.dialect.name

        adapted_sql = adapt_sql_dialect(sql, dialect=effective_dialect)
        self.validate_sql(adapted_sql)

        if self.is_api_mode:
            return self._execute_query_api(adapted_sql, max_rows=max_rows)

        engine = self.connect()

        try:
            with engine.connect() as connection:

                result = connection.execute(
                    text(adapted_sql)
                )

                rows = result.fetchmany(max_rows)

                columns = list(
                    result.keys()
                )

                serialized_rows = [
                    list(row)
                    for row in rows
                ]

                return {
                    "columns": columns,
                    "rows": serialized_rows,
                    "row_count": len(
                        serialized_rows
                    ),
                }

        except SQLAlchemyError as exc:

            raise RuntimeError(
                f"SQL execution failed: {exc}"
            ) from exc

    def _get_table_names_api(self) -> list[str]:
        import requests
        ref, base_url, token = self._parse_supabase_api_info()
        if not ref or not token:
            raise ValueError("Supabase Project URL/Ref and API Key or Access Token are required.")

        if token.startswith("sbp_"):
            sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name;"
            resp = requests.post(
                f"https://api.supabase.com/v1/projects/{ref}/database/query",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"query": sql},
                timeout=10.0,
            )
            if resp.status_code in (200, 201):
                rows = resp.json()
                return [r.get("table_name") for r in rows if isinstance(r, dict) and r.get("table_name")]
            raise RuntimeError(f"Supabase Management API table query failed ({resp.status_code}): {resp.text}")
        else:
            resp = requests.get(
                f"{base_url}/rest/v1/",
                headers={"apikey": token, "Authorization": f"Bearer {token}", "Accept": "application/openapi+json"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Supabase PostgREST API error ({resp.status_code}): {resp.text}")
            data = resp.json()
            defs = data.get("definitions") or data.get("components", {}).get("schemas", {})
            return sorted(list(defs.keys()))

    def _get_schema_dict_api(self) -> dict[str, Any]:
        import requests
        ref, base_url, token = self._parse_supabase_api_info()
        result: dict[str, Any] = {}
        if not ref or not token:
            raise ValueError("Supabase Project URL/Ref and API Key or Access Token are required.")

        if token.startswith("sbp_"):
            sql_cols = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
            """
            sql_pks = """
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = 'public';
            """
            sql_fks = """
            SELECT
                tc.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public';
            """
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            url = f"https://api.supabase.com/v1/projects/{ref}/database/query"

            r_cols = requests.post(url, headers=headers, json={"query": sql_cols}, timeout=10.0)
            if r_cols.status_code not in (200, 201):
                raise RuntimeError(f"Failed to fetch columns from Supabase Management API: {r_cols.text}")
            cols_data = r_cols.json()

            r_pks = requests.post(url, headers=headers, json={"query": sql_pks}, timeout=10.0)
            pks_data = r_pks.json() if r_pks.status_code in (200, 201) else []
            pks_by_table: dict[str, list[str]] = {}
            for r in pks_data:
                if isinstance(r, dict):
                    t = r.get("table_name")
                    c = r.get("column_name")
                    pks_by_table.setdefault(t, []).append(c)

            r_fks = requests.post(url, headers=headers, json={"query": sql_fks}, timeout=10.0)
            fks_data = r_fks.json() if r_fks.status_code in (200, 201) else []
            fks_by_table: dict[str, list[dict[str, Any]]] = {}
            for fk in fks_data:
                if isinstance(fk, dict):
                    t = fk.get("source_table")
                    fks_by_table.setdefault(t, []).append(fk)

            for col in cols_data:
                t = col.get("table_name")
                if not t:
                    continue
                if t not in result:
                    result[t] = {
                        "columns": [],
                        "primary_key": pks_by_table.get(t, []),
                        "foreign_keys": fks_by_table.get(t, []),
                    }
                result[t]["columns"].append({
                    "name": col.get("column_name"),
                    "type": col.get("data_type"),
                    "nullable": col.get("is_nullable") != "NO",
                })
            return result
        else:
            resp = requests.get(
                f"{base_url}/rest/v1/",
                headers={"apikey": token, "Authorization": f"Bearer {token}", "Accept": "application/openapi+json"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch schema from Supabase PostgREST API ({resp.status_code}): {resp.text}")
            data = resp.json()
            defs = data.get("definitions") or data.get("components", {}).get("schemas", {})
            for tbl_name, tbl_info in defs.items():
                props = tbl_info.get("properties", {})
                req_fields = set(tbl_info.get("required", []))
                cols = []
                pks = []
                for cname, cinfo in props.items():
                    raw_type = cinfo.get("format") or cinfo.get("type", "text")
                    desc = cinfo.get("description", "")
                    if "<pk/>" in desc or "Primary Key" in desc or cname in req_fields:
                        pks.append(cname)
                    cols.append({
                        "name": cname,
                        "type": str(raw_type),
                        "nullable": cname not in req_fields,
                    })
                result[tbl_name] = {
                    "columns": cols,
                    "primary_key": pks,
                    "foreign_keys": [],
                }
            return result

    def _get_schema_api(self) -> str:
        schema_dict = self._get_schema_dict_api()
        if not schema_dict:
            return "-- Database contains no public tables."

        schema_parts = []
        type_mapping = {
            "integer": "INTEGER",
            "bigint": "BIGINT",
            "smallint": "SMALLINT",
            "string": "TEXT",
            "text": "TEXT",
            "boolean": "BOOLEAN",
            "number": "NUMERIC",
            "timestamp with time zone": "TIMESTAMP WITH TIME ZONE",
            "timestamp without time zone": "TIMESTAMP",
            "date": "DATE",
            "time": "TIME",
            "uuid": "UUID",
            "json": "JSON",
            "jsonb": "JSONB",
        }

        for table_name, table_info in schema_dict.items():
            column_lines = []
            pk_cols = set(table_info.get("primary_key", []))
            for column in table_info.get("columns", []):
                col_name = column["name"]
                raw_type = str(column["type"]).lower()
                col_type = type_mapping.get(raw_type, column["type"].upper())
                line = f"    {col_name} {col_type}"
                if col_name in pk_cols:
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

            for fk in table_info.get("foreign_keys", []):
                if isinstance(fk, dict):
                    src_tbl = fk.get("source_table", table_name)
                    src_col = fk.get("source_column") or (fk.get("constrained_columns") or [""])[0]
                    tgt_tbl = fk.get("target_table") or fk.get("referred_table")
                    tgt_col = fk.get("target_column") or (fk.get("referred_columns") or [""])[0]
                    if src_col and tgt_tbl and tgt_col:
                        schema_parts.append(
                            f"-- FOREIGN KEY: {src_tbl}.{src_col} REFERENCES {tgt_tbl}.{tgt_col}"
                        )

        return "\n\n".join(schema_parts)

    def _execute_query_api(self, sql: str, max_rows: int = 1000) -> dict[str, Any]:
        import requests
        ref, base_url, token = self._parse_supabase_api_info()
        if not ref or not token:
            raise ValueError("Supabase Project URL/Ref and API Key or Access Token are required.")

        if token.startswith("sbp_"):
            # Management API
            url = f"https://api.supabase.com/v1/projects/{ref}/database/query"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            resp = requests.post(url, headers=headers, json={"query": sql}, timeout=15.0)
            if resp.status_code not in (200, 201):
                err_msg = resp.text
                try:
                    err_json = resp.json()
                    err_msg = err_json.get("message") or err_json.get("error") or resp.text
                except Exception:
                    pass
                raise RuntimeError(f"Supabase API query execution failed: {err_msg}")

            data = resp.json()
            if isinstance(data, list):
                rows_data = data[:max_rows]
                if not rows_data:
                    return {"columns": [], "rows": [], "row_count": 0}
                cols = list(rows_data[0].keys()) if isinstance(rows_data[0], dict) else []
                rows = [
                    [r.get(c) for c in cols] if isinstance(r, dict) else [r]
                    for r in rows_data
                ]
                return {"columns": cols, "rows": rows, "row_count": len(rows)}
            else:
                return {"columns": ["result"], "rows": [[str(data)]], "row_count": 1}
        else:
            # PostgREST rpc execution
            rpc_url = f"{base_url}/rest/v1/rpc/exec_sql"
            headers = {
                "apikey": token,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            resp = requests.post(rpc_url, headers=headers, json={"query": sql}, timeout=15.0)
            if resp.status_code in (200, 201):
                data = resp.json()
                if isinstance(data, list):
                    rows_data = data[:max_rows]
                    if not rows_data:
                        return {"columns": [], "rows": [], "row_count": 0}
                    cols = list(rows_data[0].keys()) if isinstance(rows_data[0], dict) else []
                    rows = [
                        [r.get(c) for c in cols] if isinstance(r, dict) else [r]
                        for r in rows_data
                    ]
                    return {"columns": cols, "rows": rows, "row_count": len(rows)}
                return {"columns": ["result"], "rows": [[str(data)]], "row_count": 1}

            rpc_url2 = f"{base_url}/rest/v1/rpc/execute_sql"
            resp2 = requests.post(rpc_url2, headers=headers, json={"sql": sql}, timeout=15.0)
            if resp2.status_code in (200, 201):
                data = resp2.json()
                if isinstance(data, list):
                    rows_data = data[:max_rows]
                    if not rows_data:
                        return {"columns": [], "rows": [], "row_count": 0}
                    cols = list(rows_data[0].keys()) if isinstance(rows_data[0], dict) else []
                    rows = [
                        [r.get(c) for c in cols] if isinstance(r, dict) else [r]
                        for r in rows_data
                    ]
                    return {"columns": cols, "rows": rows, "row_count": len(rows)}
                return {"columns": ["result"], "rows": [[str(data)]], "row_count": 1}

            raise RuntimeError(
                "Direct SQL execution via Supabase API requires either:\n"
                "1) A Supabase Personal Access Token (starts with 'sbp_') in the token "
                "field (recommended, enables 100% full SQL execution without passwords).\n"
                "2) Or create the 'exec_sql' RPC helper function in your Supabase SQL Editor:\n"
                "   CREATE OR REPLACE FUNCTION exec_sql(query text) RETURNS json "
                "LANGUAGE plpgsql SECURITY DEFINER AS $$\n"
                "   DECLARE res json; BEGIN EXECUTE 'SELECT json_agg(t) FROM (' || "
                "query || ') t' INTO res; RETURN coalesce(res, '[]'::json); END; $$;"
            )

    def close(self) -> None:

        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
