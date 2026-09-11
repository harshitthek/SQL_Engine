"""
Comprehensive tests for gradio_app business logic, UI state management, and SQL execution.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from api_client import (
    FastAPIUnavailableError,
    InferenceBusyError,
    ModelNotReadyError,
    RequestValidationError,
)
from gradio_app import (
    BANNER_DISMISS_HEAD,
    BANNER_DISMISS_SCRIPT,
    CUSTOM_CSS,
    PERMANENT_GITHUB_HTML,
    REPO_URL,
    TOP_BANNER_HTML,
    _safe_port,
    build_app,
    create_sample_sqlite_db,
    get_initial_state,
    handle_clear,
    handle_clear_credentials,
    handle_connect,
    handle_generate_sql,
    handle_load_sample,
    handle_run_sql,
    on_copy_sql,
    populate_from_client_storage,
    redact_credentials,
    refresh_health,
    switch_db_type,
)


@pytest.fixture
def clean_state():
    return get_initial_state()


@pytest.fixture
def connected_sample_state(tmp_path):
    db_file = str(tmp_path / "test_company.db")
    create_sample_sqlite_db(db_file)
    state = get_initial_state()
    _, _, _, _, _, updated_state = handle_connect(
        db_type="sqlite",
        database=db_file,
        host=None,
        port=None,
        username=None,
        password=None,
        state=state,
    )
    return updated_state


def test_initial_state(clean_state):
    assert clean_state["is_connected"] is False
    assert clean_state["database_name"] == "None"
    assert clean_state["table_count"] == 0
    assert clean_state["schema"] == ""
    assert clean_state["dialect"] is None
    assert len(clean_state["logs"]) > 0


def test_switch_db_type():
    # SQLite
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("SQLite")
    assert u_user["visible"] is False
    assert u_host["visible"] is False
    assert u_port["visible"] is False
    assert u_pw["visible"] is False
    assert "File Path" in u_name["label"]

    # PostgreSQL
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("PostgreSQL")
    assert u_user["visible"] is True
    assert u_port["value"] == 5432
    assert "company_db" in u_name["placeholder"]

    # MySQL
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("MySQL")
    assert u_user["visible"] is True
    assert u_port["value"] == 3306

    # Supabase
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("Supabase")
    assert u_user["visible"] is True
    assert u_user["value"] == "postgres"
    assert u_host["visible"] is True
    assert u_port["visible"] is True
    assert u_port["value"] == 5432
    assert u_pw["visible"] is True
    assert "Database Name" in u_name["label"]
    assert u_name["value"] == "postgres"
    assert "Supabase mode" in hint
    assert "sslmode=require" in hint

    # Supabase (API)
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("Supabase (API)")
    assert u_user["visible"] is False
    assert u_host["visible"] is False
    assert u_port["visible"] is False
    assert u_pw["visible"] is True
    assert ("API Key" in u_pw["placeholder"] or "Token" in u_pw["placeholder"])
    assert ("Project URL" in u_name["label"] or "Ref ID" in u_name["label"])
    assert "Supabase (API) mode" in hint


def test_handle_connect_sqlite_success(clean_state, tmp_path):
    db_file = str(tmp_path / "sample.db")
    create_sample_sqlite_db(db_file)

    st, db_name, tbl_cnt, logs, banner, new_state = handle_connect(
        db_type="sqlite",
        database=db_file,
        host=None,
        port=None,
        username=None,
        password=None,
        state=clean_state,
    )
    assert "Connected to Sqlite" in st
    assert db_name == db_file
    assert "3 tables" in tbl_cnt
    assert new_state["is_connected"] is True
    assert new_state["dialect"] == "sqlite"
    assert "departments" in new_state["table_names"]
    assert "employees" in new_state["table_names"]
    assert "sales" in new_state["table_names"]
    assert "CREATE TABLE employees" in new_state["schema"]
    assert "Connected successfully" in logs


def test_handle_connect_validation_errors(clean_state):
    # Missing SQLite database path
    st, db_n, tc, logs, banner, s = handle_connect(
        db_type="sqlite",
        database="",
        host=None,
        port=None,
        username=None,
        password=None,
        state=clean_state,
    )
    assert s["is_connected"] is False
    assert "Database file path is required" in banner

    # Missing Postgres credentials
    st, db_n, tc, logs, banner, s = handle_connect(
        db_type="postgresql",
        database="mydb",
        host="",
        port=5432,
        username="",
        password="",
        state=clean_state,
    )
    assert s["is_connected"] is False
    assert "required for postgresql" in banner


def test_handle_connect_failure_does_not_log_password(clean_state):
    st, db_n, tc, logs, banner, s = handle_connect(
        db_type="postgresql",
        database="mydb",
        host="localhost",
        port=5432,
        username="admin",
        password="SuperSecretPassword123",
        state=clean_state,
    )
    assert "SuperSecretPassword123" not in logs
    assert "SuperSecretPassword123" not in banner


def test_handle_connect_supabase_api(clean_state, monkeypatch):
    class MockManager:
        is_api_mode = True
        engine = None
        def connect(self):
            return None
        def get_table_names(self):
            return ["users", "orders"]
        def get_schema(self):
            return "CREATE TABLE users (id INT PRIMARY KEY);"

    monkeypatch.setattr("gradio_app.DatabaseManager", lambda cfg: MockManager())

    st, db_n, tc, logs, banner, new_state = handle_connect(
        db_type="Supabase (API)",
        database="https://myproj.supabase.co",
        host=None,
        port=None,
        username=None,
        password="sbp_mocktoken123",
        state=clean_state,
    )

    assert "Connected" in st
    assert "2 tables" in tc
    assert new_state["is_connected"] is True
    assert new_state["dialect"] == "postgresql"
    assert "users" in new_state["table_names"]
    assert "orders" in new_state["table_names"]
    assert "CREATE TABLE users" in new_state["schema"]
    assert "sbp_mocktoken123" not in logs
    assert "sbp_mocktoken123" not in banner


def test_handle_load_sample(clean_state):
    results = handle_load_sample(clean_state)
    db_type_menu, db_name, host, port, user, pw, st, db_n, tc, log_out, banner, state = results
    assert db_type_menu == "SQLite"
    assert "sample_company.db" in db_name
    assert state["is_connected"] is True
    assert state["table_count"] == 3


def test_handle_generate_sql_disconnected(clean_state):
    sql, meta, status, run_btn, state = handle_generate_sql(
        question="Show all employees",
        state=clean_state,
    )
    assert "Database schema unavailable" in status
    assert run_btn["interactive"] is False


def test_handle_generate_sql_success(connected_sample_state):
    with patch("gradio_app.fastapi_client.generate_sql") as mock_gen:
        mock_gen.return_value = {
            "request_id": "uuid-987",
            "sql": "SELECT * FROM employees WHERE salary > 100000;",
            "model": "text2sql-v1",
            "generation_time_ms": 145.2,
        }

        sql, meta, status, run_btn, new_state = handle_generate_sql(
            question="Show high earning employees",
            state=connected_sample_state,
        )

        assert "SELECT * FROM employees" in sql
        assert "text2sql-v1" in meta
        assert "145.2 ms" in meta
        assert "uuid-987" in meta
        assert "SQL generated successfully" in status
        assert run_btn["interactive"] is True
        assert new_state["last_sql"] == "SELECT * FROM employees WHERE salary > 100000;"


def test_handle_generate_sql_service_unavailable(connected_sample_state):
    with patch("gradio_app.fastapi_client.generate_sql") as mock_gen:
        mock_gen.side_effect = FastAPIUnavailableError("Service down")

        sql, meta, status, run_btn, new_state = handle_generate_sql(
            question="Show all employees",
            state=connected_sample_state,
        )

        assert "FastAPI Unavailable" in sql
        assert "unavailable" in status.lower()
        assert run_btn["interactive"] is False


def test_handle_run_sql_safe_select(connected_sample_state):
    query = "SELECT name, salary FROM employees WHERE salary > 120000 ORDER BY salary DESC"
    df_update, info_update, status = handle_run_sql(query, connected_sample_state)

    assert df_update["visible"] is True
    df = df_update["value"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Alice Chen"
    assert "Query executed successfully: 1 row(s) returned" in status


def test_handle_run_sql_blocked_destructive(connected_sample_state):
    dangerous_queries = [
        "DROP TABLE employees",
        "DELETE FROM employees WHERE id = 101",
        "UPDATE employees SET salary = 999999",
        "INSERT INTO departments VALUES (10, 'Hacked', 'Nowhere')",
        "ALTER TABLE employees ADD COLUMN ssn TEXT",
        "SELECT 1; DROP TABLE sales;",
    ]
    for dq in dangerous_queries:
        df_update, info_update, status = handle_run_sql(dq, connected_sample_state)
        assert df_update["visible"] is False
        assert "Safety Block" in info_update["value"] or "Execution rejected" in status


def test_handle_clear():
    q, sql, meta, st, run_btn, df, info = handle_clear()
    assert q == ""
    assert "Generated SQL will appear here" in sql
    assert st == "Ready"
    assert run_btn["interactive"] is False
    assert df["visible"] is False


def test_refresh_health():
    with patch("gradio_app.fastapi_client.health") as mock_h:
        mock_h.return_value = {
            "status": "healthy",
            "model_version": "text2sql-v1",
            "uptime_seconds": 500.0,
            "device": "mps",
            "error": None,
        }
        st_md, det_md = refresh_health()
        assert "Model Healthy" in st_md
        assert "MPS" in det_md
        assert "text2sql-v1" in det_md


def test_build_app():
    demo = build_app()
    assert demo is not None


def test_build_app_tabs_hierarchy_and_navigation():
    demo = build_app()
    tabs_components = [c for c in demo.config["components"] if c.get("type") == "tabs"]
    # Verify there is exactly ONE tabs container (no orphan empty tabs)
    assert len(tabs_components) == 1, f"Expected 1 Tabs component, found {len(tabs_components)}"

    # Verify both child tabs belong to this tabs container
    tab_items = [c for c in demo.config["components"] if c.get("type") == "tabitem"]
    tab_ids = [t["props"].get("id") for t in tab_items]
    assert "workspace" in tab_ids
    assert "set_db" in tab_ids

    # Find navigation dependencies
    # fn0: btn_nav_set_db.click -> switches to set_db
    fn0 = demo.fns[0]
    tabs_comp = fn0.outputs[0]
    assert len(tabs_comp.children) == 2
    res_set_db = fn0.fn()
    assert getattr(res_set_db, "selected", None) == "set_db" or (
        isinstance(res_set_db, dict) and res_set_db.get("selected") == "set_db"
    )

    # fn1: btn_back_to_workspace.click -> switches to workspace
    fn1 = demo.fns[1]
    assert fn1.outputs[0] == tabs_comp
    res_ws = fn1.fn()
    assert getattr(res_ws, "selected", None) == "workspace" or (
        isinstance(res_ws, dict) and res_ws.get("selected") == "workspace"
    )


def test_api_cors_and_routes():
    from api import app as fastapi_app
    from starlette.middleware.cors import CORSMiddleware

    # Check CORS middleware is attached
    has_cors = any(
        m.cls == CORSMiddleware or getattr(m, "cls", None) == CORSMiddleware
        for m in fastapi_app.user_middleware
    )
    assert has_cors is True

    # Check routes include /health, /v1/tosql, and /v1/reload
    route_paths = [r.path for r in fastapi_app.routes]
    assert "/health" in route_paths
    assert "/v1/tosql" in route_paths
    assert "/v1/reload" in route_paths


def test_app_is_backend_healthy():
    from app import is_backend_healthy

    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert is_backend_healthy("http://localhost:8000") is True

        mock_resp.status_code = 503
        assert is_backend_healthy("http://localhost:8000") is False

        mock_get.side_effect = Exception("Connection refused")
        assert is_backend_healthy("http://localhost:8000") is False



def test_on_copy_sql():
    assert "copied to clipboard" in on_copy_sql("SELECT * FROM employees;")
    assert "No SQL query to copy" in on_copy_sql("")
    assert "No SQL query to copy" in on_copy_sql("-- Generated SQL will appear here --")


def test_handle_run_sql_placeholder_query(connected_sample_state):
    # Running SQL on placeholder should show friendly warning, not safety block
    df_update, info_update, status = handle_run_sql("-- Generated SQL will appear here --", connected_sample_state)
    assert df_update["visible"] is False
    assert "No SQL query to execute" in status


def test_handle_run_sql_connection_lost(connected_sample_state):
    # Simulate connection drop
    with patch.object(connected_sample_state["db_manager"], "test_connection", return_value=False):
        df_update, info_update, status = handle_run_sql("SELECT * FROM employees", connected_sample_state)
        assert df_update["visible"] is False
        assert "Database connection lost" in status
        assert connected_sample_state["is_connected"] is False


def test_handle_generate_sql_connection_lost(connected_sample_state):
    with patch.object(connected_sample_state["db_manager"], "test_connection", return_value=False):
        sql, meta, status, run_btn, state = handle_generate_sql("Show all employees", connected_sample_state)
        assert "Database connection lost" in status
        assert state["is_connected"] is False
        assert run_btn["interactive"] is False


def test_handle_connect_scrubs_db_url_password(clean_state):
    # Simulate an error message that contains a DB URL with password
    with patch("gradio_app.DatabaseManager.connect", side_effect=Exception(
        "FATAL: password authentication failed for user 'admin' on postgresql+psycopg://admin:SuperSecretPass@db.internal:5432/mydb"
    )):
        st, db_n, tc, logs, banner, s = handle_connect(
            db_type="postgresql",
            database="mydb",
            host="db.internal",
            port=5432,
            username="admin",
            password="SuperSecretPass",
            state=clean_state,
        )
        assert "SuperSecretPass" not in logs
        assert "SuperSecretPass" not in banner
        assert "••••••" in banner


def test_handle_connect_quoted_sqlite_path(clean_state, tmp_path):
    # Test that paths surrounded with single or double quotes are cleaned properly
    db_file = str(tmp_path / "quoted.db")
    create_sample_sqlite_db(db_file)
    quoted_path = f"'{db_file}'"
    st, db_n, tc, logs, banner, state = handle_connect(
        db_type="sqlite",
        database=quoted_path,
        host=None,
        port=None,
        username=None,
        password=None,
        state=clean_state,
    )
    assert state["is_connected"] is True
    assert db_n == db_file
    assert "Connected to Sqlite" in st


def test_prediction_code_block_extraction():
    # Helper to simulate the cleaning block from generate_sql
    def clean(raw):
        s = raw.strip()
        if "```sql" in s:
            s = s.split("```sql", 1)[1]
            if "```" in s:
                s = s.split("```", 1)[0]
        elif "```" in s:
            s = s.split("```", 1)[1]
            if "```" in s:
                s = s.split("```", 1)[0]
        s = s.strip()
        if ";" in s:
            first_stmt = s.split(";")[0].strip() + ";"
            after_semi = s[s.index(";") + 1:].strip()
            first_word = after_semi.split()[0].upper() if after_semi.split() else ""
            if first_word and first_word not in ("SELECT", "WITH", "EXPLAIN"):
                s = first_stmt
        return s

    r1 = "```sql\nSELECT * FROM employees WHERE salary > 50000;\n```\nThis query filters for employees earning above 50k."
    assert clean(r1) == "SELECT * FROM employees WHERE salary > 50000;"

    r2 = "```\nSELECT * FROM products;\n```"
    assert clean(r2) == "SELECT * FROM products;"

    r3 = "SELECT * FROM orders; Explanation: returns all orders"
    assert clean(r3) == "SELECT * FROM orders;"

    # Unclosed ```sql fence
    r4 = "```sql\nSELECT * FROM invoices;"
    assert clean(r4) == "SELECT * FROM invoices;"

    # Unclosed ``` fence
    r5 = "```\nSELECT * FROM logs;"
    assert clean(r5) == "SELECT * FROM logs;"

    # Query without semicolon
    r6 = "SELECT count(*) FROM items"
    assert clean(r6) == "SELECT count(*) FROM items"

    # Chained query allowed keyword
    r7 = "SELECT 1; SELECT 2;"
    assert "SELECT 1; SELECT 2;" in clean(r7)

    # Semicolon followed by empty whitespace
    r8 = "SELECT 1;   "
    assert clean(r8) == "SELECT 1;"





def test_handle_generate_sql_postgresql_dialect(connected_sample_state):
    connected_sample_state["dialect"] = "postgresql"
    connected_sample_state["db_type"] = "postgresql"

    with patch("gradio_app.fastapi_client.generate_sql") as mock_gen:
        mock_gen.return_value = {
            "request_id": "pg-uuid-1",
            "sql": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';",
            "model": "text2sql-v1",
            "generation_time_ms": 125.0,
        }

        sql, meta, status, run_btn, new_state = handle_generate_sql(
            question="Show names of 14 tables",
            state=connected_sample_state,
        )

        assert "information_schema.tables" in sql
        assert "Dialect: PostgreSQL" in meta
        assert "text2sql-v1" in meta
        assert "125.0 ms" in meta
        assert run_btn["interactive"] is True
        mock_gen.assert_called_once_with(
            question="Show names of 14 tables",
            schema=connected_sample_state["schema"],
            dialect="postgresql",
        )


def test_handle_run_sql_adapts_sqlite_master_for_postgres(connected_sample_state):
    connected_sample_state["dialect"] = "postgresql"
    connected_sample_state["db_type"] = "postgresql"

    # Mock db_manager's execute_query
    mgr = connected_sample_state["db_manager"]
    with patch.object(mgr, "execute_query") as mock_exec:
        mock_exec.return_value = {
            "columns": ["table_name"],
            "rows": [["users"], ["orders"]],
            "row_count": 2,
        }

        query = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name LIMIT 14;'
        df_update, info_update, status = handle_run_sql(query, connected_sample_state)

        assert df_update["visible"] is True
        assert "Query executed successfully: 2 row(s) returned" in status
        executed_query = mock_exec.call_args[0][0]
        assert "information_schema.tables" in executed_query
        assert "table_schema = 'public'" in executed_query
        assert "sqlite_master" not in executed_query
        assert '"table"' not in executed_query


def test_handle_run_sql_adapts_sqlite_master_for_mysql(connected_sample_state):
    connected_sample_state["dialect"] = "mysql"
    connected_sample_state["db_type"] = "mysql"

    mgr = connected_sample_state["db_manager"]
    with patch.object(mgr, "execute_query") as mock_exec:
        mock_exec.return_value = {
            "columns": ["table_name"],
            "rows": [["users"], ["orders"]],
            "row_count": 2,
        }

        query = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name;'
        df_update, info_update, status = handle_run_sql(query, connected_sample_state)

        assert df_update["visible"] is True
        assert "Query executed successfully: 2 row(s) returned" in status
        executed_query = mock_exec.call_args[0][0]
        assert "information_schema.tables" in executed_query
        assert "table_schema = DATABASE()" in executed_query
        assert "table_type = 'BASE TABLE'" in executed_query
        assert "sqlite_master" not in executed_query



def test_prediction_format_prompt_dialects():
    from prediction import Text2SQLEngine
    engine = Text2SQLEngine.__new__(Text2SQLEngine)

    # SQLite
    p_sqlite = engine.format_prompt("Show users", "CREATE TABLE users (id int);", dialect="sqlite")
    assert "write the exact SQLite query that answers the user question." in p_sqlite

    # PostgreSQL
    p_pg = engine.format_prompt("Show users", "CREATE TABLE users (id int);", dialect="postgresql")
    assert "write the exact PostgreSQL query that answers the user question." in p_pg

    # MySQL
    p_mysql = engine.format_prompt("Show users", "CREATE TABLE users (id int);", dialect="mysql")
    assert "write the exact MySQL query that answers the user question." in p_mysql


def test_refresh_health_degraded_and_offline():
    from gradio_app import refresh_health

    # Degraded with error message
    with patch("gradio_app.fastapi_client.health") as mock_health:
        mock_health.return_value = {
            "status": "degraded",
            "device": "none",
            "model_version": "text2sql-v1",
            "error": "Weights download failed due to network timeout",
        }
        status_md, detail_md = refresh_health()
        assert status_md == "○ Model Degraded"
        assert "Degraded (Weights download failed due to..." in detail_md

    # Degraded without error message
    with patch("gradio_app.fastapi_client.health") as mock_health:
        mock_health.return_value = {
            "status": "degraded",
            "device": "none",
            "model_version": "text2sql-v1",
            "error": None,
        }
        status_md, detail_md = refresh_health()
        assert status_md == "○ Model Degraded"
        assert "Degraded (Unknown error...)" in detail_md

    # Offline
    with patch("gradio_app.fastapi_client.health") as mock_health:
        mock_health.return_value = {
            "status": "offline",
            "device": "none",
            "model_version": "unknown",
            "error": "Connection refused",
        }
        status_md, detail_md = refresh_health()
        assert status_md == "✕ FastAPI Offline"
        assert "Target:" in detail_md


def test_switch_db_type_unsupported():
    from gradio_app import switch_db_type
    res = switch_db_type("oracle")
    assert len(res) == 6
    assert res[5] == ""


def test_handle_connect_edge_cases(clean_state, tmp_path):
    from gradio_app import handle_connect

    # 1. Invalid port string -> falls back to None and default port
    st, db_n, tc, logs, banner, state = handle_connect(
        db_type="postgresql",
        database="analytics",
        host="localhost",
        port="not_a_valid_port",
        username="dbuser",
        password="secretpassword",
        state=clean_state,
    )
    assert "Connection error" in banner

    # 2. SQLite file that does not exist -> notice logged in state logs
    new_db = tmp_path / "brand_new.db"
    st, db_n, tc, logs, banner, state = handle_connect(
        db_type="sqlite",
        database=str(new_db),
        host=None,
        port=None,
        username=None,
        password=None,
        state=clean_state,
    )
    assert "Notice: SQLite file" in logs
    assert "does not exist on disk" in logs

    # 3. Empty database name for postgresql / mysql
    st, db_n, tc, logs, banner, state = handle_connect(
        db_type="postgresql",
        database="",
        host="localhost",
        port=5432,
        username="postgres",
        password="pw",
        state=clean_state,
    )
    assert "Database name is required for postgresql" in banner

    # 4. Unsupported database type
    st, db_n, tc, logs, banner, state = handle_connect(
        db_type="mongodb",
        database="nosql_db",
        host="localhost",
        port=27017,
        username="admin",
        password="pw",
        state=clean_state,
    )
    assert "Unsupported database type: mongodb" in banner

    # 5. Exception in urllib.parse.quote_plus during password scrubbing
    with patch("urllib.parse.quote_plus", side_effect=Exception("Quote crash")):
        st, db_n, tc, logs, banner, state = handle_connect(
            db_type="postgresql",
            database="analytics",
            host="localhost",
            port=5432,
            username="dbuser",
            password="secretpassword",
            state=clean_state,
        )
        assert "Connection error" in banner

    # 6. Connection exception with no password (covers line 499->506)
    fresh_state = get_initial_state()
    with patch("gradio_app.DatabaseManager.connect", side_effect=RuntimeError("Cannot open SQLite db")):
        st, db_n, tc, logs, banner, state = handle_connect(
            db_type="sqlite",
            database="/bad/path.db",
            host=None,
            port=None,
            username=None,
            password=None,
            state=fresh_state,
        )
        assert "Cannot open SQLite db" in banner
        assert state["is_connected"] is False




def test_format_conn_status_branches():
    from gradio_app import format_conn_status
    # Connected with db_type
    assert format_conn_status({"is_connected": True, "db_type": "postgresql"}) == "● Connected to Postgresql"
    # Connected without db_type
    assert format_conn_status({"is_connected": True, "db_type": None}) == "● Connected to Sql"
    # Disconnected
    assert format_conn_status({"is_connected": False}) == "○ No database connected"


def test_handle_generate_sql_empty_question_and_error_handlers(connected_sample_state):
    from api_client import (
        InferenceFailedError,
    )
    from gradio_app import handle_generate_sql

    # 1. Empty question
    sql, meta, status, run_btn, state = handle_generate_sql("", connected_sample_state)
    assert "Please enter a question" in status
    assert run_btn["interactive"] is False

    # 2. RequestValidationError
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=RequestValidationError("Query too short")):
        sql, meta, status, run_btn, state = handle_generate_sql("Who?", connected_sample_state)
        assert "Validation Failure" in meta
        assert "Query too short" in status
        assert run_btn["interactive"] is False

    # 3. ModelNotReadyError
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=ModelNotReadyError("Model loading")):
        sql, meta, status, run_btn, state = handle_generate_sql("Valid question?", connected_sample_state)
        assert "Model Not Ready" in meta
        assert "Model is currently unavailable" in status
        assert run_btn["interactive"] is False

    # 4. InferenceBusyError
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=InferenceBusyError("Lock timeout")):
        sql, meta, status, run_btn, state = handle_generate_sql("Valid question?", connected_sample_state)
        assert "Inference Timeout" in meta
        assert "server is busy" in status
        assert run_btn["interactive"] is False

    # 5. InferenceFailedError
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=InferenceFailedError("OOM crash")):
        sql, meta, status, run_btn, state = handle_generate_sql("Valid question?", connected_sample_state)
        assert "Model Inference Error" in meta
        assert "encountered an internal error" in status
        assert run_btn["interactive"] is False

    # 6. RateLimitExceededError
    from api_client import RateLimitExceededError
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=RateLimitExceededError("Too many requests")):
        sql, meta, status, run_btn, state = handle_generate_sql("Valid question?", connected_sample_state)
        assert "Rate Limit Exceeded (429)" in meta
        assert "Rate limit exceeded" in status
        assert run_btn["interactive"] is False

    # 7. Generic Exception
    with patch("gradio_app.fastapi_client.generate_sql", side_effect=RuntimeError("Unexpected glitch")):
        sql, meta, status, run_btn, state = handle_generate_sql("Valid question?", connected_sample_state)
        assert "Metadata: Error" in meta
        assert "Generation failed: Unexpected glitch" in status
        assert run_btn["interactive"] is False


def test_handle_run_sql_disconnected_and_execution_error(connected_sample_state):
    from gradio_app import handle_run_sql
    # Disconnected state
    disconnected_state = {"is_connected": False, "db_manager": None}
    df_up, info_up, status = handle_run_sql("SELECT 1;", disconnected_state)
    assert "Database disconnected" in status

    # Execution error in DatabaseManager.execute_query
    mgr = connected_sample_state["db_manager"]
    with patch.object(mgr, "execute_query", side_effect=RuntimeError("Disk I/O failure")):
        df_up, info_up, status = handle_run_sql("SELECT * FROM employees;", connected_sample_state)
        assert "Disk I/O failure" in status
        assert "Execution failed" in status


def test_get_app_theme():
    from gradio_app import get_app_theme
    theme = get_app_theme()
    assert theme is not None


def test_gradio_launch_and_main(monkeypatch):
    import gradio_app
    mock_demo = MagicMock()
    monkeypatch.setattr("gradio_app.build_app", lambda: mock_demo)

    gradio_app.launch(host="127.0.0.1", port=7860, share=False)
    mock_demo.launch.assert_called_once()
    call_kwargs = mock_demo.launch.call_args.kwargs
    assert call_kwargs.get("head") == gradio_app.BANNER_DISMISS_HEAD
    assert call_kwargs.get("js") == gradio_app.BANNER_DISMISS_SCRIPT


def test_gradio_main_and_no_proxy():
    import os
    import runpy

    import gradio_app

    app_path = os.path.abspath(gradio_app.__file__)

    with patch("gradio.Blocks.launch") as mock_blocks_launch, \
         patch.dict(os.environ, {"NO_PROXY": "10.0.0.1", "no_proxy": "10.0.0.1"}):
        runpy.run_path(app_path, run_name="__main__")
        mock_blocks_launch.assert_called_once()
        assert "127.0.0.1" in os.environ["NO_PROXY"]


def test_switch_db_type_with_saved_profiles():
    # 1. Supabase with saved profile
    saved_sb = json.dumps({
        "supabase": {
            "host": "aws-0.pooler.supabase.com",
            "port": 6543,
            "username": "postgres.ref",
            "password": "mypassword",
            "database": "my_db",
        }
    })
    u_user, u_host, u_port, u_pw, u_name, hint = switch_db_type("Supabase", saved_sb)
    assert u_user["value"] == "postgres.ref"
    assert u_host["value"] == "aws-0.pooler.supabase.com"
    assert u_port["value"] == 6543
    assert u_pw["value"] == "mypassword"
    assert u_name["value"] == "my_db"

    # 2. SQLite with saved profile
    saved_sqlite = json.dumps({"sqlite": {"database": "custom.db"}})
    _, _, _, _, u_name, _ = switch_db_type("SQLite", saved_sqlite)
    assert u_name["value"] == "custom.db"

    # 3. PostgreSQL with saved profile
    saved_pg = json.dumps({
        "postgresql": {
            "host": "pg.corp.local",
            "port": 5433,
            "username": "pguser",
            "password": "pgpassword",
            "database": "pgdb",
        }
    })
    u_user, u_host, u_port, u_pw, u_name, _ = switch_db_type("PostgreSQL", saved_pg)
    assert u_user["value"] == "pguser"
    assert u_host["value"] == "pg.corp.local"
    assert u_port["value"] == 5433
    assert u_pw["value"] == "pgpassword"
    assert u_name["value"] == "pgdb"

    # 4. MySQL with saved profile
    saved_my = json.dumps({
        "mysql": {
            "host": "mysql.corp.local",
            "port": 3307,
            "username": "myuser",
            "password": "mypassword",
            "database": "mydb",
        }
    })
    u_user, u_host, u_port, u_pw, u_name, _ = switch_db_type("MySQL", saved_my)
    assert u_user["value"] == "myuser"
    assert u_host["value"] == "mysql.corp.local"
    assert u_port["value"] == 3307
    assert u_pw["value"] == "mypassword"
    assert u_name["value"] == "mydb"

    # 5. Malformed JSON string fallback
    res = switch_db_type("Supabase", "{invalid json")
    assert res[0]["value"] == "postgres"
    assert res[4]["value"] == "postgres"

    # 6. JSON that parses to list or contains non-dict value
    res_list = switch_db_type("Supabase", json.dumps([1, 2, 3]))
    assert res_list[0]["value"] == "postgres"
    res_non_dict_val = switch_db_type("Supabase", json.dumps({"supabase": "string_not_dict", "other": {"k": "v"}}))
    assert res_non_dict_val[0]["value"] == "postgres"


def test_populate_from_client_storage():
    # 1. Supabase profile
    sb_payload = json.dumps({
        "supabase": {
            "host": "aws-0.pooler.supabase.com",
            "port": 5432,
            "username": "postgres.team",
            "password": "securepassword",
            "database": "postgres",
        }
    })
    u, h, p, pw, db, bridge = populate_from_client_storage(sb_payload, "Supabase")
    assert u["value"] == "postgres.team"
    assert h["value"] == "aws-0.pooler.supabase.com"
    assert p["value"] == 5432
    assert pw["value"] == "securepassword"
    assert db["value"] == "postgres"
    assert bridge == sb_payload

    # 2. SQLite profile
    sqlite_payload = json.dumps({"sqlite": {"database": "app_data.db"}})
    u, h, p, pw, db, bridge = populate_from_client_storage(sqlite_payload, "SQLite")
    assert db["value"] == "app_data.db"
    assert p["value"] is None

    # 3. PostgreSQL profile
    pg_payload = json.dumps({"postgresql": {"host": "pg.local", "port": 5432, "username": "admin", "password": "pw", "database": "orders"}})
    u, h, p, pw, db, bridge = populate_from_client_storage(pg_payload, "PostgreSQL")
    assert u["value"] == "admin"
    assert h["value"] == "pg.local"
    assert db["value"] == "orders"

    # 4. MySQL profile
    my_payload = json.dumps({"mysql": {"host": "my.local", "port": 3306, "username": "root", "password": "pw", "database": "store"}})
    u, h, p, pw, db, bridge = populate_from_client_storage(my_payload, "MySQL")
    assert u["value"] == "root"
    assert h["value"] == "my.local"
    assert db["value"] == "store"

    # 5. Empty and invalid payloads fallback
    u, h, p, pw, db, bridge = populate_from_client_storage("{}", "Supabase")
    assert u["value"] == "postgres"
    assert db["value"] == "postgres"
    assert p["value"] == 5432

    u, h, p, pw, db, bridge = populate_from_client_storage("{}", "SQLite")
    assert "sample_company.db" in db["value"]

    u, h, p, pw, db, bridge = populate_from_client_storage("not a json", "Supabase")
    assert u["value"] == "postgres"

    # 6. Unsupported DB type
    res = populate_from_client_storage("{}", "redis")
    assert len(res) == 6

    # 7. List payload, non-dict value in dict payload, and non-string payload
    res_list = populate_from_client_storage(json.dumps([1, 2]), "Supabase")
    assert res_list[0]["value"] == "postgres"
    res_non_dict = populate_from_client_storage(json.dumps({"supabase": "bad", 123: "val"}), "Supabase")
    assert res_non_dict[0]["value"] == "postgres"
    res_non_str = populate_from_client_storage(12345, "Supabase")
    assert res_non_str[0]["value"] == "postgres"


def test_handle_clear_credentials():
    # SQLite
    u, h, p, pw, db, banner, bridge = handle_clear_credentials("SQLite")
    assert "sample_company.db" in db["value"]
    assert p["value"] is None
    assert pw["value"] == ""
    assert bridge == "{}"
    assert "purged" in banner

    # Supabase
    u, h, p, pw, db, banner, bridge = handle_clear_credentials("Supabase")
    assert u["value"] == "postgres"
    assert db["value"] == "postgres"
    assert p["value"] == 5432
    assert pw["value"] == ""
    assert bridge == "{}"

    # PostgreSQL
    u, h, p, pw, db, banner, bridge = handle_clear_credentials("PostgreSQL")
    assert p["value"] == 5432
    assert db["value"] == ""
    assert bridge == "{}"

    # MySQL
    u, h, p, pw, db, banner, bridge = handle_clear_credentials("MySQL")
    assert p["value"] == 3306
    assert db["value"] == ""
    assert bridge == "{}"


def test_handle_connect_supabase_success(clean_state):
    with patch("gradio_app.DatabaseManager.connect"), \
         patch("gradio_app.DatabaseManager.get_table_names", return_value=["users", "profiles"]), \
         patch("gradio_app.DatabaseManager.get_schema", return_value="CREATE TABLE users (id int);"):
        st, db_n, tc, logs, banner, new_state = handle_connect(
            db_type="Supabase",
            database="postgres",
            host="aws-0-us-east-1.pooler.supabase.com",
            port=5432,
            username="postgres.myref",
            password="securepassword",
            state=clean_state,
        )
        assert "Connected to Supabase" in st
        assert db_n == "postgres"
        assert "2 tables" in tc
        assert new_state["is_connected"] is True
        assert new_state["db_type"] == "supabase"
        assert new_state["database_name"] == "postgres"
        assert "Connected successfully to SUPABASE" in logs
        assert "✓ Connected to Supabase (postgres)" in banner


def test_handle_connect_supabase_defaults_and_validation(clean_state):
    # Missing host
    st, db_n, tc, logs, banner, s = handle_connect(
        db_type="supabase",
        database="postgres",
        host="",
        port=5432,
        username="postgres",
        password="secretpassword",
        state=clean_state,
    )
    assert s["is_connected"] is False
    assert "Host and password are required for supabase" in banner

    # Missing password
    st, db_n, tc, logs, banner, s = handle_connect(
        db_type="supabase",
        database="postgres",
        host="db.supabase.co",
        port=5432,
        username="postgres",
        password="",
        state=clean_state,
    )
    assert s["is_connected"] is False
    assert "Host and password are required for supabase" in banner

    # Defaults applied when database, port, and username are empty/omitted
    with patch("gradio_app.DatabaseManager.connect"), \
         patch("gradio_app.DatabaseManager.get_table_names", return_value=["products"]), \
         patch("gradio_app.DatabaseManager.get_schema", return_value="CREATE TABLE products (id int);"):
        st, db_n, tc, logs, banner, s = handle_connect(
            db_type="supabase",
            database="",
            host="db.supabase.co",
            port=None,
            username="",
            password="secretpassword",
            state=clean_state,
        )
        assert s["is_connected"] is True
        assert s["database_name"] == "postgres"
        assert s["config"].port == 5432
        assert s["config"].username == "postgres"
        assert s["config"].database == "postgres"


def test_handle_connect_supabase_failure_redacts_password(clean_state):
    with patch("gradio_app.DatabaseManager.connect", side_effect=Exception(
        "FATAL: password authentication failed for user 'postgres' on postgresql://postgres:SupabaseSuperSecretPass123@aws-0.pooler.supabase.com:5432/postgres?sslmode=require, password='SupabaseSuperSecretPass123'"
    )):
        st, db_n, tc, logs, banner, s = handle_connect(
            db_type="supabase",
            database="postgres",
            host="aws-0.pooler.supabase.com",
            port=5432,
            username="postgres",
            password="SupabaseSuperSecretPass123",
            state=clean_state,
        )
        assert "SupabaseSuperSecretPass123" not in logs
        assert "SupabaseSuperSecretPass123" not in banner
        assert "••••••" in banner


def test_handle_generate_sql_supabase(connected_sample_state):
    connected_sample_state["dialect"] = "supabase"
    connected_sample_state["db_type"] = "supabase"

    with patch("gradio_app.fastapi_client.generate_sql") as mock_gen:
        mock_gen.return_value = {
            "request_id": "sb-uuid-1",
            "sql": "SELECT * FROM profiles WHERE active = true;",
            "model": "text2sql-v1",
            "generation_time_ms": 110.5,
        }

        sql, meta, status, run_btn, new_state = handle_generate_sql(
            question="Show active profiles",
            state=connected_sample_state,
        )

        assert "SELECT * FROM profiles" in sql
        assert "Dialect: PostgreSQL" in meta
        assert "text2sql-v1" in meta
        assert "110.5 ms" in meta
        assert run_btn["interactive"] is True
        mock_gen.assert_called_once_with(
            question="Show active profiles",
            schema=connected_sample_state["schema"],
            dialect="supabase",
        )


def test_handle_run_sql_supabase_adapts_query(connected_sample_state):
    connected_sample_state["dialect"] = "supabase"
    connected_sample_state["db_type"] = "supabase"

    mgr = connected_sample_state["db_manager"]
    with patch.object(mgr, "execute_query") as mock_exec:
        mock_exec.return_value = {
            "columns": ["table_name"],
            "rows": [["profiles"], ["teams"]],
            "row_count": 2,
        }

        query = 'SELECT name FROM sqlite_master WHERE type = "table";'
        df_update, info_update, status = handle_run_sql(query, connected_sample_state)

        assert df_update["visible"] is True
        assert "Query executed successfully: 2 row(s) returned" in status
        executed_query = mock_exec.call_args[0][0]
        assert "information_schema.tables" in executed_query
        assert "table_schema = 'public'" in executed_query
        assert "sqlite_master" not in executed_query


def test_build_app_client_storage_components():
    demo = build_app()
    component_types = [c.get("type") for c in demo.config["components"]]
    assert "textbox" in component_types

    # Find client_storage_bridge
    bridge_comp = next((c for c in demo.config["components"] if c.get("props", {}).get("elem_id") == "client_storage_bridge"), None)
    assert bridge_comp is not None
    assert bridge_comp["props"]["visible"] is False

    # Find db_type_menu
    menu_comp = next((c for c in demo.config["components"] if c.get("props", {}).get("label") == "DB type(menu)"), None)
    assert menu_comp is not None
    choices = [c[0] if isinstance(c, (tuple, list)) else c for c in menu_comp["props"]["choices"]]
    assert "Supabase" in choices

    # Find clear creds button
    clear_btn_comp = next((c for c in demo.config["components"] if "Clear Saved Credentials" in str(c.get("props", {}).get("value", ""))), None)
    assert clear_btn_comp is not None


def test_safe_port():
    assert _safe_port(None, 5432) == 5432
    assert _safe_port(5432, 5432) == 5432
    assert _safe_port("5432", 5432) == 5432
    assert _safe_port("  3306  ", 5432) == 3306
    assert _safe_port("", 5432) == 5432
    assert _safe_port("not_a_number", 5432) == 5432
    assert _safe_port([], 5432) == 5432
    assert _safe_port(None, None) is None


def test_redact_credentials():
    # Null and non-string inputs
    assert redact_credentials(None) == ""
    assert redact_credentials(12345) == "12345"
    assert redact_credentials("") == ""

    # Normal text without credentials
    assert redact_credentials("No secrets here.") == "No secrets here."

    # URI password
    uri_text = "Failed connecting to postgresql://postgres:SuperSecret123@aws-0.supabase.com:5432/postgres"
    redacted = redact_credentials(uri_text)
    assert "SuperSecret123" not in redacted
    assert "postgresql://postgres:••••••@aws-0.supabase.com:5432/postgres" in redacted

    # Key-value assignments (unquoted and quoted)
    for kw in ["password", "pwd", "pass", "passwd"]:
        raw = f"Error: host=localhost {kw}=my_secret_pw dbname=test"
        res = redact_credentials(raw)
        assert "my_secret_pw" not in res
        assert "••••••" in res

    quoted = 'host=localhost password="secret in quotes" db=db'
    assert "secret in quotes" not in redact_credentials(quoted)

    # JSON style password
    json_err = '{"status": "error", "password": "hidden_secret"}'
    assert "hidden_secret" not in redact_credentials(json_err)
    assert '"password": "••••••"' in redact_credentials(json_err)

    # Explicit password and quote_plus replacement
    explicit = "Login error for secret@123 with quote_plus secret%40123"
    res_exp = redact_credentials(explicit, password="secret@123")
    assert "secret@123" not in res_exp
    assert "secret%40123" not in res_exp


def test_populate_and_switch_malformed_port_and_nulls():
    # Malformed port in profile does not crash populate_from_client_storage
    bad_port_payload = json.dumps({
        "supabase": {
            "host": None,
            "port": "invalid_port",
            "username": None,
            "password": None,
            "database": None,
        },
        "postgresql": {
            "host": None,
            "port": "abc",
            "username": None,
            "password": None,
            "database": None,
        },
        "mysql": {
            "host": None,
            "port": "xyz",
            "username": None,
            "password": None,
            "database": None,
        },
    })

    # Supabase defaults
    u, h, p, pw, db, _ = populate_from_client_storage(bad_port_payload, "Supabase")
    assert u["value"] == "postgres"
    assert h["value"] == ""
    assert p["value"] == 5432
    assert pw["value"] == ""
    assert db["value"] == "postgres"

    # PostgreSQL defaults
    u_pg, h_pg, p_pg, pw_pg, db_pg, _ = populate_from_client_storage(bad_port_payload, "PostgreSQL")
    assert u_pg["value"] == ""
    assert h_pg["value"] == ""
    assert p_pg["value"] == 5432

    # MySQL defaults
    u_my, h_my, p_my, pw_my, db_my, _ = populate_from_client_storage(bad_port_payload, "MySQL")
    assert u_my["value"] == ""
    assert p_my["value"] == 3306

    # Malformed port in profile does not crash switch_db_type
    sw_u, sw_h, sw_p, sw_pw, sw_db, _ = switch_db_type("Supabase", bad_port_payload)
    assert sw_u["value"] == "postgres"
    assert sw_p["value"] == 5432
    assert sw_db["value"] == "postgres"

    sw_u_pg, _, sw_p_pg, _, _, _ = switch_db_type("PostgreSQL", bad_port_payload)
    assert sw_p_pg["value"] == 5432

    sw_u_my, _, sw_p_my, _, _, _ = switch_db_type("MySQL", bad_port_payload)
    assert sw_p_my["value"] == 3306


def test_handle_connect_redacts_unquoted_passwords_and_uris(clean_state):
    with patch("gradio_app.DatabaseManager") as mock_dm_cls:
        mock_dm = MagicMock()
        mock_dm.connect.side_effect = Exception("connection failure password=UltraSecret host=db.supabase.co")
        mock_dm_cls.return_value = mock_dm

        _, _, _, logs, banner, _ = handle_connect(
            db_type="supabase",
            database="postgres",
            host="db.supabase.co",
            port=5432,
            username="postgres",
            password="UltraSecret",
            state=clean_state,
        )

        assert "UltraSecret" not in logs
        assert "UltraSecret" not in banner
        assert "••••••" in banner


def test_handle_run_sql_redacts_sensitive_error(connected_sample_state):
    mgr = connected_sample_state["db_manager"]
    with patch.object(mgr, "execute_query") as mock_exec:
        mock_exec.side_effect = Exception("connection to postgresql://user:MySecretPassword@localhost:5432/db failed")

        _, info, status = handle_run_sql("SELECT 1;", connected_sample_state)
        assert "MySecretPassword" not in info["value"]
        assert "MySecretPassword" not in status
        assert "postgresql://user:••••••@localhost:5432/db" in info["value"]


def test_top_banner_content_and_structure():
    """Verify dismissible top banner text, repo link in new tab, and cross dismiss button."""
    assert REPO_URL == "https://github.com/here-2007/SQL_Engine"
    assert "You can run it locally for even Better Experience" in TOP_BANNER_HTML
    assert "Github" in TOP_BANNER_HTML
    assert f'href="{REPO_URL}"' in TOP_BANNER_HTML
    assert 'target="_blank"' in TOP_BANNER_HTML
    assert 'rel="noopener noreferrer"' in TOP_BANNER_HTML
    # Cross button '✕' for client-side JavaScript dismissal
    assert "✕" in TOP_BANNER_HTML
    assert 'type="button"' in TOP_BANNER_HTML
    assert "document.getElementById('top-announcement-banner').style.display='none';" in TOP_BANNER_HTML
    assert "top_announcement_banner_wrapper" in TOP_BANNER_HTML
    assert ".terminal-banner, .banner-wrapper" in TOP_BANNER_HTML
    assert "sessionStorage.setItem('dismiss_local_run_banner', '1')" in TOP_BANNER_HTML
    assert 'id="top-announcement-banner"' in TOP_BANNER_HTML
    assert 'id="banner-dismiss-btn"' in TOP_BANNER_HTML
    assert "terminal-banner" in TOP_BANNER_HTML


def test_permanent_github_logo():
    """Verify permanent GitHub logo redirects to repo in new tab with SVG icon."""
    assert f'href="{REPO_URL}"' in PERMANENT_GITHUB_HTML
    assert 'target="_blank"' in PERMANENT_GITHUB_HTML
    assert 'rel="noopener noreferrer"' in PERMANENT_GITHUB_HTML
    assert "<svg" in PERMANENT_GITHUB_HTML
    assert "github-icon" in PERMANENT_GITHUB_HTML
    assert 'id="permanent-github-link"' in PERMANENT_GITHUB_HTML
    assert 'aria-label="GitHub Repository"' in PERMANENT_GITHUB_HTML


def test_build_app_top_banner_and_permanent_github_logo():
    """Verify build_app renders top banner and permanent GitHub logo components."""
    demo = build_app()
    components = demo.config["components"]

    # 1. Permanent GitHub Logo component
    github_comp = next(
        (c for c in components if c.get("props", {}).get("elem_id") == "permanent_github_logo"),
        None,
    )
    assert github_comp is not None, "permanent_github_logo component not found in build_app"
    assert github_comp["type"] == "html"
    assert github_comp["props"]["visible"] is True
    assert REPO_URL in github_comp["props"]["value"]
    assert "<svg" in github_comp["props"]["value"]

    # 2. Top Announcement Banner component
    banner_comp = next(
        (c for c in components if c.get("props", {}).get("elem_id") == "top_announcement_banner_wrapper"),
        None,
    )
    assert banner_comp is not None, "top_announcement_banner_wrapper component not found in build_app"
    assert banner_comp["type"] == "html"
    assert banner_comp["props"]["visible"] is True
    banner_val = banner_comp["props"]["value"]
    assert "You can run it locally for even Better Experience" in banner_val
    assert "Github" in banner_val
    assert REPO_URL in banner_val
    assert "✕" in banner_val
    assert "style.display='none'" in banner_val

    # 3. Verify tab switching event listeners remain intact
    fn0 = demo.fns[0]
    res_set_db = fn0.fn()
    assert getattr(res_set_db, "selected", None) == "set_db" or (
        isinstance(res_set_db, dict) and res_set_db.get("selected") == "set_db"
    )

    fn1 = demo.fns[1]
    res_ws = fn1.fn()
    assert getattr(res_ws, "selected", None) == "workspace" or (
        isinstance(res_ws, dict) and res_ws.get("selected") == "workspace"
    )


def test_custom_css_banner_and_github_logo_styling():
    """Verify CUSTOM_CSS styles top banner, repo link, dismiss button, and permanent GitHub logo."""
    # Permanent GitHub logo fixed top-right styling
    assert "#permanent_github_logo" in CUSTOM_CSS
    assert ".permanent-github-container" in CUSTOM_CSS
    assert "position: fixed" in CUSTOM_CSS
    assert "top: 14px" in CUSTOM_CSS
    assert "right: 18px" in CUSTOM_CSS
    assert ".permanent-github-link" in CUSTOM_CSS
    assert ".permanent-github-link:hover" in CUSTOM_CSS

    # Top announcement banner terminal styling
    assert ".terminal-banner" in CUSTOM_CSS
    assert "#top-announcement-banner" in CUSTOM_CSS
    assert "border-radius: 10px" in CUSTOM_CSS
    assert ".banner-repo-link" in CUSTOM_CSS
    assert ".banner-repo-link:hover" in CUSTOM_CSS

    # Cross button styling
    assert ".banner-close-btn" in CUSTOM_CSS
    assert "#banner-dismiss-btn" in CUSTOM_CSS
    assert "cursor: pointer" in CUSTOM_CSS


def test_banner_html_parser_and_text_extraction():
    """Verify banner HTML parsing, link attributes, text, and dismiss button with HTMLParser."""
    from html.parser import HTMLParser

    class BannerParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self.links = []
            self.buttons = []
            self.in_anchor = False
            self.in_button = False
            self.cur_anchor = {}
            self.cur_button = {}

        def handle_starttag(self, tag, attrs):
            attr_dict = dict(attrs)
            if tag == "a":
                self.in_anchor = True
                self.cur_anchor = {"attrs": attr_dict, "text": ""}
            elif tag == "button":
                self.in_button = True
                self.cur_button = {"attrs": attr_dict, "text": ""}

        def handle_endtag(self, tag):
            if tag == "a" and self.in_anchor:
                self.links.append(self.cur_anchor)
                self.in_anchor = False
            elif tag == "button" and self.in_button:
                self.buttons.append(self.cur_button)
                self.in_button = False

        def handle_data(self, data):
            clean = data.strip()
            if clean:
                self.texts.append(clean)
            if self.in_anchor:
                self.cur_anchor["text"] += data
            if self.in_button:
                self.cur_button["text"] += data

    parser = BannerParser()
    parser.feed(TOP_BANNER_HTML)

    # 1. Check extracted links
    assert len(parser.links) == 1
    anchor = parser.links[0]
    assert anchor["attrs"].get("href") == "https://github.com/here-2007/SQL_Engine"
    assert anchor["attrs"].get("target") == "_blank"
    assert "noopener" in anchor["attrs"].get("rel", "")
    assert anchor["text"].strip() == "Github"

    # 2. Check dismiss button
    assert len(parser.buttons) == 1
    btn = parser.buttons[0]
    assert btn["text"].strip() == "✕"
    assert btn["attrs"].get("type") == "button"
    assert "style.display='none'" in btn["attrs"].get("onclick", "")

    # 3. Check combined displayed message
    full_text = " ".join(parser.texts)
    assert "You can run it locally for even Better Experience" in full_text
    assert "Github" in full_text
    assert "✕" in full_text


def test_banner_dismiss_listener_and_pointer_events():
    """Verify demo.load contains the dismiss listener and CUSTOM_CSS contains pointer-events rules."""
    # 1. Verify CUSTOM_CSS pointer-events and button stacking rules
    assert "pointer-events: none !important;" in CUSTOM_CSS
    assert "pointer-events: auto !important;" in CUSTOM_CSS
    assert "position: relative !important;" in CUSTOM_CSS
    assert "z-index: 101 !important;" in CUSTOM_CSS
    assert "cursor: pointer !important;" in CUSTOM_CSS

    # Verify selectors in CUSTOM_CSS
    assert "#permanent_github_logo" in CUSTOM_CSS
    assert ".permanent-github-container" in CUSTOM_CSS
    assert ".permanent-github-link" in CUSTOM_CSS
    assert "#banner-dismiss-btn" in CUSTOM_CSS
    assert ".banner-close-btn" in CUSTOM_CSS

    # 2. Verify build_app attaches head script to top_banner
    demo = build_app()
    components = demo.config["components"]
    banner_comp = next(
        (c for c in components if c.get("props", {}).get("elem_id") == "top_announcement_banner_wrapper"),
        None,
    )
    assert banner_comp is not None, "top_announcement_banner_wrapper not found"
    head_script = banner_comp["props"].get("head", "")
    assert "<script>" in head_script
    assert "addEventListener('click'" in head_script or 'addEventListener("click"' in head_script
    assert "#banner-dismiss-btn" in head_script
    assert ".banner-close-btn" in head_script
    assert "dismiss_local_run_banner" in head_script
    assert "sessionStorage" in head_script

    # 3. Verify demo.load contains capturing dismiss listener and sessionStorage check
    load_dep = next(
        (d for d in demo.config["dependencies"] if any(event == "load" for _, event in d.get("targets", []))),
        None,
    )
    assert load_dep is not None, "load dependency not found in demo.config"
    load_js = load_dep.get("js", "")
    assert "addEventListener('click'" in load_js
    assert "#banner-dismiss-btn" in load_js
    assert ".banner-close-btn" in load_js
    assert "sessionStorage.getItem('dismiss_local_run_banner')" in load_js
    assert "top-announcement-banner" in load_js
    assert "top_announcement_banner_wrapper" in load_js
    assert ".terminal-banner, .banner-wrapper" in load_js

    # 4. Verify BANNER_DISMISS_SCRIPT and BANNER_DISMISS_HEAD constants
    assert "dismissTopBanner" in BANNER_DISMISS_SCRIPT
    assert "initBannerDismiss" in BANNER_DISMISS_SCRIPT
    assert BANNER_DISMISS_HEAD.startswith("<script>")
    assert BANNER_DISMISS_HEAD.endswith("</script>")


def test_dynamic_repo_url_configuration(monkeypatch):
    """Verify get_repo_url and banner HTML dynamically reflect REPO_URL environment variable."""
    from deployment.gradio_app import (
        get_permanent_github_html,
        get_repo_url,
        get_top_banner_html,
    )

    # Default fallback to upstream friend repo
    monkeypatch.delenv("REPO_URL", raising=False)
    assert get_repo_url() == "https://github.com/here-2007/SQL_Engine"
    assert "https://github.com/here-2007/SQL_Engine" in get_top_banner_html()
    assert "https://github.com/here-2007/SQL_Engine" in get_permanent_github_html()

    # Dynamic custom override
    custom_repo = "https://github.com/custom-user/Custom_SQL_Engine"
    monkeypatch.setenv("REPO_URL", custom_repo)
    assert get_repo_url() == custom_repo
    assert custom_repo in get_top_banner_html()
    assert custom_repo in get_permanent_github_html()








