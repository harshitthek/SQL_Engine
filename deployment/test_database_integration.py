"""
Unit and integration tests for DatabaseManager and DatabaseConfig.
"""
import os
import sqlite3
import pytest

from database import DatabaseConfig, DatabaseManager


@pytest.fixture
def sqlite_test_db(tmp_path):
    db_file = tmp_path / "test_store.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price REAL,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories (id)
        );
    """)
    cur.execute("INSERT INTO categories VALUES (1, 'Electronics'), (2, 'Books');")
    cur.execute("INSERT INTO products VALUES (101, 'Laptop', 999.99, 1), (102, 'Novel', 14.50, 2);")
    conn.commit()
    conn.close()
    return str(db_file)


def test_sqlite_connect_and_tables(sqlite_test_db):
    config = DatabaseConfig(db_type="sqlite", database=sqlite_test_db)
    manager = DatabaseManager(config)
    assert manager.test_connection() is True
    tables = manager.get_table_names()
    assert "categories" in tables
    assert "products" in tables
    manager.close()


def test_schema_introspection(sqlite_test_db):
    config = DatabaseConfig(db_type="sqlite", database=sqlite_test_db)
    manager = DatabaseManager(config)
    schema = manager.get_schema()
    assert "CREATE TABLE categories" in schema
    assert "CREATE TABLE products" in schema
    assert "price" in schema
    assert "FOREIGN KEY" in schema
    manager.close()


def test_safe_query_execution(sqlite_test_db):
    config = DatabaseConfig(db_type="sqlite", database=sqlite_test_db)
    manager = DatabaseManager(config)
    result = manager.execute_query("SELECT id, title, price FROM products ORDER BY price DESC")
    assert result["columns"] == ["id", "title", "price"]
    assert result["row_count"] == 2
    assert result["rows"][0][1] == "Laptop"
    manager.close()


def test_query_validation_safe_queries():
    # Valid SELECT, WITH, EXPLAIN
    DatabaseManager.validate_sql("SELECT * FROM users")
    DatabaseManager.validate_sql("WITH cte AS (SELECT 1 AS val) SELECT * FROM cte")
    DatabaseManager.validate_sql("EXPLAIN QUERY PLAN SELECT * FROM users")


def test_query_validation_blocked_statements():
    blocked_cases = [
        "DROP TABLE users",
        "DELETE FROM users WHERE id = 1",
        "INSERT INTO users VALUES (1, 'test')",
        "UPDATE users SET name = 'hacked'",
        "ALTER TABLE users ADD COLUMN age INT",
        "CREATE TABLE rogue (id INT)",
        "TRUNCATE TABLE users",
        "ATTACH DATABASE 'evil.db' AS evil",
        "PRAGMA table_info(users)",
    ]
    for sql in blocked_cases:
        with pytest.raises(ValueError, match="(Blocked SQL operation|Only SELECT, WITH)"):
            DatabaseManager.validate_sql(sql)


def test_query_validation_multiple_statements():
    with pytest.raises(ValueError, match="Multiple SQL statements are not allowed"):
        DatabaseManager.validate_sql("SELECT 1; DROP TABLE users;")


def test_query_validation_empty():
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        DatabaseManager.validate_sql("   ")


def test_invalid_database_connection():
    config = DatabaseConfig(
        db_type="postgresql",
        database="nonexistent_db",
        host="127.0.0.1",
        port=59999,
        username="fake_user",
        password="fake_password",
    )
    manager = DatabaseManager(config)
    assert manager.test_connection() is False
    with pytest.raises(ConnectionError):
        manager.connect()


def test_query_validation_column_names_with_keywords():
    # Columns like created_at, updated_at, attachment should NOT be blocked
    DatabaseManager.validate_sql("SELECT id, created_at, updated_at FROM orders")
    DatabaseManager.validate_sql("SELECT attachment_url, drop_off_time FROM deliveries")
    DatabaseManager.validate_sql("SELECT grant_amount, grant_date FROM research_grants")


def test_query_validation_with_comments():
    # SQL queries starting with comments should still be recognized as SELECT queries
    sql_with_comment = "-- Get all active customers\nSELECT * FROM customers WHERE active = 1"
    DatabaseManager.validate_sql(sql_with_comment)
    sql_with_block_comment = "/* Multi-line comment\nabout query */\nSELECT id FROM users"
    DatabaseManager.validate_sql(sql_with_block_comment)


def test_dialect_and_liveness_check(sqlite_test_db):
    config = DatabaseConfig(db_type="sqlite", database=sqlite_test_db)
    manager = DatabaseManager(config)
    manager.connect()
    assert manager.engine.dialect.name == "sqlite"
    assert manager.test_connection() is True
    manager.close()


def test_query_validation_string_literals_with_semicolons_and_keywords():
    # Semicolons and blocked keywords inside string literals must not trigger false positives
    DatabaseManager.validate_sql("SELECT * FROM messages WHERE text = 'hello; world'")
    DatabaseManager.validate_sql("SELECT * FROM audit_log WHERE change_type = 'update'")
    DatabaseManager.validate_sql("SELECT * FROM operations WHERE action = 'delete' AND note = 'drop;'")
    DatabaseManager.validate_sql("SELECT id FROM users WHERE status = 'DROP'")
    DatabaseManager.validate_sql('SELECT * FROM items WHERE description = "insert into database;"')


def test_adapt_sql_dialect_postgres_user_error_case():
    from database import adapt_sql_dialect
    sql = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name LIMIT 14;'
    adapted = adapt_sql_dialect(sql, dialect="postgresql")
    assert "sqlite_master" not in adapted
    assert '"table"' not in adapted
    assert "information_schema.tables" in adapted
    assert "table_schema = 'public'" in adapted
    assert "table_type = 'BASE TABLE'" in adapted
    assert "ORDER BY table_name LIMIT 14" in adapted
    assert adapted.startswith("SELECT table_name FROM information_schema.tables")


def test_adapt_sql_dialect_postgres_patterns():
    from database import adapt_sql_dialect
    # Query without WHERE clause
    adapted1 = adapt_sql_dialect("SELECT name FROM sqlite_master;", dialect="postgresql")
    assert adapted1 == "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"

    # Query with sqlite_schema
    adapted2 = adapt_sql_dialect('SELECT name FROM sqlite_schema WHERE type = "table" LIMIT 5;', dialect="postgresql")
    assert "information_schema.tables" in adapted2
    assert "table_schema = 'public'" in adapted2
    assert "table_type = 'BASE TABLE'" in adapted2

    # Query with tbl_name
    adapted3 = adapt_sql_dialect("SELECT tbl_name FROM sqlite_master WHERE type = 'table';", dialect="postgresql")
    assert "table_name" in adapted3
    assert "tbl_name" not in adapted3

    # View type mapping
    adapted4 = adapt_sql_dialect("SELECT name FROM sqlite_master WHERE type = 'view';", dialect="postgresql")
    assert "table_type = 'VIEW'" in adapted4


def test_adapt_sql_dialect_postgres_functions_and_quotes():
    from database import adapt_sql_dialect
    # IFNULL -> COALESCE
    sql1 = 'SELECT IFNULL(bonus, 0) FROM employees WHERE status = "active";'
    adapted1 = adapt_sql_dialect(sql1, dialect="postgresql")
    assert "COALESCE(bonus, 0)" in adapted1
    assert "IFNULL" not in adapted1
    assert "status = 'active'" in adapted1

    # strftime to TO_CHAR
    sql2 = "SELECT * FROM orders WHERE strftime('%Y', order_date) = '2024';"
    adapted2 = adapt_sql_dialect(sql2, dialect="postgresql")
    assert "TO_CHAR(order_date, 'YYYY') = '2024'" in adapted2

    # strftime to EXTRACT when compared to integer
    sql3 = "SELECT * FROM orders WHERE strftime('%Y', order_date) = 2024;"
    adapted3 = adapt_sql_dialect(sql3, dialect="postgresql")
    assert "EXTRACT(YEAR FROM order_date) = 2024" in adapted3

    # strftime with month
    sql4 = "SELECT * FROM orders WHERE strftime('%Y-%m', order_date) = '2024-05';"
    adapted4 = adapt_sql_dialect(sql4, dialect="postgresql")
    assert "TO_CHAR(order_date, 'YYYY-MM') = '2024-05'" in adapted4


def test_adapt_sql_dialect_sqlite_and_mysql():
    from database import adapt_sql_dialect
    # SQLite should retain sqlite_master
    sql = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name LIMIT 14;'
    sqlite_adapted = adapt_sql_dialect(sql, dialect="sqlite")
    assert "sqlite_master" in sqlite_adapted

    # MySQL adapts to information_schema.tables with DATABASE()
    mysql_adapted = adapt_sql_dialect(sql, dialect="mysql")
    assert "information_schema.tables" in mysql_adapted
    assert "DATABASE()" in mysql_adapted
    assert "table_type = 'BASE TABLE'" in mysql_adapted


def test_database_manager_execute_query_postgres_adaptation():
    from unittest.mock import MagicMock
    config = DatabaseConfig(
        db_type="postgresql",
        database="analytics",
        host="localhost",
        port=5432,
        username="user",
        password="pw",
    )
    manager = DatabaseManager(config)

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.keys.return_value = ["table_name"]
    mock_result.fetchmany.return_value = [("users",), ("orders",)]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    mock_engine.connect.return_value = mock_conn

    manager.engine = mock_engine
    manager.connect = MagicMock(return_value=mock_engine)

    query = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name LIMIT 14;'
    res = manager.execute_query(query)

    assert res["columns"] == ["table_name"]
    assert res["row_count"] == 2
    # Verify the executed SQL statement string was adapted for PostgreSQL
    executed_text_obj = mock_conn.execute.call_args[0][0]
    executed_sql = str(executed_text_obj)
    assert "information_schema.tables" in executed_sql
    assert "table_schema = 'public'" in executed_sql
    assert "sqlite_master" not in executed_sql


def test_adapt_sql_dialect_invalid_and_empty():
    from database import adapt_sql_dialect
    assert adapt_sql_dialect(None) is None
    assert adapt_sql_dialect("") == ""
    assert adapt_sql_dialect(123) == 123


def test_adapt_sql_dialect_postgres_operators_and_clauses():
    from database import adapt_sql_dialect
    # Operators LIKE, ILIKE, NOT LIKE
    q1 = 'SELECT * FROM users WHERE name LIKE "Alice%" AND email ILIKE "%.com" AND tag NOT LIKE "test";'
    ad1 = adapt_sql_dialect(q1, dialect="postgresql")
    assert "LIKE 'Alice%'" in ad1
    assert "ILIKE '%.com'" in ad1
    assert "NOT LIKE 'test'" in ad1
    assert '"' not in ad1

    # BETWEEN
    q2 = 'SELECT * FROM metrics WHERE date BETWEEN "2024-01-01" AND "2024-12-31";'
    ad2 = adapt_sql_dialect(q2, dialect="postgresql")
    assert "BETWEEN '2024-01-01' AND '2024-12-31'" in ad2

    # IN lists with double quotes
    q3 = 'SELECT * FROM orders WHERE status IN ("pending", "completed", "cancelled");'
    ad3 = adapt_sql_dialect(q3, dialect="postgresql")
    assert "IN ('pending', 'completed', 'cancelled')" in ad3

    # WHERE clause already present with sqlite_master
    q4 = "SELECT name FROM sqlite_master WHERE active = 1;"
    ad4 = adapt_sql_dialect(q4, dialect="postgresql")
    assert "WHERE table_schema = 'public' AND active = 1;" in ad4

    # No WHERE, but LIMIT without semicolon
    q5 = "SELECT name FROM sqlite_master LIMIT 10"
    ad5 = adapt_sql_dialect(q5, dialect="postgresql")
    assert "WHERE table_schema = 'public' LIMIT 10" in ad5

    # No WHERE, no clause, no semicolon
    q6 = "SELECT name FROM sqlite_master"
    ad6 = adapt_sql_dialect(q6, dialect="postgresql")
    assert ad6 == "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"

    # strftime with other format specifiers (%d, %H, %M, %S, %j, %w)
    q7 = "SELECT strftime('%d', ts), strftime('%H:%M:%S', ts), strftime('%j', ts), strftime('%w', ts) FROM log;"
    ad7 = adapt_sql_dialect(q7, dialect="postgresql")
    assert "TO_CHAR(ts, 'DD')" in ad7
    assert "TO_CHAR(ts, 'HH24:MI:SS')" in ad7
    assert "TO_CHAR(ts, 'DDD')" in ad7
    assert "TO_CHAR(ts, 'D')" in ad7


def test_adapt_sql_dialect_mysql_branches():
    from database import adapt_sql_dialect
    # Comparison operators with double quotes in MySQL
    q1 = 'SELECT * FROM items WHERE price != "10" AND cost <> "5" AND qty <= "100";'
    ad1 = adapt_sql_dialect(q1, dialect="mysql")
    assert "!= '10'" in ad1
    assert "<> '5'" in ad1
    assert "<= '100'" in ad1

    # MySQL metadata table and type replacement with existing WHERE
    q2 = 'SELECT name, tbl_name, type FROM sqlite_master WHERE active = 1;'
    ad2 = adapt_sql_dialect(q2, dialect="mysql")
    assert "information_schema.tables" in ad2
    assert "WHERE table_schema = DATABASE() AND active = 1;" in ad2
    assert "table_name" in ad2
    assert "tbl_name" not in ad2
    assert "table_type" in ad2

    # MySQL with ORDER BY (no WHERE)
    q3 = 'SELECT name FROM sqlite_master ORDER BY name;'
    ad3 = adapt_sql_dialect(q3, dialect="mysql")
    assert "WHERE table_schema = DATABASE() ORDER BY table_name;" in ad3

    # MySQL with trailing semicolon (no WHERE, no clause)
    q4 = 'SELECT name FROM sqlite_master;'
    ad4 = adapt_sql_dialect(q4, dialect="mysql")
    assert ad4 == "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE();"

    # MySQL without semicolon and no clause
    q5 = 'SELECT name FROM sqlite_master'
    ad5 = adapt_sql_dialect(q5, dialect="mysql")
    assert ad5 == "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"

    # MariaDB alias behaves identically
    ad6 = adapt_sql_dialect("SELECT name FROM sqlite_master;", dialect="mariadb")
    assert "table_schema = DATABASE()" in ad6

    # MySQL view mapping
    ad7 = adapt_sql_dialect('SELECT name FROM sqlite_master WHERE type = "view";', dialect="mysql")
    assert "table_schema = DATABASE() AND table_type = 'VIEW'" in ad7

    # MySQL composite table & view mapping
    ad8 = adapt_sql_dialect('SELECT name FROM sqlite_master WHERE type IN ("table", "view");', dialect="mysql")
    assert "table_schema = DATABASE() AND table_type IN ('BASE TABLE', 'VIEW')" in ad8

    # MySQL LIKE, BETWEEN, IN with double quotes
    q_ops = 'SELECT * FROM items WHERE name LIKE "prod%" AND category IN ("A", "B") AND price BETWEEN "10" AND "50";'
    ad_ops = adapt_sql_dialect(q_ops, dialect="mysql")
    assert "LIKE 'prod%'" in ad_ops
    assert "IN ('A', 'B')" in ad_ops
    assert "BETWEEN '10' AND '50'" in ad_ops

    # MySQL with GROUP BY (no WHERE)
    q_grp = 'SELECT type, count(*) FROM sqlite_master GROUP BY type;'
    ad_grp = adapt_sql_dialect(q_grp, dialect="mysql")
    assert "WHERE table_schema = DATABASE() GROUP BY table_type;" in ad_grp

    # MySQL with HAVING (no WHERE)
    q_hav = 'SELECT type FROM sqlite_master HAVING count(*) > 1;'
    ad_hav = adapt_sql_dialect(q_hav, dialect="mysql")
    assert "WHERE table_schema = DATABASE() HAVING count(*) > 1;" in ad_hav



def test_database_config_and_urls():
    # URL provided directly
    cfg_url = DatabaseConfig(db_type="sqlite", database="", url="sqlite:///:memory:")
    mgr_url = DatabaseManager(cfg_url)
    assert mgr_url._build_connection_url() == "sqlite:///:memory:"

    # Unsupported db type
    cfg_bad = DatabaseConfig(db_type="oracle", database="mydb")
    with pytest.raises(ValueError, match="Unsupported database type"):
        DatabaseManager(cfg_bad)._build_connection_url()

    # Missing credentials for PostgreSQL
    cfg_pg_missing = DatabaseConfig(db_type="postgresql", database="analytics", host="localhost", port=5432, username=None, password=None)
    with pytest.raises(ValueError, match="host, port, username and password are required"):
        DatabaseManager(cfg_pg_missing)._build_connection_url()

    # Valid MySQL configuration
    cfg_mysql = DatabaseConfig(
        db_type="mysql",
        database="production",
        host="db.example.com",
        port=3306,
        username="dbadmin",
        password="secretpassword",
    )
    url = DatabaseManager(cfg_mysql)._build_connection_url()
    assert url == "mysql+pymysql://dbadmin:secretpassword@db.example.com:3306/production"

    # Missing credentials for MySQL
    cfg_my_missing = DatabaseConfig(db_type="mysql", database="mydb", host="localhost", port=3306, username="user", password=None)
    with pytest.raises(ValueError, match="host, port, username and password are required"):
        DatabaseManager(cfg_my_missing)._build_connection_url()


def test_empty_database_schema_and_dict(tmp_path):
    empty_db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(empty_db))
    conn.close()

    mgr = DatabaseManager(DatabaseConfig(db_type="sqlite", database=str(empty_db)))
    schema = mgr.get_schema()
    assert schema == "-- Database contains no tables."

    schema_dict = mgr.get_schema_dict()
    assert schema_dict == {}
    mgr.close()


def test_get_schema_dict_populated(sqlite_test_db):
    mgr = DatabaseManager(DatabaseConfig(db_type="sqlite", database=sqlite_test_db))
    schema_dict = mgr.get_schema_dict()
    assert "categories" in schema_dict
    assert "products" in schema_dict

    cat_info = schema_dict["categories"]
    col_names = [col["name"] for col in cat_info["columns"]]
    assert "id" in col_names
    assert "name" in col_names
    assert cat_info["primary_key"] == ["id"]

    prod_info = schema_dict["products"]
    assert len(prod_info["foreign_keys"]) == 1
    assert prod_info["foreign_keys"][0]["referred_table"] == "categories"
    mgr.close()


def test_query_validation_edge_cases():
    # Only comments
    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        DatabaseManager.validate_sql("-- comment line 1\n-- comment line 2")

    with pytest.raises(ValueError, match="SQL query cannot be empty"):
        DatabaseManager.validate_sql("/* only block comments */")

    # Blocked keyword embedded inside a SELECT statement
    with pytest.raises(ValueError, match="Blocked SQL operation detected: DROP"):
        DatabaseManager.validate_sql("SELECT * FROM users WHERE id IN (SELECT id FROM accounts) AND DROP = 1")


def test_build_connection_url_supported_db_fallback(monkeypatch):
    monkeypatch.setattr(DatabaseManager, "SUPPORTED_DATABASES", {"sqlite", "postgresql", "mysql", "other"})
    cfg = DatabaseConfig(db_type="other", database="mydb", host="localhost", port=1234, username="u", password="p")
    with pytest.raises(ValueError, match="Unsupported database type: other"):
        DatabaseManager(cfg)._build_connection_url()


def test_database_manager_context_manager_and_error(sqlite_test_db):
    cfg = DatabaseConfig(db_type="sqlite", database=sqlite_test_db)
    with DatabaseManager(cfg) as mgr:
        assert mgr.test_connection() is True
        # Execute invalid query raises RuntimeError
        with pytest.raises(RuntimeError, match="SQL execution failed"):
            mgr.execute_query("SELECT * FROM non_existent_table")

    # Outside context manager, engine should be closed
    assert mgr.engine is None


def test_adapt_sql_dialect_postgres_group_by_and_having():
    from database import adapt_sql_dialect
    # Query with GROUP BY (no WHERE)
    q1 = "SELECT type, count(*) FROM sqlite_master GROUP BY type;"
    ad1 = adapt_sql_dialect(q1, dialect="postgresql")
    assert "WHERE table_schema = 'public' GROUP BY table_type;" in ad1

    # Query with HAVING (no WHERE)
    q2 = "SELECT type FROM sqlite_master HAVING count(*) > 1;"
    ad2 = adapt_sql_dialect(q2, dialect="postgresql")
    assert "WHERE table_schema = 'public' HAVING count(*) > 1;" in ad2


def test_adapt_sql_dialect_postgres_strftime_comparison_operators():
    from database import adapt_sql_dialect
    # Test operators !=, <>, <=, >=, <, > with integer comparison
    ops = ["!=", "<>", "<=", ">=", "<", ">"]
    for op in ops:
        q = f"SELECT * FROM logs WHERE strftime('%Y', created_at) {op} 2023;"
        ad = adapt_sql_dialect(q, dialect="postgresql")
        assert f"EXTRACT(YEAR FROM created_at) {op} 2023" in ad


def test_adapt_sql_dialect_postgres_column_name_safety():
    from database import adapt_sql_dialect
    # Columns containing 'name' or 'type' as part of their identifier should not be altered
    q = 'SELECT first_name, last_name, user_type FROM sqlite_master WHERE type = "table";'
    ad = adapt_sql_dialect(q, dialect="postgresql")
    assert "first_name" in ad
    assert "last_name" in ad
    assert "user_type" in ad
    assert "table_type = 'BASE TABLE'" in ad


def test_database_manager_execute_query_dialect_fallback():
    from unittest.mock import MagicMock
    cfg = DatabaseConfig(db_type="sqlite", database=":memory:")
    mgr = DatabaseManager(cfg)

    # Mock engine without dialect attribute
    mock_engine = MagicMock(spec=[])
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.keys.return_value = ["num"]
    mock_result.fetchmany.return_value = [(1,)]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    mock_engine.connect = MagicMock(return_value=mock_conn)

    mgr.engine = mock_engine
    mgr.connect = MagicMock(return_value=mock_engine)

    res = mgr.execute_query("SELECT 1 AS num;")
    assert res["columns"] == ["num"]
    assert res["rows"] == [[1]]


def test_database_manager_close_idempotent():
    cfg = DatabaseConfig(db_type="sqlite", database=":memory:")
    mgr = DatabaseManager(cfg)
    assert mgr.engine is None
    # Calling close() on uninitialized manager should not error (covers line 535->exit)
    mgr.close()
    assert mgr.engine is None


def test_supabase_build_connection_url_defaults():
    cfg = DatabaseConfig(
        db_type="supabase",
        database="",
        host="aws-0-us-east-1.pooler.supabase.com",
        port=None,
        username="",
        password="secretpassword",
    )
    mgr = DatabaseManager(cfg)
    url = mgr._build_connection_url()
    assert url == "postgresql://postgres:secretpassword@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"


def test_supabase_build_connection_url_custom_and_ssl_enforcement():
    # Custom credentials and port
    cfg = DatabaseConfig(
        db_type="supabase",
        database="analytics_prod",
        host="db.myref.supabase.co",
        port=6543,
        username="postgres.myref",
        password="custom_password",
    )
    mgr = DatabaseManager(cfg)
    url = mgr._build_connection_url()
    assert url == "postgresql://postgres.myref:custom_password@db.myref.supabase.co:6543/analytics_prod?sslmode=require"

    # Database with sslmode already present should not duplicate it
    cfg2 = DatabaseConfig(
        db_type="supabase",
        database="postgres?sslmode=require",
        host="db.myref.supabase.co",
        port=5432,
        username="postgres",
        password="secretpassword",
    )
    url2 = DatabaseManager(cfg2)._build_connection_url()
    assert url2 == "postgresql://postgres:secretpassword@db.myref.supabase.co:5432/postgres?sslmode=require"
    assert url2.count("sslmode=require") == 1

    # Direct URL with supabase enforces sslmode=require
    cfg_url1 = DatabaseConfig(
        db_type="supabase",
        database="",
        url="postgresql://user:pass@db.myref.supabase.co:5432/postgres",
    )
    assert DatabaseManager(cfg_url1)._build_connection_url() == "postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=require"

    # Direct URL with query param
    cfg_url2 = DatabaseConfig(
        db_type="supabase",
        database="",
        url="postgresql://user:pass@db.myref.supabase.co:5432/postgres?connect_timeout=10",
    )
    assert DatabaseManager(cfg_url2)._build_connection_url() == "postgresql://user:pass@db.myref.supabase.co:5432/postgres?connect_timeout=10&sslmode=require"

    # Direct URL already with sslmode=require
    cfg_url3 = DatabaseConfig(
        db_type="supabase",
        database="",
        url="postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=require",
    )
    assert DatabaseManager(cfg_url3)._build_connection_url() == "postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=require"

    # Direct URL with sslmode=disable overridden to sslmode=require
    cfg_url4 = DatabaseConfig(
        db_type="supabase",
        database="",
        url="postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=disable",
    )
    assert DatabaseManager(cfg_url4)._build_connection_url() == "postgresql://user:pass@db.myref.supabase.co:5432/postgres?sslmode=require"

    # Password with special characters URL-encoded safely and database param overriding sslmode
    cfg_special = DatabaseConfig(
        db_type="supabase",
        database="postgres?sslmode=disable",
        host="db.myref.supabase.co",
        port=5432,
        username="postgres.team",
        password="my@p:a/s#s",
    )
    url_special = DatabaseManager(cfg_special)._build_connection_url()
    assert "sslmode=require" in url_special
    assert "sslmode=disable" not in url_special
    assert "my%40p%3Aa%2Fs%23s" in url_special
    assert "postgres.team" in url_special


def test_supabase_build_connection_url_missing_credentials():
    # Missing host
    cfg_no_host = DatabaseConfig(
        db_type="supabase",
        database="postgres",
        host=None,
        password="pwd",
    )
    with pytest.raises(ValueError, match="host, port, username and password are required for supabase"):
        DatabaseManager(cfg_no_host)._build_connection_url()

    # Whitespace host
    cfg_ws_host = DatabaseConfig(
        db_type="supabase",
        database="postgres",
        host="   ",
        password="pwd",
    )
    with pytest.raises(ValueError, match="host, port, username and password are required for supabase"):
        DatabaseManager(cfg_ws_host)._build_connection_url()

    # Missing password
    cfg_no_pw = DatabaseConfig(
        db_type="supabase",
        database="postgres",
        host="db.supabase.co",
        password=None,
    )
    with pytest.raises(ValueError, match="host, port, username and password are required for supabase"):
        DatabaseManager(cfg_no_pw)._build_connection_url()


def test_supabase_and_postgres_public_schema_filtering():
    from unittest.mock import MagicMock, patch

    # 1. Supabase schema introspection strictly queries schema='public'
    cfg_supabase = DatabaseConfig(
        db_type="supabase",
        database="postgres",
        host="db.supabase.co",
        port=5432,
        username="postgres",
        password="pw",
    )
    mgr_sb = DatabaseManager(cfg_supabase)
    mock_engine = MagicMock()
    mgr_sb.engine = mock_engine
    mgr_sb.connect = MagicMock(return_value=mock_engine)

    with patch("database.inspect") as mock_inspect:
        mock_inspector = MagicMock()
        # Mock public user tables vs system schemas
        mock_inspector.get_table_names.return_value = ["customers", "orders"]
        mock_inspector.get_columns.return_value = [{"name": "id", "type": "INTEGER", "nullable": False}]
        mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspect.return_value = mock_inspector

        tables = mgr_sb.get_table_names()
        assert tables == ["customers", "orders"]
        mock_inspector.get_table_names.assert_called_with(schema="public")

        schema_text = mgr_sb.get_schema()
        assert "CREATE TABLE customers" in schema_text
        assert "CREATE TABLE orders" in schema_text
        mock_inspector.get_columns.assert_any_call("customers", schema="public")
        mock_inspector.get_columns.assert_any_call("orders", schema="public")

        schema_dict = mgr_sb.get_schema_dict()
        assert "customers" in schema_dict
        assert "orders" in schema_dict

        # Introspect table with foreign key referring to auth schema (auth.users)
        mock_inspector.get_table_names.return_value = ["profiles"]
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "UUID", "nullable": False},
            {"name": "user_id", "type": "UUID", "nullable": False},
        ]
        mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
        mock_inspector.get_foreign_keys.return_value = [
            {
                "name": "profiles_user_id_fkey",
                "constrained_columns": ["user_id"],
                "referred_schema": "auth",
                "referred_table": "users",
                "referred_columns": ["id"],
            }
        ]
        schema_text = mgr_sb.get_schema()
        assert "-- FOREIGN KEY: profiles.user_id REFERENCES auth.users.id" in schema_text

        # Schema alias targets
        assert DatabaseManager(DatabaseConfig(db_type="postgres", database="db"))._get_target_schema() == "public"
        assert DatabaseManager(DatabaseConfig(db_type="psql", database="db"))._get_target_schema() == "public"
        assert DatabaseManager(DatabaseConfig(db_type="mysql", database="db"))._get_target_schema() is None
        assert DatabaseManager(DatabaseConfig(db_type="sqlite", database=":memory:"))._get_target_schema() is None

    # 2. SQLite queries without schema kwarg
    cfg_sqlite = DatabaseConfig(db_type="sqlite", database=":memory:")
    mgr_sqlite = DatabaseManager(cfg_sqlite)
    mgr_sqlite.engine = mock_engine
    mgr_sqlite.connect = MagicMock(return_value=mock_engine)

    with patch("database.inspect") as mock_inspect:
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ["items"]
        mock_inspector.get_columns.return_value = [{"name": "id", "type": "INTEGER", "nullable": False}]
        mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspect.return_value = mock_inspector

        mgr_sqlite.get_table_names()
        mock_inspector.get_table_names.assert_called_with()

        mgr_sqlite.get_schema()
        mock_inspector.get_columns.assert_called_with("items")

        mgr_sqlite.get_schema_dict()


def test_adapt_sql_dialect_supabase():
    from database import adapt_sql_dialect

    # 1. Map sqlite_master -> information_schema.tables with schema='public'
    sql = 'SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name;'
    adapted = adapt_sql_dialect(sql, dialect="supabase")
    assert "information_schema.tables" in adapted
    assert "table_schema = 'public'" in adapted
    assert "table_type = 'BASE TABLE'" in adapted
    assert "sqlite_master" not in adapted
    assert '"table"' not in adapted

    # 2. Convert double quotes on string literals
    sql2 = 'SELECT * FROM users WHERE status = "active" AND role IN ("admin", "editor");'
    adapted2 = adapt_sql_dialect(sql2, dialect="supabase")
    assert "status = 'active'" in adapted2
    assert "IN ('admin', 'editor')" in adapted2

    # 3. IFNULL -> COALESCE
    sql3 = 'SELECT IFNULL(score, 0) FROM scores;'
    adapted3 = adapt_sql_dialect(sql3, dialect="supabase")
    assert "COALESCE(score, 0)" in adapted3
    assert "IFNULL" not in adapted3

    # 4. strftime -> TO_CHAR / EXTRACT
    sql4 = "SELECT * FROM sales WHERE strftime('%Y', sale_date) = 2024;"
    adapted4 = adapt_sql_dialect(sql4, dialect="supabase")
    assert "EXTRACT(YEAR FROM sale_date) = 2024" in adapted4


def test_database_manager_execute_query_supabase_adaptation():
    from unittest.mock import MagicMock
    cfg = DatabaseConfig(
        db_type="supabase",
        database="postgres",
        host="aws-0.pooler.supabase.com",
        port=5432,
        username="postgres",
        password="secretpassword",
    )
    mgr = DatabaseManager(cfg)

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_conn = MagicMock()
    mock_result = MagicMock()
    mock_result.keys.return_value = ["table_name"]
    mock_result.fetchmany.return_value = [("users",), ("profiles",)]
    mock_conn.execute.return_value = mock_result
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False
    mock_engine.connect = MagicMock(return_value=mock_conn)

    mgr.engine = mock_engine
    mgr.connect = MagicMock(return_value=mock_engine)

    query = 'SELECT name FROM sqlite_master WHERE type = "table";'
    res = mgr.execute_query(query)

    assert res["columns"] == ["table_name"]
    assert res["row_count"] == 2
    executed_text_obj = mock_conn.execute.call_args[0][0]
    executed_sql = str(executed_text_obj)
    assert "information_schema.tables" in executed_sql
    assert "table_schema = 'public'" in executed_sql
    assert "sqlite_master" not in executed_sql





