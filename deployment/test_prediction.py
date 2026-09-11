"""
Unit tests for Text2SQLEngine in prediction.py.
"""
import os
from unittest.mock import MagicMock, patch
import pytest
import torch

from prediction import Text2SQLEngine


@pytest.fixture
def mock_transformers():
    with patch("prediction.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("prediction.AutoModelForCausalLM.from_pretrained") as mock_model:
        tok_inst = MagicMock()
        tok_inst.pad_token = None
        tok_inst.eos_token = "<eos>"
        tok_inst.pad_token_id = 0
        tok_inst.eos_token_id = 1
        tok_inst.model_max_length = 2048
        mock_tok.return_value = tok_inst

        model_inst = MagicMock()
        model_inst.to.return_value = model_inst
        mock_model.return_value = model_inst

        yield mock_tok, mock_model, tok_inst, model_inst


def test_engine_init_with_explicit_params(mock_transformers):
    mock_tok, mock_model, tok_inst, model_inst = mock_transformers
    engine = Text2SQLEngine(
        model_path="/custom/model/path",
        device="cpu",
        torch_dtype=torch.float32,
    )
    assert engine.model_path == "/custom/model/path"
    assert engine.device == "cpu"
    assert engine.torch_dtype == torch.float32
    assert tok_inst.pad_token == "<eos>"
    mock_tok.assert_called_once_with("/custom/model/path", trust_remote_code=True)
    mock_model.assert_called_once_with(
        "/custom/model/path",
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model_inst.to.assert_called_once_with("cpu")
    model_inst.eval.assert_called_once()


def test_engine_init_with_env_var(mock_transformers):
    with patch.dict(os.environ, {"TEXT2SQL_MODEL_PATH": "/env/model/path"}):
        engine = Text2SQLEngine(device="cpu")
        assert engine.model_path == "/env/model/path"


def test_engine_init_with_candidate_path(mock_transformers):
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.isdir") as mock_isdir:
        # candidate_paths[1] exists
        def isdir_side_effect(path):
            return os.path.normpath(path).endswith(os.path.normpath("models/text2sql-v1"))
        mock_isdir.side_effect = isdir_side_effect

        engine = Text2SQLEngine(device="cpu")
        assert os.path.normpath(engine.model_path).endswith(os.path.normpath("models/text2sql-v1"))


def test_engine_init_with_glob_cached_match(mock_transformers):
    with patch.dict(os.environ, {}, clear=True), \
         patch("os.path.isdir", return_value=False), \
         patch("glob.glob", return_value=["/cache/v1/text2sql-v1"]):
        engine = Text2SQLEngine(device="cpu")
        assert engine.model_path == "/cache/v1/text2sql-v1"


def test_engine_init_with_kagglehub_download_subpath_exists(mock_transformers):
    with patch.dict(os.environ, {}, clear=True), \
         patch("glob.glob", return_value=[]), \
         patch("kagglehub.model_download", return_value="/download/dir") as mock_dl, \
         patch("os.path.isdir") as mock_isdir:
        # First candidate paths return False, sub_path returns True
        def isdir_side_effect(path):
            return os.path.normpath(path) == os.path.normpath("/download/dir/text2sql-v1")
        mock_isdir.side_effect = isdir_side_effect

        engine = Text2SQLEngine(device="cpu")
        assert os.path.normpath(engine.model_path) == os.path.normpath("/download/dir/text2sql-v1")
        mock_dl.assert_called_once_with("pernavjain/text2sql-qwen/pyTorch/v1")


def test_engine_init_with_kagglehub_download_direct_path(mock_transformers):
    with patch.dict(os.environ, {}, clear=True), \
         patch("glob.glob", return_value=[]), \
         patch("kagglehub.model_download", return_value="/download/direct"), \
         patch("os.path.isdir", return_value=False):
        engine = Text2SQLEngine(device="cpu")
        assert engine.model_path == "/download/direct"


def test_engine_device_and_dtype_cuda_bf16(mock_transformers):
    _, mock_model, _, _ = mock_transformers
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        engine = Text2SQLEngine(model_path="/fake/path")
        assert engine.device == "cuda"
        assert engine.torch_dtype == torch.bfloat16
        mock_model.assert_called_once_with(
            "/fake/path",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )


def test_engine_device_and_dtype_cuda_fp16(mock_transformers):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=False):
        engine = Text2SQLEngine(model_path="/fake/path")
        assert engine.device == "cuda"
        assert engine.torch_dtype == torch.float16


def test_engine_device_and_dtype_mps(mock_transformers):
    _, _, _, model_inst = mock_transformers
    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps") as mock_mps:
        mock_mps.is_available.return_value = True
        engine = Text2SQLEngine(model_path="/fake/path")
        assert engine.device == "mps"
        assert engine.torch_dtype == torch.float16
        model_inst.to.assert_called_once_with("mps")


def test_engine_device_and_dtype_cpu(mock_transformers):
    _, _, _, model_inst = mock_transformers
    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps") as mock_mps:
        mock_mps.is_available.return_value = False
        engine = Text2SQLEngine(model_path="/fake/path")
        assert engine.device == "cpu"
        assert engine.torch_dtype == torch.float32
        model_inst.to.assert_called_once_with("cpu")


def test_engine_pad_token_already_present(mock_transformers):
    mock_tok, _, tok_inst, _ = mock_transformers
    tok_inst.pad_token = "<pad>"
    engine = Text2SQLEngine(model_path="/fake/path", device="cpu")
    assert engine.tokenizer.pad_token == "<pad>"


def test_format_prompt_dialects():
    engine = Text2SQLEngine.__new__(Text2SQLEngine)

    prompt_sqlite = engine.format_prompt("How many orders?", "CREATE TABLE orders (id INT);", dialect="sqlite")
    assert "write the exact SQLite query" in prompt_sqlite
    assert "CREATE TABLE orders" in prompt_sqlite
    assert "How many orders?" in prompt_sqlite

    prompt_pg1 = engine.format_prompt("Show users", "CREATE TABLE users (id INT);", dialect="postgresql")
    assert "write the exact PostgreSQL query" in prompt_pg1

    prompt_pg2 = engine.format_prompt("Show users", "CREATE TABLE users (id INT);", dialect="postgres")
    assert "write the exact PostgreSQL query" in prompt_pg2

    prompt_my = engine.format_prompt("Show users", "CREATE TABLE users (id INT);", dialect="mysql")
    assert "write the exact MySQL query" in prompt_my

    prompt_custom = engine.format_prompt("Show users", "CREATE TABLE users (id INT);", dialect="snowflake")
    assert "write the exact Snowflake query" in prompt_custom

    prompt_none = engine.format_prompt("Show users", "CREATE TABLE users (id INT);", dialect=None)
    assert "write the exact SQLite query" in prompt_none


def test_generate_sql_deterministic_and_sampling(mock_transformers):
    _, _, tok_inst, model_inst = mock_transformers
    engine = Text2SQLEngine(model_path="/fake/path", device="cpu")

    # Setup tokenizer call return
    input_ids = torch.tensor([[10, 20, 30]])
    tok_output = {"input_ids": input_ids}
    tok_inst.return_value = MagicMock()
    tok_inst.return_value.to.return_value = tok_output

    # Setup model output (input_ids + generated tokens)
    full_tokens = torch.tensor([[10, 20, 30, 40, 50]])
    model_inst.generate.return_value = full_tokens
    tok_inst.decode.return_value = "SELECT * FROM orders;"

    # 1. Deterministic (temperature = 0.0)
    sql = engine.generate_sql(
        question="Show orders",
        schema="CREATE TABLE orders (id INT);",
        dialect="sqlite",
        max_new_tokens=128,
        temperature=0.0,
    )
    assert sql == "SELECT * FROM orders;"
    gen_call_args = model_inst.generate.call_args[1]
    assert gen_call_args["do_sample"] is False
    assert "temperature" not in gen_call_args

    # 2. Sampling (temperature = 0.7)
    engine.generate_sql(
        question="Show orders",
        schema="CREATE TABLE orders (id INT);",
        dialect="sqlite",
        max_new_tokens=128,
        temperature=0.7,
    )
    gen_call_args_sampling = model_inst.generate.call_args[1]
    assert gen_call_args_sampling["do_sample"] is True
    assert gen_call_args_sampling["temperature"] == 0.7


def test_generate_sql_markdown_code_fences_stripping(mock_transformers):
    _, _, tok_inst, model_inst = mock_transformers
    engine = Text2SQLEngine(model_path="/fake/path", device="cpu")

    input_ids = torch.tensor([[1, 2]])
    tok_inst.return_value = MagicMock()
    tok_inst.return_value.to.return_value = {"input_ids": input_ids}
    model_inst.generate.return_value = torch.tensor([[1, 2, 3]])

    # Case A: ```sql ... ``` with trailing notes
    tok_inst.decode.return_value = "```sql\nSELECT id, name FROM users;\n```\nHere is your query."
    sql1 = engine.generate_sql("Users", "CREATE TABLE users (id INT, name TEXT);", dialect="sqlite")
    assert sql1 == "SELECT id, name FROM users;"

    # Case B: ``` ... ``` without sql tag
    tok_inst.decode.return_value = "```\nSELECT count(*) FROM users;\n```"
    sql2 = engine.generate_sql("Count", "CREATE TABLE users (id INT);", dialect="sqlite")
    assert sql2 == "SELECT count(*) FROM users;"

    # Case C: ```sql ... without closing fence
    tok_inst.decode.return_value = "```sql\nSELECT email FROM accounts;"
    sql3 = engine.generate_sql("Accounts", "CREATE TABLE accounts (email TEXT);", dialect="sqlite")
    assert sql3 == "SELECT email FROM accounts;"

    # Case D: ``` ... without closing fence
    tok_inst.decode.return_value = "```\nSELECT score FROM results;"
    sql4 = engine.generate_sql("Results", "CREATE TABLE results (score INT);", dialect="sqlite")
    assert sql4 == "SELECT score FROM results;"

    # Case E: SQL without semicolon
    tok_inst.decode.return_value = "SELECT name FROM customers"
    sql5 = engine.generate_sql("Customers", "CREATE TABLE customers (name TEXT);", dialect="sqlite")
    assert sql5 == "SELECT name FROM customers"


def test_generate_sql_commentary_after_semicolon(mock_transformers):
    _, _, tok_inst, model_inst = mock_transformers
    engine = Text2SQLEngine(model_path="/fake/path", device="cpu")

    input_ids = torch.tensor([[1]])
    tok_inst.return_value = MagicMock()
    tok_inst.return_value.to.return_value = {"input_ids": input_ids}
    model_inst.generate.return_value = torch.tensor([[1, 2]])

    # Commentary after semicolon starting with non-SQL keyword
    tok_inst.decode.return_value = "SELECT * FROM sales; Explanation: this extracts sales."
    sql = engine.generate_sql("Sales", "CREATE TABLE sales (id INT);", dialect="sqlite")
    assert sql == "SELECT * FROM sales;"

    # Valid query keyword after semicolon is not stripped by commentary filter
    tok_inst.decode.return_value = "SELECT 1; SELECT 2;"
    sql_multi = engine.generate_sql("Sales", "CREATE TABLE sales (id INT);", dialect="sqlite")
    assert "SELECT 1; SELECT 2;" in sql_multi

    # Semicolon followed by whitespace only
    tok_inst.decode.return_value = "SELECT * FROM products;   "
    sql_semi_ws = engine.generate_sql("Products", "CREATE TABLE products (id INT);", dialect="sqlite")
    assert sql_semi_ws == "SELECT * FROM products;"


def test_generate_sql_applies_dialect_adaptation(mock_transformers):
    _, _, tok_inst, model_inst = mock_transformers
    engine = Text2SQLEngine(model_path="/fake/path", device="cpu")

    input_ids = torch.tensor([[1]])
    tok_inst.return_value = MagicMock()
    tok_inst.return_value.to.return_value = {"input_ids": input_ids}
    model_inst.generate.return_value = torch.tensor([[1, 2]])

    # 1. Generate SQLite-style query targeting PostgreSQL:
    # IFNULL -> COALESCE, sqlite_master -> information_schema.tables, double quotes -> single quotes
    tok_inst.decode.return_value = 'SELECT IFNULL(name, 0) FROM sqlite_master WHERE type = "table" AND status = "active";'
    sql_pg = engine.generate_sql("Tables", "CREATE TABLE t (name TEXT);", dialect="postgresql")

    assert "COALESCE" in sql_pg
    assert "IFNULL" not in sql_pg
    assert "information_schema.tables" in sql_pg
    assert "table_schema = 'public'" in sql_pg
    assert "table_type = 'BASE TABLE'" in sql_pg
    assert '"table"' not in sql_pg
    assert "status = 'active'" in sql_pg
    assert '"active"' not in sql_pg

    # 2. Generate SQLite-style query targeting MySQL:
    tok_inst.decode.return_value = 'SELECT name FROM sqlite_master WHERE type = "table";'
    sql_mysql = engine.generate_sql("Tables", "CREATE TABLE t (name TEXT);", dialect="mysql")
    assert "information_schema.tables" in sql_mysql
    assert "table_schema = DATABASE()" in sql_mysql
    assert "table_type = 'BASE TABLE'" in sql_mysql

    # 3. Generate SQLite-style query targeting SQLite (should preserve SQLite constructs):
    tok_inst.decode.return_value = 'SELECT IFNULL(name, 0) FROM sqlite_master WHERE type = "table";'
    sql_sqlite = engine.generate_sql("Tables", "CREATE TABLE t (name TEXT);", dialect="sqlite")
    assert "sqlite_master" in sql_sqlite
    assert "IFNULL" in sql_sqlite

    # 4. Dialect None fallback (defaults to sqlite)
    sql_default = engine.generate_sql("Tables", "CREATE TABLE t (name TEXT);", dialect=None)
    assert "sqlite_master" in sql_default

