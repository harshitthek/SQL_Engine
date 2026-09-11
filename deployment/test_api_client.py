"""
Unit tests for FastAPIClient.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests
from api_client import (
    FastAPIClient,
    FastAPIClientError,
    FastAPIUnavailableError,
    InferenceBusyError,
    InferenceFailedError,
    ModelNotReadyError,
    RateLimitExceededError,
    RequestValidationError,
)


def test_health_healthy():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "healthy",
            "model_version": "text2sql-v1",
            "uptime_seconds": 123.45,
            "device": "mps",
            "error": None,
        }
        mock_get.return_value = mock_resp

        res = client.health()
        assert res["status"] == "healthy"
        assert res["model_version"] == "text2sql-v1"
        assert res["device"] == "mps"
        assert res["error"] is None


def test_health_degraded():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "degraded",
            "model_version": "text2sql-v1",
            "uptime_seconds": 12.0,
            "device": "none",
            "error": "Failed to load model weights",
        }
        mock_get.return_value = mock_resp

        res = client.health()
        assert res["status"] == "degraded"
        assert res["error"] == "Failed to load model weights"


def test_health_offline_on_connection_error():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        res = client.health()
        assert res["status"] == "offline"
        assert "Cannot reach FastAPI server" in res["error"]


def test_generate_sql_success():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "request_id": "req-1234",
            "sql": "SELECT count(*) FROM users;",
            "model": "text2sql-v1",
            "generation_time_ms": 120.5,
        }
        mock_post.return_value = mock_resp

        res = client.generate_sql(
            question="How many users are there?",
            schema="CREATE TABLE users (id int);",
        )
        assert res["sql"] == "SELECT count(*) FROM users;"
        assert res["request_id"] == "req-1234"
        assert res["model"] == "text2sql-v1"


def test_generate_sql_input_validation():
    client = FastAPIClient(base_url="http://mock-server:8000")
    # Empty question
    with pytest.raises(RequestValidationError, match="Question cannot be empty"):
        client.generate_sql(question="", schema="CREATE TABLE users (id int);")

    # Short question
    with pytest.raises(RequestValidationError, match="at least 3 characters"):
        client.generate_sql(question="ab", schema="CREATE TABLE users (id int);")

    # Empty schema
    with pytest.raises(RequestValidationError, match="schema cannot be empty"):
        client.generate_sql(question="How many users?", schema="")


def test_generate_sql_model_not_ready():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {
            "error_code": "MODEL_NOT_READY",
            "message": "Model not loaded.",
        }
        mock_post.return_value = mock_resp

        with pytest.raises(ModelNotReadyError, match="Model not ready"):
            client.generate_sql(
                question="How many users?",
                schema="CREATE TABLE users (id int);",
            )


def test_generate_sql_service_busy():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {
            "error_code": "SERVICE_BUSY",
            "message": "Queue wait limit exceeded.",
        }
        mock_post.return_value = mock_resp

        with pytest.raises(InferenceBusyError, match="Inference queue saturated"):
            client.generate_sql(
                question="How many users?",
                schema="CREATE TABLE users (id int);",
            )


def test_generate_sql_server_unavailable():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(FastAPIUnavailableError, match="FastAPI service is unavailable"):
            client.generate_sql(
                question="How many users?",
                schema="CREATE TABLE users (id int);",
            )


def test_generate_sql_passes_dialect():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "request_id": "req-999",
            "sql": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';",
            "model": "text2sql-v1",
            "generation_time_ms": 110.0,
        }
        mock_post.return_value = mock_resp

        res = client.generate_sql(
            question="Show 14 tables",
            schema="CREATE TABLE users (id int);",
            dialect="postgresql",
        )

        assert res["sql"].startswith("SELECT table_name")
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["dialect"] == "postgresql"
        assert sent_payload["question"] == "Show 14 tables"


def test_generate_sql_default_dialect_sqlite():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "request_id": "req-998",
            "sql": "SELECT count(*) FROM users;",
            "model": "text2sql-v1",
            "generation_time_ms": 95.0,
        }
        mock_post.return_value = mock_resp

        client.generate_sql(
            question="How many users?",
            schema="CREATE TABLE users (id int);",
        )

        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["dialect"] == "sqlite"


def test_api_tosql_request_dialect_validation():
    from api import ToSQLRequest
    # Default dialect
    req1 = ToSQLRequest(
        question="Show all users",
        schema="CREATE TABLE users (id INT);",
    )
    assert req1.dialect == "sqlite"

    # Explicit postgresql dialect
    req2 = ToSQLRequest(
        question="Show all users",
        schema="CREATE TABLE users (id INT);",
        dialect="postgresql",
    )
    assert req2.dialect == "postgresql"


def test_health_non_200_status():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_get.return_value = mock_resp

        res = client.health()
        assert res["status"] == "degraded"
        assert "Health check returned HTTP 502" in res["error"]


def test_health_unexpected_exception():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "get") as mock_get:
        mock_get.side_effect = RuntimeError("DNS resolution crash")

        res = client.health()
        assert res["status"] == "offline"
        assert "Health check failed: DNS resolution crash" in res["error"]


def test_generate_sql_length_validations():
    client = FastAPIClient(base_url="http://mock-server:8000")
    # Question too long (> 2000 chars)
    with pytest.raises(RequestValidationError, match="Question exceeds maximum allowed length"):
        client.generate_sql(question="x" * 2001, schema="CREATE TABLE t (id int);")

    # Schema too short (< 5 chars)
    with pytest.raises(RequestValidationError, match="Database schema must contain at least 5 characters"):
        client.generate_sql(question="Show tables", schema="ab")

    # Schema too long (> 50000 chars)
    with pytest.raises(RequestValidationError, match="Database schema exceeds maximum allowed length"):
        client.generate_sql(question="Show tables", schema="CREATE TABLE t (" + "x" * 50001 + ");")


def test_generate_sql_generic_request_exception():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_post.side_effect = requests.RequestException("SSL handshake failed")

        with pytest.raises(FastAPIClientError, match="HTTP communication failed: SSL handshake failed"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")


def test_generate_sql_malformed_success_payload():
    client = FastAPIClient(base_url="http://mock-server:8000")
    # Missing 'sql' or 'request_id'
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}  # missing sql, request_id
        mock_post.return_value = mock_resp

        with pytest.raises(FastAPIClientError, match="Malformed response payload from /v1/tosql"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")

    # JSON decode failure on 200 response
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_resp

        with pytest.raises(FastAPIClientError, match="Failed to decode response JSON"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")


def test_generate_sql_server_error_codes():
    client = FastAPIClient(base_url="http://mock-server:8000")

    # 422 Validation Error
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 422
        mock_resp.json.return_value = {"message": "Invalid field value"}
        mock_post.return_value = mock_resp

        with pytest.raises(RequestValidationError, match="Validation Error: Invalid field value"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")

    # 503 Generic Service Unavailable (not MODEL_NOT_READY or SERVICE_BUSY)
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {"error_code": "GENERIC_UNAVAILABLE", "message": "Host in maintenance"}
        mock_post.return_value = mock_resp

        with pytest.raises(FastAPIUnavailableError, match="Service unavailable: Host in maintenance"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")

    # 500 Inference Failure
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"message": "CUDA out of memory"}
        mock_post.return_value = mock_resp

        with pytest.raises(InferenceFailedError, match="Model generation failed on server: CUDA out of memory"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")

    # 400 Bad Request
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"message": "Malformed JSON payload"}
        mock_post.return_value = mock_resp

        with pytest.raises(RequestValidationError, match="Bad request: Malformed JSON payload"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")

    # Other status code (e.g. 502) with non-JSON error body
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("Not JSON")
        mock_resp.text = "Bad Gateway HTML error"
        mock_post.return_value = mock_resp

        with pytest.raises(FastAPIClientError, match="Server returned HTTP 502: Bad Gateway HTML error"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")


def test_generate_sql_rate_limited():
    client = FastAPIClient(base_url="http://mock-server:8000")
    with patch.object(client.session, "post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {
            "error_code": "RATE_LIMITED",
            "message": "Too many requests from this IP. Please try again later.",
            "request_id": "test-rate-limit-id",
        }
        mock_post.return_value = mock_resp

        with pytest.raises(RateLimitExceededError, match="Rate limit exceeded: Too many requests from this IP"):
            client.generate_sql(question="Count users", schema="CREATE TABLE users (id int);")



