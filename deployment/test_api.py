"""
Unit tests for FastAPI Inference Gateway in api.py.
"""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from api import ErrorCode, ToSQLRequest, app, lifespan
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Setup test state on app.state
    app.state.start_time = 0.0
    app.state.inference_lock = asyncio.Lock()
    app.state.engine = None
    app.state.is_ready = False
    app.state.device = "none"
    app.state.startup_error = None
    if hasattr(app.state, "limiter") and app.state.limiter:
        app.state.limiter.reset()
    return TestClient(app, raise_server_exceptions=False)


def test_tosql_request_whitespace_validators():
    # Test classmethod validators directly for 100% branch coverage
    with pytest.raises(ValueError, match="Question cannot be empty or contain only whitespace"):
        ToSQLRequest.validate_question_non_empty("   ")

    with pytest.raises(ValueError, match="Schema cannot be empty or contain only whitespace"):
        ToSQLRequest.validate_schema_non_empty("   \t\n")

    # Valid values
    assert ToSQLRequest.validate_question_non_empty("  valid question?  ") == "valid question?"
    assert ToSQLRequest.validate_schema_non_empty("  CREATE TABLE t (id INT);  ") == "CREATE TABLE t (id INT);"


@pytest.mark.anyio
async def test_lifespan_success_and_shutdown():
    mock_app = MagicMock()
    mock_app.state = MagicMock()

    mock_engine = MagicMock()
    mock_engine.device = "mps"

    with patch("prediction.Text2SQLEngine", return_value=mock_engine), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True), \
         patch("torch.mps.empty_cache") as mock_mps_cache:
        async with lifespan(mock_app):
            assert mock_app.state.is_ready is True
            assert mock_app.state.device == "mps"
            assert mock_app.state.startup_error is None
            assert mock_app.state.engine == mock_engine

        assert mock_app.state.is_ready is False
        assert mock_app.state.engine is None
        mock_mps_cache.assert_called_once()


@pytest.mark.anyio
async def test_lifespan_cuda_shutdown():
    mock_app = MagicMock()
    mock_app.state = MagicMock()
    mock_engine = MagicMock()
    mock_engine.device = "cuda"

    with patch("prediction.Text2SQLEngine", return_value=mock_engine), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_cuda_cache:
        async with lifespan(mock_app):
            assert mock_app.state.device == "cuda"

        mock_cuda_cache.assert_called_once()


@pytest.mark.anyio
async def test_lifespan_empty_cache_exception():
    mock_app = MagicMock()
    mock_app.state = MagicMock()
    mock_engine = MagicMock()
    mock_engine.device = "cuda"

    with patch("prediction.Text2SQLEngine", return_value=mock_engine), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache", side_effect=RuntimeError("GPU disconnected")):
        # Should not raise even if empty_cache crashes
        async with lifespan(mock_app):
            pass



def test_lifespan_startup_failure():
    with patch("prediction.Text2SQLEngine", side_effect=RuntimeError("Weights download failed")):
        with TestClient(app):
            assert app.state.is_ready is False
            assert app.state.engine is None
            assert app.state.device == "none"
            assert "Weights download failed" in app.state.startup_error



@pytest.mark.anyio
async def test_lifespan_cpu_shutdown():
    mock_app = MagicMock()
    mock_app.state = MagicMock()
    mock_engine = MagicMock()
    mock_engine.device = "cpu"

    with patch("prediction.Text2SQLEngine", return_value=mock_engine), \
         patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps") as mock_mps:
        mock_mps.is_available.return_value = False
        async with lifespan(mock_app):
            assert mock_app.state.device == "cpu"

        assert mock_app.state.is_ready is False
        assert mock_app.state.engine is None




def test_request_id_middleware(client):
    # With explicit header
    resp1 = client.get("/health", headers={"X-Request-ID": "trace-uuid-123"})
    assert resp1.headers["X-Request-ID"] == "trace-uuid-123"

    # Without explicit header
    resp2 = client.get("/health")
    assert "X-Request-ID" in resp2.headers
    assert len(resp2.headers["X-Request-ID"]) > 10


def test_validation_exception_handler(client):
    # Missing required question and schema fields
    resp = client.post("/v1/tosql", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == ErrorCode.VALIDATION_ERROR
    assert "question" in body["message"]
    assert "schema" in body["message"]
    assert "request_id" in body


def test_health_endpoint(client):
    # Healthy state
    app.state.is_ready = True
    app.state.device = "mps"
    app.state.startup_error = None
    resp_healthy = client.get("/health")
    assert resp_healthy.status_code == 200
    data_healthy = resp_healthy.json()
    assert data_healthy["status"] == "healthy"
    assert data_healthy["device"] == "mps"
    assert data_healthy["error"] is None

    # Degraded state
    app.state.is_ready = False
    app.state.device = "none"
    app.state.startup_error = "Out of memory"
    resp_degraded = client.get("/health")
    assert resp_degraded.status_code == 200
    data_degraded = resp_degraded.json()
    assert data_degraded["status"] == "degraded"
    assert data_degraded["error"] == "Out of memory"


def test_tosql_model_not_ready(client):
    app.state.is_ready = False
    app.state.engine = None
    app.state.startup_error = "Tokenizer load crash"

    resp = client.post(
        "/v1/tosql",
        json={"question": "Count rows", "schema": "CREATE TABLE t (id int);"},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error_code"] == ErrorCode.MODEL_NOT_READY
    assert "Tokenizer load crash" in body["message"]

    # Without startup error
    app.state.startup_error = None
    resp2 = client.post(
        "/v1/tosql",
        json={"question": "Count rows", "schema": "CREATE TABLE t (id int);"},
    )
    assert resp2.status_code == 503
    assert resp2.json()["error_code"] == ErrorCode.MODEL_NOT_READY


def test_tosql_queue_timeout(client):
    app.state.is_ready = True
    app.state.engine = MagicMock()

    async def fake_wait_for(fut, timeout=None):
        fut.close()
        raise TimeoutError()

    with patch("asyncio.wait_for", side_effect=fake_wait_for):
        resp = client.post(
            "/v1/tosql",
            json={"question": "Count rows", "schema": "CREATE TABLE t (id int);"},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error_code"] == ErrorCode.SERVICE_BUSY
        assert "Inference queue timeout" in body["message"]


def test_tosql_success_and_dialect_forwarding(client):
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT * FROM orders WHERE table_schema = 'public';"
    app.state.engine = mock_engine

    # 1. Explicit dialect="postgresql"
    resp = client.post(
        "/v1/tosql",
        json={
            "question": "Show all orders",
            "schema": "CREATE TABLE orders (id INT, amount FLOAT);",
            "dialect": "postgresql",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sql"] == "SELECT * FROM orders WHERE table_schema = 'public';"
    assert data["model"] == "text2sql-v1"
    assert data["generation_time_ms"] >= 0.0

    mock_engine.generate_sql.assert_called_with(
        question="Show all orders",
        schema="CREATE TABLE orders (id INT, amount FLOAT);",
        dialect="postgresql",
        max_new_tokens=256,
        temperature=0.0,
    )

    # 2. Default dialect fallback when omitted
    client.post(
        "/v1/tosql",
        json={
            "question": "Show all orders",
            "schema": "CREATE TABLE orders (id INT, amount FLOAT);",
        },
    )
    call_args = mock_engine.generate_sql.call_args[1]
    assert call_args["dialect"] == "sqlite"


def test_tosql_inference_failed(client):
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.side_effect = RuntimeError("CUDA memory corruption")
    app.state.engine = mock_engine

    resp = client.post(
        "/v1/tosql",
        json={"question": "Count rows", "schema": "CREATE TABLE t (id int);"},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == ErrorCode.INFERENCE_FAILED
    assert "CUDA memory corruption" in body["message"]


class MockAsyncLock:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def test_reload_endpoint_success_cuda(client):
    app.state.inference_lock = MockAsyncLock()
    mock_new_engine = MagicMock()
    mock_new_engine.device = "cuda"

    with patch("prediction.Text2SQLEngine", return_value=mock_new_engine), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_cuda_cache:
        resp = client.post("/v1/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["device"] == "cuda"
        mock_cuda_cache.assert_called_once()


def test_reload_endpoint_success_mps(client):
    app.state.inference_lock = MockAsyncLock()
    mock_new_engine = MagicMock()
    mock_new_engine.device = "mps"

    with patch("prediction.Text2SQLEngine", return_value=mock_new_engine), \
         patch("torch.cuda.is_available", return_value=False):
        # Allow native torch mps on macOS to execute empty_cache
        resp = client.post("/v1/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


def test_reload_endpoint_empty_cache_exception(client):
    app.state.inference_lock = MockAsyncLock()
    mock_new_engine = MagicMock()
    mock_new_engine.device = "cuda"
    with patch("prediction.Text2SQLEngine", return_value=mock_new_engine), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache", side_effect=RuntimeError("GPU memory fault")):
        resp = client.post("/v1/reload")
        assert resp.status_code == 200


def test_reload_endpoint_success_cpu(client):
    app.state.inference_lock = MockAsyncLock()
    mock_new_engine = MagicMock()
    mock_new_engine.device = "cpu"
    with patch("prediction.Text2SQLEngine", return_value=mock_new_engine), \
         patch("torch.cuda.is_available", return_value=False), \
         patch.object(torch.backends, "mps") as mock_mps:
        mock_mps.is_available.return_value = False
        resp = client.post("/v1/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["device"] == "cpu"





def test_reload_endpoint_failure(client):
    app.state.inference_lock = MockAsyncLock()
    with patch("prediction.Text2SQLEngine", side_effect=RuntimeError("Weights download timeout")):
        resp = client.post("/v1/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert "Weights download timeout" in data["error"]
        assert app.state.is_ready is False


def test_reload_endpoint_no_lock(client):
    app.state.inference_lock = None
    resp = client.post("/v1/reload")
    assert resp.status_code == 200



def test_http_exception_handlers(client):
    # Define temporary routes to trigger various HTTPExceptions
    @app.get("/test/http-error-dict")
    async def route_dict():
        raise HTTPException(
            status_code=400,
            detail={"error_code": "CUSTOM_BAD_PARAM", "message": "Param foo is invalid."},
        )

    @app.get("/test/http-error-429")
    async def route_429():
        raise HTTPException(status_code=429, detail="Rate limit hit")

    @app.get("/test/http-error-500")
    async def route_500():
        raise HTTPException(status_code=500, detail="Internal DB fail")

    @app.get("/test/http-error-403")
    async def route_403():
        raise HTTPException(status_code=403, detail="Forbidden area")

    # 1. Dict detail
    r_dict = client.get("/test/http-error-dict")
    assert r_dict.status_code == 400
    assert r_dict.json()["error_code"] == "CUSTOM_BAD_PARAM"
    assert r_dict.json()["message"] == "Param foo is invalid."

    # 2. Status 429 -> RATE_LIMITED
    r_429 = client.get("/test/http-error-429")
    assert r_429.status_code == 429
    assert r_429.json()["error_code"] == ErrorCode.RATE_LIMITED

    # 3. Status 500 -> INTERNAL_SERVER_ERROR
    r_500 = client.get("/test/http-error-500")
    assert r_500.status_code == 500
    assert r_500.json()["error_code"] == ErrorCode.INTERNAL_SERVER_ERROR

    # 4. Status 403 -> default HTTP_ERROR
    r_403 = client.get("/test/http-error-403")
    assert r_403.status_code == 403
    assert r_403.json()["error_code"] == "HTTP_ERROR"


def test_unhandled_exception_handler(client):
    @app.get("/test/unhandled-crash")
    async def route_crash():
        raise ZeroDivisionError("division by zero crash")

    resp = client.get("/test/unhandled-crash")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == ErrorCode.INTERNAL_SERVER_ERROR
    assert body["message"] == "An unexpected internal server error occurred."


def test_api_main_block():
    import runpy
    with patch("uvicorn.run") as mock_uvicorn:
        with patch.dict(os.environ, {"API_HOST": "127.0.0.1", "API_PORT": "9000"}):
            runpy.run_module("api", run_name="__main__")
            mock_uvicorn.assert_called_once_with("api:app", host="127.0.0.1", port=9000, reload=False, workers=1)


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------

def test_health_not_blocked_by_inference_rate_limit(client):
    """Verify /health is exempt from the strict inference rate limit."""
    app.state.is_ready = True
    app.state.device = "mps"
    for _ in range(15):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


def test_tosql_succeeds_under_limit(client):
    """Verify /v1/tosql succeeds normally under the rate limit."""
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine

    resp = client.post(
        "/v1/tosql",
        json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sql"] == "SELECT 1;"
    assert data["model"] == "text2sql-v1"
    assert "generation_time_ms" in data
    assert "request_id" in data


def test_tosql_burst_rate_limit_exceeded(client):
    """Verify /v1/tosql enforces 3 req / 10s burst limit and returns standardized 429."""
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine

    # 3 requests within burst window succeed
    for i in range(3):
        resp = client.post(
            "/v1/tosql",
            json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
        )
        assert resp.status_code == 200, f"Request {i+1} should succeed"

    # 4th request from same IP within burst window must trigger 429
    resp_limited = client.post(
        "/v1/tosql",
        json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
        headers={"X-Request-ID": "test-burst-trace-id"},
    )
    assert resp_limited.status_code == 429
    body = resp_limited.json()
    assert body["error_code"] == ErrorCode.RATE_LIMITED
    assert body["message"] == "Too many requests from this IP. Please try again later."
    assert body["request_id"] == "test-burst-trace-id"
    assert resp_limited.headers["X-Request-ID"] == "test-burst-trace-id"

    # Check Retry-After header
    assert "Retry-After" in resp_limited.headers
    retry_after = int(resp_limited.headers["Retry-After"])
    assert retry_after > 0

    # Ensure no tracebacks or internal leaks in response
    assert "traceback" not in resp_limited.text.lower()
    assert "Traceback" not in resp_limited.text
    assert "File \"" not in resp_limited.text


def test_tosql_per_minute_rate_limit_exceeded(client):
    """Verify /v1/tosql enforces 10 req / minute limit when burst limit is satisfied."""
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine

    current_time = 1000.0

    def fake_time():
        return current_time

    with patch("time.time", side_effect=fake_time):
        # Perform 5 pairs of requests separated by 11 seconds (total 10 requests within 60s)
        for pair in range(5):
            current_time = 1000.0 + (pair * 11.0)
            r1 = client.post(
                "/v1/tosql",
                json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
            )
            r2 = client.post(
                "/v1/tosql",
                json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
            )
            assert r1.status_code == 200
            assert r2.status_code == 200

        # 11th request at t=1045s (still within 60s of t=1000s) exceeds 10/min
        current_time = 1045.0
        r_limited = client.post(
            "/v1/tosql",
            json={"question": "Count users", "schema": "CREATE TABLE users (id int);"},
        )
        assert r_limited.status_code == 429
        assert r_limited.json()["error_code"] == ErrorCode.RATE_LIMITED


def test_independent_ip_rate_limit_buckets():
    """Verify different client IP addresses have independent rate limit buckets."""
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine
    if hasattr(app.state, "limiter") and app.state.limiter:
        app.state.limiter.reset()

    client_ip1 = TestClient(app, client=("192.168.1.1", 50000), raise_server_exceptions=False)
    client_ip2 = TestClient(app, client=("192.168.1.2", 50000), raise_server_exceptions=False)

    payload = {"question": "Count users", "schema": "CREATE TABLE users (id int);"}

    # IP 1 uses up its burst allowance (3 requests)
    for _ in range(3):
        assert client_ip1.post("/v1/tosql", json=payload).status_code == 200

    # IP 1 is now rate limited
    assert client_ip1.post("/v1/tosql", json=payload).status_code == 429

    # IP 2 has its own separate bucket and should succeed
    resp_ip2 = client_ip2.post("/v1/tosql", json=payload)
    assert resp_ip2.status_code == 200
    assert resp_ip2.json()["sql"] == "SELECT 1;"


def test_openapi_documents_429_response(client):
    """Verify OpenAPI schema documentation (/openapi.json) documents 429 rate limiting."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    openapi = resp.json()

    tosql_responses = openapi["paths"]["/v1/tosql"]["post"]["responses"]
    assert "200" in tosql_responses
    assert "400" in tosql_responses
    assert "422" in tosql_responses
    assert "429" in tosql_responses
    assert "500" in tosql_responses
    assert "503" in tosql_responses

    desc_429 = tosql_responses["429"]["description"]
    assert "Rate limit exceeded" in desc_429


def test_rate_limit_handler_fallback_retry_after(client):
    """Verify fallback Retry-After when window_stats computation raises."""
    from slowapi.errors import RateLimitExceeded
    from slowapi.wrappers import Limit

    mock_limit = MagicMock(spec=Limit)
    mock_limit.error_message = "Rate limit hit"
    mock_limit.limit = "5/minute"

    with patch.object(app.state.limiter.limiter, "get_window_stats", side_effect=RuntimeError("Storage glitch")):
        @app.get("/test/trigger-rate-limit-exc")
        async def trigger_exc(request: Request):
            request.state.view_rate_limit = ("mock_item", ["key"])
            raise RateLimitExceeded(mock_limit)

        resp = client.get("/test/trigger-rate-limit-exc")
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "10"
        assert resp.json()["error_code"] == ErrorCode.RATE_LIMITED


def test_health_available_even_when_client_is_rate_limited():
    """Verify that an IP that is actively rate limited on /v1/tosql can still access /health."""
    app.state.is_ready = True
    app.state.device = "cpu"
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine
    if hasattr(app.state, "limiter") and app.state.limiter:
        app.state.limiter.reset()

    client_ip = TestClient(app, client=("10.0.0.99", 50000), raise_server_exceptions=False)
    payload = {"question": "Count users", "schema": "CREATE TABLE users (id int);"}

    # Exhaust burst allowance
    for _ in range(3):
        assert client_ip.post("/v1/tosql", json=payload).status_code == 200

    # 4th request is rate limited
    r_blocked = client_ip.post("/v1/tosql", json=payload)
    assert r_blocked.status_code == 429

    # The same IP can still query /health without restriction
    r_health = client_ip.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


def test_docs_and_openapi_endpoints(client):
    """Verify /docs and /openapi.json are accessible and expose the documented API."""
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200
    assert "text/html" in resp_docs.headers["content-type"]

    resp_openapi = client.get("/openapi.json")
    assert resp_openapi.status_code == 200
    openapi = resp_openapi.json()
    assert "/v1/tosql" in openapi["paths"]
    assert "429" in openapi["paths"]["/v1/tosql"]["post"]["responses"]


def test_rate_limit_auto_generated_request_id():
    """Verify that auto-generated request ID is propagated in both body and header on 429."""
    app.state.is_ready = True
    mock_engine = MagicMock()
    mock_engine.generate_sql.return_value = "SELECT 1;"
    app.state.engine = mock_engine
    if hasattr(app.state, "limiter") and app.state.limiter:
        app.state.limiter.reset()

    client_ip = TestClient(app, client=("10.0.0.101", 50000), raise_server_exceptions=False)
    payload = {"question": "Count users", "schema": "CREATE TABLE users (id int);"}

    for _ in range(3):
        client_ip.post("/v1/tosql", json=payload)

    # 4th request without explicit X-Request-ID header
    resp = client_ip.post("/v1/tosql", json=payload)
    assert resp.status_code == 429
    body = resp.json()

    assert "X-Request-ID" in resp.headers
    assert "request_id" in body
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert len(body["request_id"]) >= 32


def test_rate_limit_handler_tuple_window_stats(client):
    """Verify Retry-After is correctly extracted when window_stats is returned as a plain tuple."""
    from slowapi.errors import RateLimitExceeded
    from slowapi.wrappers import Limit

    mock_limit = MagicMock(spec=Limit)
    mock_limit.error_message = "Rate limit hit"
    mock_limit.limit = "3/10second"

    with patch("time.time", return_value=1000.0):
        # Plain tuple (reset_time, remaining) without named attributes
        fake_tuple_stats = (1008.0, 0)
        with patch.object(app.state.limiter.limiter, "get_window_stats", return_value=fake_tuple_stats):
            @app.get("/test/trigger-tuple-stats")
            async def trigger_tuple(request: Request):
                request.state.view_rate_limit = ("mock_item", ["key"])
                raise RateLimitExceeded(mock_limit)

            resp = client.get("/test/trigger-tuple-stats")
            assert resp.status_code == 429
            # 1008.0 - 1000.0 = 8 seconds
            assert resp.headers["Retry-After"] == "8"
            assert resp.json()["error_code"] == ErrorCode.RATE_LIMITED


def test_rate_limit_handler_fallback_when_view_rate_limit_missing(client):
    """Verify rate_limit_handler recovers limit from exc.limit when view_rate_limit is missing from request.state."""
    from limits import parse
    from slowapi.errors import RateLimitExceeded
    from slowapi.wrappers import Limit

    item = parse("10/minute")
    mock_limit = Limit(item, lambda: "testclient", None, False, None, None, None, 1, True)
    # Record hit on testclient so window_stats has active window
    app.state.limiter.limiter.hit(item, "testclient")

    @app.get("/test/trigger-missing-view-rate-limit")
    async def trigger_missing(request: Request):
        raise RateLimitExceeded(mock_limit)

    resp = client.get("/test/trigger-missing-view-rate-limit")
    assert resp.status_code == 429
    retry_after = int(resp.headers["Retry-After"])
    assert 50 <= retry_after <= 60
    assert resp.json()["error_code"] == ErrorCode.RATE_LIMITED




