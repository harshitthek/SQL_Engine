from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
import sys
import time
from typing import Any, Optional
import uuid
import warnings

# Suppress Pydantic warning about field 'schema' shadowing deprecated BaseModel.schema
warnings.filterwarnings("ignore", message=".*shadows an attribute in parent.*")

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("text2sql_api")

# Model & Serving Constants
MODEL_VERSION: str = "text2sql-v1"
SERVER_MAX_NEW_TOKENS: int = 256
SERVER_TEMPERATURE: float = 0.0
INFERENCE_QUEUE_TIMEOUT_SECONDS: float = 30.0

# Rate Limiting Configuration
# Limits protect expensive Text-to-SQL GPU inference from abuse.
# Defaults: 10 requests per minute and 3 requests per 10 seconds per IP.
RATE_LIMIT_PER_MINUTE: str = os.getenv("RATE_LIMIT_PER_MINUTE", "10/minute")
RATE_LIMIT_BURST: str = os.getenv("RATE_LIMIT_BURST", "3/10seconds")


# ---------------------------------------------------------------------------
# Machine-Readable Error Codes
# ---------------------------------------------------------------------------

class ErrorCode:
    MODEL_NOT_READY = "MODEL_NOT_READY"
    SERVICE_BUSY = "SERVICE_BUSY"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INVALID_REQUEST = "INVALID_REQUEST"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    RATE_LIMITED = "RATE_LIMITED"


# ---------------------------------------------------------------------------
# Pydantic Request & Response Models
# ---------------------------------------------------------------------------

class ToSQLRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "question": "What is the total revenue for the year 2024?",
                "schema": "CREATE TABLE orders (id INT PRIMARY KEY, amount FLOAT, order_date DATE);",
            }
        },
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to translate into SQL. Must not be empty or blank.",
    )
    schema: str = Field(
        ...,
        min_length=5,
        max_length=50000,
        description="Database schema (DDL statements) providing table and column definitions.",
    )
    dialect: Optional[str] = Field(
        default="sqlite",
        description="Target database SQL dialect ('sqlite', 'postgresql', 'mysql', etc.). Defaults to 'sqlite'.",
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return cleaned

    @field_validator("schema")
    @classmethod
    def validate_schema_non_empty(cls, v: str) -> str:
        """Validate schema contains meaningful, non-whitespace characters."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Schema cannot be empty or contain only whitespace.")
        return cleaned


class ToSQLResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "8c7a6e12-b13c-4a37-b12e-fa68903c72b1",
                "sql": "SELECT SUM(amount) FROM orders WHERE strftime('%Y', order_date) = '2024';",
                "model": "text2sql-v1",
                "generation_time_ms": 183.42,
            }
        }
    )

    request_id: str = Field(
        ...,
        description="Unique request tracing identifier (UUID4).",
    )
    sql: str = Field(
        ...,
        description="Generated executable SQL query.",
    )
    model: str = Field(
        ...,
        description="Name/version of the Text-to-SQL model used for inference.",
    )
    generation_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference execution duration in milliseconds, rounded to 2 decimal places.",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_version": "text2sql-v1",
                "uptime_seconds": 3612.45,
                "device": "mps",
            }
        }
    )

    status: str = Field(
        ...,
        description="Health status of the service ('healthy' when ready for inference, 'degraded' if model failed to load).",
    )
    model_version: str = Field(
        ...,
        description="Version string of the Text-to-SQL model.",
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed server uptime in seconds.",
    )
    device: str = Field(
        ...,
        description="Hardware accelerator device hosting the model ('cuda', 'mps', 'cpu', or 'none').",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error details if the service is in degraded status.",
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "MODEL_NOT_READY",
                "message": "Text-to-SQL model is not loaded or currently unavailable.",
                "request_id": "8c7a6e12-b13c-4a37-b12e-fa68903c72b1",
            }
        }
    )

    error_code: str = Field(
        ...,
        description="Machine-readable error classification code.",
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of what caused the failure.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for log tracing.",
    )


# ---------------------------------------------------------------------------
# Application Lifespan & Singleton Initialization
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Initializing FastAPI Text-to-SQL Inference Gateway")
    logger.info("CONCURRENCY NOTE: Process-local asyncio.Lock is in effect.")
    logger.info("CRITICAL DEPLOYMENT RULE: Run with exactly 1 worker (--workers 1).")
    logger.info("=" * 60)

    app.state.start_time = time.time()
    app.state.inference_lock = asyncio.Lock()
    app.state.engine = None
    app.state.is_ready = False
    app.state.device = "none"
    app.state.startup_error = None

    try:
        from prediction import Text2SQLEngine
        logger.info("Loading Text2SQLEngine singleton weights into memory...")
        
        # Offload heavy synchronous model download & weight loading to thread
        engine = await asyncio.to_thread(Text2SQLEngine)
        app.state.engine = engine
        app.state.is_ready = True
        app.state.device = getattr(engine, "device", "unknown")
        app.state.startup_error = None
        logger.info(f"Text2SQLEngine loaded successfully on target device: '{app.state.device}'.")
    except Exception as exc:
        logger.critical(
            f"Failed to load Text2SQLEngine on startup: {exc}. "
            f"Service entering DEGRADED mode (/health=degraded, /v1/tosql=503).",
            exc_info=True,
        )
        app.state.engine = None
        app.state.is_ready = False
        app.state.device = "none"
        app.state.startup_error = str(exc)

    yield

    logger.info("Shutting down Text-to-SQL service and releasing resources...")
    app.state.engine = None
    app.state.is_ready = False

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    logger.info("Shutdown sequence finished cleanly.")


# ---------------------------------------------------------------------------
# Rate Limiter & FastAPI Application Declaration
# ---------------------------------------------------------------------------

# Concurrency & Deployment Architecture:
# The application is designed to run as a single FastAPI worker process because the Text-to-SQL
# model occupies dedicated hardware accelerator (GPU / MPS) memory. The in-memory Limiter is
# appropriate for this single-process deployment without external infrastructure.
# NOTE: If this service is horizontally scaled across multiple worker processes or replica containers,
# the rate limiter storage must be migrated to a shared backend (e.g. Redis via storage_uri)
# or enforced at an edge reverse proxy / API gateway layer (e.g. Nginx, Cloudflare).
#
# Client IP / Reverse Proxy Handling:
# get_remote_address inspects request.client.host. Arbitrary client-supplied headers like
# X-Forwarded-For or X-Real-IP are NOT blindly trusted to prevent IP spoofing attacks.
# When deployed behind a reverse proxy, the proxy must be configured to pass the real client IP,
# and Uvicorn should be run with trusted proxy forwarding enabled (--proxy-headers).
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Text-to-SQL Inference API",
    description=(
        "Production-grade Text-to-SQL model serving API with serialized hardware "
        "inference, bounded queue protection, and health monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: Request ID Correlation
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Extracts or generates an X-Request-ID for distributed tracing.
    Stores the ID in request.state and propagates it in the response header.
    """
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Formats slowapi RateLimitExceeded exceptions into standard ErrorResponse structure.
    Returns HTTP 429 Too Many Requests with correlation ID and Retry-After header.
    """
    req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    client_ip = get_remote_address(request)
    logger.warning(
        f"[{req_id}] rate_limit_exceeded ip={client_ip} endpoint={request.url.path}"
    )

    retry_after_seconds = 10
    try:
        # Determine fallback window duration from the exceeded limit if possible
        if hasattr(exc, "limit") and exc.limit and hasattr(exc.limit, "limit"):
            rate_item = exc.limit.limit
            if hasattr(rate_item, "get_expiry"):
                retry_after_seconds = max(1, int(rate_item.get_expiry()))

        view_rate_limit = getattr(request.state, "view_rate_limit", None)
        l = getattr(request.app.state, "limiter", None)
        if l:
            if not view_rate_limit and hasattr(exc, "limit") and exc.limit:
                view_rate_limit = (exc.limit.limit, [client_ip])

            if view_rate_limit:
                window_stats = l.limiter.get_window_stats(view_rate_limit[0], *view_rate_limit[1])
                reset_time = getattr(window_stats, "reset_time", None)
                if reset_time is None and isinstance(window_stats, (tuple, list)) and len(window_stats) > 0:
                    reset_time = window_stats[0]
                if reset_time is not None:
                    retry_after_seconds = max(1, math.ceil(reset_time - time.time()))
    except Exception:
        pass

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        headers={
            "X-Request-ID": req_id,
            "Retry-After": str(retry_after_seconds),
        },
        content={
            "error_code": ErrorCode.RATE_LIMITED,
            "message": "Too many requests from this IP. Please try again later.",
            "request_id": req_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats Pydantic 422 validation errors into standard ErrorResponse format."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    error_messages = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error.get("loc", []))
        message = error.get("msg", "Invalid field")
        error_messages.append(f"{location}: {message}")

    joined_message = "; ".join(error_messages) if error_messages else "Request validation failed."
    logger.warning(f"[{req_id}] 422 Validation Error: {joined_message}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        headers={"X-Request-ID": req_id},
        content={
            "error_code": ErrorCode.VALIDATION_ERROR,
            "message": joined_message,
            "request_id": req_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Formats HTTPExceptions into standardized ErrorResponse structure."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        payload = {
            "error_code": exc.detail.get("error_code", ErrorCode.INVALID_REQUEST),
            "message": exc.detail.get("message", "An error occurred."),
            "request_id": req_id,
        }
    else:
        status_to_code = {
            status.HTTP_400_BAD_REQUEST: ErrorCode.INVALID_REQUEST,
            status.HTTP_429_TOO_MANY_REQUESTS: ErrorCode.RATE_LIMITED,
            status.HTTP_503_SERVICE_UNAVAILABLE: ErrorCode.MODEL_NOT_READY,
            status.HTTP_500_INTERNAL_SERVER_ERROR: ErrorCode.INTERNAL_SERVER_ERROR,
        }
        error_code = status_to_code.get(exc.status_code, "HTTP_ERROR")
        payload = {
            "error_code": error_code,
            "message": str(exc.detail),
            "request_id": req_id,
        }

    return JSONResponse(
        status_code=exc.status_code,
        headers={"X-Request-ID": req_id},
        content=payload,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server exceptions, logs trace, and returns 500 ErrorResponse."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error(f"[{req_id}] Unhandled server exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"X-Request-ID": req_id},
        content={
            "error_code": ErrorCode.INTERNAL_SERVER_ERROR,
            "message": "An unexpected internal server error occurred.",
            "request_id": req_id,
        },
    )


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health & Hardware Status",
    tags=["System"],
    responses={
        200: {"model": HealthResponse, "description": "Service health and device report."},
    },
)
async def health(request: Request) -> HealthResponse:
    """
    Check the health of the Text-to-SQL inference server.

    Returns:
        - status: 'healthy' if model singleton is loaded and ready; 'degraded' otherwise.
        - model_version: Identifier of the deployed model.
        - uptime_seconds: Server uptime since startup.
        - device: Compute accelerator ('cuda', 'mps', 'cpu', or 'none').
    """
    uptime = round(time.time() - getattr(request.app.state, "start_time", time.time()), 2)
    is_ready = getattr(request.app.state, "is_ready", False)
    device = getattr(request.app.state, "device", "none")
    startup_error = getattr(request.app.state, "startup_error", None)

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_version=MODEL_VERSION,
        uptime_seconds=uptime,
        device=device,
        error=startup_error if not is_ready else None,
    )


@app.post(
    "/v1/tosql",
    response_model=ToSQLResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate SQL from Natural Language",
    tags=["Inference"],
    responses={
        200: {"model": ToSQLResponse, "description": "Successfully generated SQL."},
        400: {"model": ErrorResponse, "description": "Bad or malformed request payload."},
        422: {"model": ErrorResponse, "description": "Field validation failure on input."},
        429: {
            "model": ErrorResponse,
            "description": "Rate limit exceeded (per-IP limits: 10 requests/minute, 3 requests/10 seconds burst).",
        },
        503: {
            "model": ErrorResponse,
            "description": "Service unavailable (model not ready or inference queue timeout).",
        },
        500: {"model": ErrorResponse, "description": "Unexpected inference failure on server."},
    },
)
@limiter.limit(RATE_LIMIT_PER_MINUTE)
@limiter.limit(RATE_LIMIT_BURST)
async def tosql(
    payload: ToSQLRequest,
    request: Request,
) -> ToSQLResponse:
    """
    Translates a natural language question into an executable SQL query given a schema.

    - **Rate Limiting**: Enforces per-client-IP rate limits (10/min, 3/10s burst) to prevent hardware abuse.
    - **Inference Lock**: Serializes hardware accelerator access to prevent GPU OOM crashes.
    - **Queue Timeout**: Rejects requests if waiting for the lock exceeds 30 seconds with 503 SERVICE_BUSY.
    - **Model Readiness**: Immediately returns 503 MODEL_NOT_READY if model failed to load.
    - **Execution Limits**: Strict server-side parameters (max_new_tokens=256, temperature=0.0).
    """
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # 1. Verify model readiness (fail fast if degraded)
    if not getattr(request.app.state, "is_ready", False) or request.app.state.engine is None:
        startup_err = getattr(request.app.state, "startup_error", None)
        err_msg = "Text-to-SQL model is not loaded or currently unavailable."
        if startup_err:
            err_msg += f" (Failure reason: {startup_err})"
        logger.warning(f"[{req_id}] Inference rejected: {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": ErrorCode.MODEL_NOT_READY,
                "message": err_msg,
            },
        )

    # 2. Acquire lock with bounded wait protection
    lock: asyncio.Lock = request.app.state.inference_lock
    try:
        await asyncio.wait_for(
            lock.acquire(),
            timeout=INFERENCE_QUEUE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[{req_id}] Inference queue wait exceeded {INFERENCE_QUEUE_TIMEOUT_SECONDS}s limit. Server busy."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": ErrorCode.SERVICE_BUSY,
                "message": (
                    f"Inference queue timeout: server is currently saturated "
                    f"({INFERENCE_QUEUE_TIMEOUT_SECONDS}s wait limit exceeded)."
                ),
            },
        )

    # 3. Perform synchronized inference offloaded to worker thread
    try:
        t_start = time.perf_counter()

        sql = await asyncio.to_thread(
            request.app.state.engine.generate_sql,
            question=payload.question,
            schema=payload.schema,
            dialect=payload.dialect or "sqlite",
            max_new_tokens=SERVER_MAX_NEW_TOKENS,
            temperature=SERVER_TEMPERATURE,
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.info(f"[{req_id}] SQL generated in {elapsed_ms}ms (model={MODEL_VERSION})")

        return ToSQLResponse(
            request_id=req_id,
            sql=sql,
            model=MODEL_VERSION,
            generation_time_ms=elapsed_ms,
        )

    except Exception as exc:
        logger.error(f"[{req_id}] Model inference failed during generation: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": ErrorCode.INFERENCE_FAILED,
                "message": f"Inference execution failed: {str(exc)}",
            },
        )
    finally:
        # Guarantee lock is released for subsequent queued requests
        lock.release()


@app.post(
    "/v1/reload",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Reload Model Weights",
    tags=["System"],
)
async def reload_model(request: Request) -> HealthResponse:
    """
    Trigger dynamic re-initialization of the Text2SQLEngine model.
    Allows recovering from degraded mode without restarting the server process.

    Protected by RELOAD_SECRET env var when set. Requests must include
    a matching X-Reload-Secret header.
    """
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    # Optional secret-based protection for production environments
    reload_secret = os.getenv("RELOAD_SECRET")
    if reload_secret:
        provided = request.headers.get("X-Reload-Secret", "")
        if provided != reload_secret:
            logger.warning(f"[{req_id}] Reload rejected: invalid or missing X-Reload-Secret.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "FORBIDDEN",
                    "message": "Invalid or missing reload secret.",
                },
            )

    logger.info(f"[{req_id}] Manual model reload requested.")
    lock: Optional[asyncio.Lock] = getattr(request.app.state, "inference_lock", None)
    if lock:
        async with lock:
            try:
                # Release existing engine and clear hardware accelerator cache before reloading
                request.app.state.engine = None
                request.app.state.is_ready = False
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

                from prediction import Text2SQLEngine
                engine = await asyncio.to_thread(Text2SQLEngine)
                request.app.state.engine = engine
                request.app.state.is_ready = True
                request.app.state.device = getattr(engine, "device", "unknown")
                request.app.state.startup_error = None
                logger.info(f"[{req_id}] Model reloaded successfully on device: '{request.app.state.device}'.")
            except Exception as exc:
                logger.error(f"[{req_id}] Failed to reload model: {exc}", exc_info=True)
                request.app.state.engine = None
                request.app.state.is_ready = False
                request.app.state.device = "none"
                request.app.state.startup_error = str(exc)

    return await health(request)


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    logger.info(f"Starting FastAPI Inference Gateway on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False, workers=1)