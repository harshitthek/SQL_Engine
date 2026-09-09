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


# --------------------------------------------------------------------