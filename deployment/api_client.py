"""
FastAPI Client for Text-to-SQL Inference API.

Handles HTTP communication with the Text-to-SQL backend:
- Health checks (/health)
- SQL generation (/v1/tosql)
- Request timeout and error mapping
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional
import requests

logger = logging.getLogger("fastapi_client")


# ---------------------------------------------------------------------------
# Client Exceptions
# ---------------------------------------------------------------------------

class FastAPIClientError(Exception):
    """Base exception for all FastAPI client interactions."""
    pass


class FastAPIUnavailableError(FastAPIClientError):
    """Raised when the FastAPI backend is unreachable or connection is refused."""
    pass


class ModelNotReadyError(FastAPIClientError):
    """Raised when the backend reports model is still loading or degraded."""
    pass


class InferenceBusyError(FastAPIClientError):
    """Raised when the backend queue is saturated or times out (503 SERVICE_BUSY)."""
    pass


class InferenceFailedError(FastAPIClientError):
    """Raised when inference execution fails on the server (500)."""
    pass


class RequestValidationError(FastAPIClientError):
    """Raised when request parameters fail client or server-side validation."""
    pass


class RateLimitExceededError(FastAPIClientError):
    """Raised when the backend returns 429 Too Many Requests (rate limit exceeded)."""
    pass


# ---------------------------------------------------------------------------
# Client Implementation
# ---------------------------------------------------------------------------

class FastAPIClient:
    """
    Dedicated client abstraction for the FastAPI Text-to-SQL service.

    Attributes:
        base_url: Base URL of the FastAPI server (e.g. "http://127.0.0.1:8000").
        timeout: Request timeout in seconds for inference calls.
        connect_timeout: Network connection timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 65.0,
        connect_timeout: float = 5.0,
    ) -> None:
        raw_url = base_url or os.getenv("FASTAPI_URL") or "http://127.0.0.1:8000"
        self.base_url = raw_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout

        self.session = requests.Session()
        # Prevent localhost / loopback requests from routing through corporate / sandbox HTTP proxies
        self.session.trust_env = False

    def health(self) -> dict[str, Any]:
        """
        Query GET /health on the FastAPI server.

        Returns:
            Dict containing:
                - status: 'healthy', 'degraded', or 'offline'
                - model_version: string identifier of the model
                - uptime_seconds: float server uptime
                - device: string compute device ('cuda', 'mps', 'cpu', or 'none')
                - error: error string if degraded or offline, None otherwise
        """
        url = f"{self.base_url}/health"
        try:
            resp = self.session.get(
                url,
                timeout=(min(self.connect_timeout, 2.0), 3.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": data.get("status", "healthy"),
                    "model_version": data.get("model_version", "unknown"),
                    "uptime_seconds": float(data.get("uptime_seconds", 0.0)),
                    "device": data.get("device", "none"),
                    "error": data.get("error"),
                }
            else:
                return {
                    "status": "degraded",
                    "model_version": "unknown",
                    "uptime_seconds": 0.0,
                    "device": "none",
                    "error": f"Health check returned HTTP {resp.status_code}",
                }
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning(f"FastAPI health check failed ({url}): {exc}")
            return {
                "status": "offline",
                "model_version": "unknown",
                "uptime_seconds": 0.0,
                "device": "none",
                "error": f"Cannot reach FastAPI server at {self.base_url}. Ensure the service is running.",
            }
        except Exception as exc:
            logger.error(f"Unexpected error querying health ({url}): {exc}")
            return {
                "status": "offline",
                "model_version": "unknown",
                "uptime_seconds": 0.0,
                "device": "none",
                "error": f"Health check failed: {exc}",
            }

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
    ) -> dict[str, Any]:
        """
        Send a natural language question and database schema to POST /v1/tosql.

        Args:
            question: Natural language question about the database.
            schema: Database schema / DDL representation.
            dialect: Target database SQL dialect (e.g. 'sqlite', 'postgresql', 'mysql').

        Returns:
            Dict containing:
                - request_id: Unique tracking UUID
                - sql: Generated executable SQL query
                - model: Model identifier string
                - generation_time_ms: Inference execution duration in milliseconds

        Raises:
            RequestValidationError: If question or schema does not meet validation constraints.
            FastAPIUnavailableError: If the server cannot be reached.
            ModelNotReadyError: If the model failed to load or is not ready.
            InferenceBusyError: If the inference queue wait limit was exceeded.
            InferenceFailedError: If generation failed inside the model.
            FastAPIClientError: For any other unexpected error status.
        """
        cleaned_question = question.strip() if question else ""
        cleaned_schema = schema.strip() if schema else ""
        cleaned_dialect = (dialect or "sqlite").strip().lower()

        if not cleaned_question:
            raise RequestValidationError("Question cannot be empty.")
        if len(cleaned_question) < 3:
            raise RequestValidationError("Question must be at least 3 characters long.")
        if len(cleaned_question) > 2000:
            raise RequestValidationError("Question exceeds maximum allowed length of 2000 characters.")

        if not cleaned_schema:
            raise RequestValidationError("Database schema cannot be empty. Please connect to a database first.")
        if len(cleaned_schema) < 5:
            raise RequestValidationError("Database schema must contain at least 5 characters.")
        if len(cleaned_schema) > 50000:
            raise RequestValidationError("Database schema exceeds maximum allowed length of 50000 characters.")

        url = f"{self.base_url}/v1/tosql"
        payload = {
            "question": cleaned_question,
            "schema": cleaned_schema,
            "dialect": cleaned_dialect,
        }

        try:
            resp = self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(self.connect_timeout, self.timeout),
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.error(f"Inference request failed to reach {url}: {exc}")
            raise FastAPIUnavailableError(
                f"FastAPI service is unavailable at {self.base_url}. Please verify the server is running."
            ) from exc
        except requests.RequestException as exc:
            logger.error(f"HTTP error communicating with {url}: {exc}")
            raise FastAPIClientError(f"HTTP communication failed: {exc}") from exc

        # Success path
        if resp.status_code == 200:
            try:
                data = resp.json()
                if "sql" not in data or "request_id" not in data:
                    raise FastAPIClientError(f"Malformed response payload from /v1/tosql: {data}")
                return data
            except Exception as exc:
                raise FastAPIClientError(f"Failed to decode response JSON: {exc}") from exc

        # Process server error responses
        try:
            err_body = resp.json()
            err_code = err_body.get("error_code", "")
            err_msg = err_body.get("message", resp.text)
        except Exception:
            err_code = ""
            err_msg = resp.text

        if resp.status_code == 422:
            raise RequestValidationError(f"Validation Error: {err_msg}")
        elif resp.status_code == 429:
            raise RateLimitExceededError(f"Rate limit exceeded: {err_msg}")
        elif resp.status_code == 503:
            if err_code == "MODEL_NOT_READY":
                raise ModelNotReadyError(f"Model not ready: {err_msg}")
            elif err_code == "SERVICE_BUSY":
                raise InferenceBusyError(f"Inference queue saturated: {err_msg}")
            raise FastAPIUnavailableError(f"Service unavailable: {err_msg}")
        elif resp.status_code == 500:
            raise InferenceFailedError(f"Model generation failed on server: {err_msg}")
        elif resp.status_code == 400:
            raise RequestValidationError(f"Bad request: {err_msg}")
        else:
            raise FastAPIClientError(f"Server returned HTTP {resp.status_code}: {err_msg}")
