"""
Logging setup for the PII Redaction API.
 
Key design choices:
  - JSON output to stdout (ECS Fargate ships container stdout to CloudWatch
    for free — no file handlers needed, and file logging inside a container
    just disappears when the task recycles).
  - Privacy First: Never log raw input text, detected entity values, or raw 
    IP addresses. IPs are hashed (SHA-256) for anonymous users. Only metadata
    (lengths, counts, timings) is logged so the logs themselves can't leak PII.
  - State Aggregation (One Log Per Request): Routers don't log directly. Instead, 
    they attach metadata to `request.state`, and the middleware aggregates everything 
    into a single structured JSON log line when the request completes.
  - uvicorn's own loggers ("uvicorn", "uvicorn.error", "uvicorn.access")
    are pointed at the same JSON handler instead of their default colored
    text formatter, so every line in CloudWatch has the same shape.
"""
import json
import logging
import logging.config
import os
import time
from typing import Awaitable, Callable, Optional
import uuid
import hashlib
from datetime import datetime, timezone

from starlette.datastructures import State
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter, safe for CloudWatch ingestion."""
 
    # Whitelist of extra fields allowed through. Anything not listed here
    # is dropped, so a stray `logger.info(text)` elsewhere in the codebase
    # can't accidentally smuggle PII into `extra=`.
    ALLOWED_EXTRA = (
        "request_id", "client_id", "endpoint", "method", "status_code",
        "duration_ms", "text_length", "entity_count",
    )
    
    def format(self, record:logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in self.ALLOWED_EXTRA:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)
    
    
def build_logging_config(level: Optional[str] = None) -> dict:
    level = level or os.getenv("LOG_LEVEL", "INFO")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JSONFormatter}},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "pii_redaction_api": {"handlers": ["default"], "level": level, "propagate": False},
            # Override uvicorn's own loggers instead of letting it install
            # its default text formatters.
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
            # access logs are noisy and duplicate what our middleware logs
            # with more detail (request_id, duration) - keep at WARNING.
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
        "root": {"handlers": ["default"], "level": level},
    }


_configured = False

def setup_logging(level: Optional[str] = None) -> logging.Logger:
    global _configured
    if not _configured:
        logging.config.dictConfig(build_logging_config(level))
        _configured = True
    return logging.getLogger("pii_redaction_api")

logger = logging.getLogger("pii_redaction_api")


def _mask_api_key(raw_key: Optional[str]=None, ip_address: Optional[str]=None) -> str:
    if not raw_key:
        if ip_address:
            # Hash the IP address to maintain privacy (first 8 chars of SHA256)
            return f"ip_{hashlib.sha256(ip_address.encode()).hexdigest()[:8]}"
        return "anonymous" # fallback if neither is available
    return f"{raw_key[:4]}...{raw_key[-4:]}"

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request metadata only - never the request/response body (for privacy)"""
    async def dispatch(
        self, 
        request: Request[State], 
        call_next: Callable[[Request[State]], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        client_ip = request.client.host if request.client else None
        client_id = _mask_api_key(request.headers.get("X-API-Key", ""), client_ip)
        
        try:
            response = await call_next(request) # route handler
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "client_id": client_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                },
            )
            raise
        
        duration_ms = round((time.perf_counter() - start) * 1_000, 2)
        
        # Gather base metadata
        extra_data = {
            "request_id": request_id,
            "client_id": client_id,
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        
        # Pull any extra metric data added by the routers
        if hasattr(request.state, "text_length"):
            extra_data["text_length"] = request.state.text_length
        if hasattr(request.state, "entity_count"):
            extra_data["entity_count"] = request.state.entity_count
            
        logger.info(
            "request_completed",
            extra=extra_data,
        )
        response.headers["X-Request-ID"] = request_id
        return response