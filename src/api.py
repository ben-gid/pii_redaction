"""
PII Redaction API — FastAPI service wrapping the PII redactor.

Endpoints:
    GET  /health       — liveness check
    GET  /model-info   — model metadata and supported entity types
    POST /redact       — detect and redact PII in text
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.redactor import PIIRedactor, RedactionResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton app state (model loaded once at startup)
# ---------------------------------------------------------------------------

class AppState:
    redactor: PIIRedactor | None = None


_state = AppState()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RedactRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to redact (max 10k characters).",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for detected entities.",
    )

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace-only")
        return v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_variant: str
    entity_types: list[str]
    max_length: int
    default_threshold: float


# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("PII_API_KEY")


async def _verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    if _API_KEY is not None and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Application lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    model_variant = os.environ.get("PII_MODEL_VARIANT", "small")
    threshold = float(os.environ.get("PII_DEFAULT_THRESHOLD", "0.3"))
    logger.info("Loading model variant '%s' (threshold=%.2f) …", model_variant, threshold)
    _state.redactor = PIIRedactor(model_id=model_variant, threshold=threshold)
    logger.info("Model loaded successfully")
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PII Redaction API",
    description="Detect and redact personally identifiable information from text "
    "using a fine-tuned DeBERTa-v3 model.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns ``healthy`` when the model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=_state.redactor is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the currently loaded model."""
    if _state.redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_labels = set()
    for label in _state.redactor.id2label.values():
        if isinstance(label, str) and label != "O":
            raw_labels.add(label[2:] if label.startswith(("B-", "I-")) else label)

    return ModelInfoResponse(
        model_name="DeBERTa-v3 PII Redaction",
        model_variant=os.environ.get("PII_MODEL_VARIANT", "small"),
        entity_types=sorted(raw_labels),
        max_length=_state.redactor.max_length,
        default_threshold=_state.redactor.threshold,
    )


@app.post("/redact", response_model=RedactionResponse, tags=["PII Redaction"])
async def redact(
    request: RedactRequest,
    _: None = Depends(_verify_api_key),
):
    """Detect PII entities in ``text`` and return a redacted version."""
    if _state.redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    redactor = _state.redactor

    old_threshold = redactor.threshold
    redactor.threshold = request.threshold
    try:
        result = redactor.predict(request.text)
    except Exception:
        logger.exception("Redaction failed during inference")
        raise HTTPException(status_code=500, detail="Redaction failed during inference")
    finally:
        redactor.threshold = old_threshold

    return result


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )
