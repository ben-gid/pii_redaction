"""
PII Redaction API — FastAPI service wrapping the PII redactor.

Endpoints:
    GET  /health       — liveness check
    GET  /model-info   — model metadata and supported entity types
    GET  /             — demo index
    POST /redact       — detect and redact PII in text
    POST /demo/redact  — redact demo input text
    GET  /Api-keys     - admin create api key
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status, Depends, Header 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.redactor import PIIRedactor
from src.models import (
    AppSettings, RedactRequest, HealthResponse, ModelInfoResponse, 
    RedactionResponse,
)

logger = logging.getLogger(__name__)
src_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(src_dir/"templates")

def get_ip(request: Request) -> str:
    # when using a proxy
    if forwarded := request.headers.get("X-Forwarded-For"):
        return forwarded.split(",")[0].strip()
    
    # request.client may be None (e.g., in some ASGI setups), so guard access
    client = request.client
    if client is None:
        return "unknown"
    return getattr(client, "host", "unknown")

limiter = Limiter(key_func=get_ip)

settings = AppSettings()

class AppState:
    redactor: Optional[PIIRedactor] = None
    limiter: Optional[Limiter] = None
    
_state = AppState()

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    model_variant = settings.model_id
    threshold = settings.threshold
    logger.info(f"loading model variant {model_variant} (threshold={threshold:.2f}) ...")
    _state.redactor = PIIRedactor(model_id=model_variant)
    _state.limiter = limiter
    logger.info("model loaded successfully")
    yield
    
app = FastAPI(
    title="PII Redaction API",
    description="Detect and redact personally identifiable information from text "
    "using a fine-tuned DeBERTa-v3 model.",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(src_dir / "static")), name="static")

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore[arg-type]

# optional API key
# TODO    
_API_KEY: Optional[str] = settings.api_key

async def _verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> None:
    if _API_KEY is not None and x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        ) 

async def run_redaction(text: str, threshold: float) -> RedactionResponse:
    if _state.redactor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redactor isn't loaded in app state"
        )
    
    try:
        result = _state.redactor.predict(text, threshold)
    except Exception:
        logger.exception("Redaction failed during inference")
        raise HTTPException(status_code=500, detail="Redaction failed during inference")    
    
    return result

@app.get("/", response_class=HTMLResponse, description="serve demo html", tags=["Demo"])
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
    
    
@app.post("/demo/redact", response_model=RedactionResponse, description="return redacted text", tags=["Demo", "PII Redaction"])
@limiter.limit("10/day")
async def demo_redact(
    request: Request, # fastapi request for slowapi limiter
    body: RedactRequest
):
    return await run_redaction(body.text, body.threshold)
    
    
@app.post("/redact", response_model=RedactionResponse, tags=["PII Redaction"], 
          dependencies=[Depends(_verify_api_key)])
async def redact(
    request: RedactRequest,
):
    return await run_redaction(request.text, request.threshold)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_state.redactor is not None
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
        model_variant="small",
        entity_types=sorted(raw_labels),
        max_length=_state.redactor.max_length,
    )
    
@app.post("/api-keys", tags=[""])
async def create_key():
    raise NotImplementedError()