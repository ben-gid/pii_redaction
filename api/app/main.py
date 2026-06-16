from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from pathlib import Path

from .config import settings, _state
from .dependencies import limiter
from .routers import redact, demo, system

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Loading model {settings.model_id} (threshold={settings.threshold:.2f}) ...")
    _state.load(settings)
    app.state.limiter = limiter
    logger.info("Model loaded.")
    yield
    _state.clear()

app = FastAPI(
    title="PII Redaction API",
    description="Detect and redact PII using a fine-tuned DeBERTa-v3 model.",
    version="0.1.0",
    lifespan=lifespan,
)

src_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(src_dir / "static")), name="static")
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.include_router(redact.router)
app.include_router(demo.router)
app.include_router(system.router)