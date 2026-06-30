from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from pathlib import Path

import uvicorn

from .core.config import settings, state
from .core.logging_setup import build_logging_config, setup_logging, RequestLoggingMiddleware
from .dependencies import limiter, fetch_and_write_cert
from .routers import redact, demo, system

logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Loading model {settings.model_id} (threshold={settings.threshold:.2f}) ...")
    state.load(settings)
    app.state.limiter = limiter
    logger.info("Model loaded.")
    yield
    state.clear()

app = FastAPI(
    title="PII Redaction API",
    description="Detect and redact PII using a fine-tuned DeBERTa-v3 model.",
    version="0.1.0",
    lifespan=lifespan,
)

src_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(src_dir / "static")), name="static")
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(RequestLoggingMiddleware)

app.include_router(redact.router)
app.include_router(demo.router)
app.include_router(system.router)

if __name__ == "__main__":
    if os.environ.get("ENV") == "production":
        cert_path, key_path = fetch_and_write_cert()
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            ssl_certfile=cert_path, 
            ssl_keyfile=key_path
        )
        
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_config=build_logging_config(),
        )