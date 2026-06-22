from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from ..dependencies import limiter, run_redaction
from ..models import RedactRequest
from pii_redaction.models import RedactionResponse
from ..core.config import settings

router = APIRouter()
templates = Jinja2Templates(Path(__file__).resolve().parent.parent/ "templates")

@router.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/demo")

@router.get("/demo", response_class=HTMLResponse, description="serve demo html",tags=["Demo"])
async def demo(request: Request):
    return templates.TemplateResponse(request, "index.html", context={"model_id": settings.model_id})


@router.post("/demo/redact", response_model=RedactionResponse, description="redact text", tags=["Demo"])
@limiter.limit("10/day")
async def demo_redact(request: Request, body: RedactRequest):
    request.state.text_length = len(body.text)
    redaction_result = await run_redaction(body.text, body.threshold)
    request.state.entity_count = len(redaction_result.entities)
    return redaction_result