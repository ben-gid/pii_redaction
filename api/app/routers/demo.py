from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from ..dependencies import limiter, run_redaction
from ..models import RedactRequest
from pii_redaction.models import RedactionResponse

router = APIRouter()
templates = Jinja2Templates(Path(__file__).resolve().parent.parent/ "templates")

@router.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/demo")

@router.get("/demo", response_class=HTMLResponse, description="serve demo html",tags=["Demo"])
async def demo(request: Request):
    return templates.TemplateResponse(request, "index.html")

@router.post("/demo/redact", response_model=RedactionResponse, description="redact text", tags=["Demo"])
@limiter.limit("10/day")
async def demo_redact(request: Request, body: RedactRequest):
    return await run_redaction(body.text, body.threshold)