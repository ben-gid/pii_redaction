from fastapi import APIRouter, Depends
from ..dependencies import verify_api_key, run_redaction
from ..models import RedactRequest
from pii_redaction.models import RedactionResponse

router = APIRouter()
   
@router.post("/redact", response_model=RedactionResponse, tags=["API endpoint", "PII Redaction"], 
          dependencies=[Depends(verify_api_key)])
async def redact(request: RedactRequest):
    return await run_redaction(request.text, request.threshold)