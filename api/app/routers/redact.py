from fastapi import Request
from fastapi import APIRouter, Depends
from ..dependencies import verify_api_key, run_redaction
from ..models import RedactRequest
from pii_redaction.models import RedactionResponse

router = APIRouter()
   
@router.post("/redact", response_model=RedactionResponse, tags=["API endpoint", "PII Redaction"], 
          dependencies=[Depends(verify_api_key)])
async def redact(request: Request, body: RedactRequest):
    request.state.text_length = len(body.text)
    redaction_result = await run_redaction(body.text, body.threshold)
    request.state.entity_count = len(redaction_result.entities)
    return redaction_result