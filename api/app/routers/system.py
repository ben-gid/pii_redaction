from fastapi import APIRouter, HTTPException
from ..config import _state
from ..models import HealthResponse, ModelInfoResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_state.redactor is not None
    )

@router.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    if _state.redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    raw_labels = {
        label[2:] if label.startswith(("B-", "I-")) else label
        for label in _state.redactor.id2label.values()
        if isinstance(label, str) and label != "O"
    }
    
    return ModelInfoResponse(
        model_name="DeBERTa-v3 PII Redaction",
        model_variant="small",
        entity_types=sorted(raw_labels),
        max_length=_state.redactor.max_length,
    )