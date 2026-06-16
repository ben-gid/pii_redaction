from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    api_key: str = ""
    model_id: str = "bengid/pii-redaction-deberta-small"
    threshold: float = 0.3
    
    model_config = SettingsConfigDict(env_file=".env")

class RedactRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Text to redact (max 10k characters)"
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for detected Entities.",
    )
    
    @field_validator("text")
    @classmethod
    def _reject_blanks(cls, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must contain non white-space characters.")
        else:
            return text
        
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
class ModelInfoResponse(BaseModel):
    model_name: str
    model_variant: str
    entity_types: list[str]
    max_length: int