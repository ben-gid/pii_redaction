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

class Entity(BaseModel):
    """A single detected PII entity.

    Attributes:
        text: The original substring that was identified as PII.
        label: The entity type label (e.g. ``EMAIL``, ``SOCIALNUMBER``).
        start: Character offset where the entity begins (inclusive).
        end: Character offset where the entity ends (exclusive).
        score: Model confidence score in [0, 1].
    """

    text: str
    label: str
    start: int
    end: int
    score: float


class RedactionResponse(BaseModel):
    """Full result of a redaction pass.

    Attributes:
        original: The input text, verbatim.
        redacted: The input text with every detected PII span replaced by a
            typed placeholder (e.g. ``[EMAIL]``).
        entities: List of all detected :class:`Entity` objects, sorted by
            start position.
        entity_count: Convenience shortcut for ``len(entities)``.
    """

    original: str
    redacted: str
    entities: list[Entity]
    entity_count: int
        
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    
class ModelInfoResponse(BaseModel):
    model_name: str
    model_variant: str
    entity_types: list[str]
    max_length: int