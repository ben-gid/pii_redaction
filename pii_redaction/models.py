from pydantic import BaseModel

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