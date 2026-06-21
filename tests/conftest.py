from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app.main import app, settings, state
from pii_redaction.models import RedactionResponse, Entity

@pytest.fixture
def mock_redactor():
    """Mock the PIIRedactor class to avoid loading the real model weights during testing."""
    with patch("api.app.core.config.PIIRedactor") as mock_class:
        # Configure the mock instance
        mock_instance = MagicMock()
        mock_instance.max_length = 512
        mock_instance.id2label = {0: "O", 1: "B-EMAIL", 2: "I-EMAIL", 3: "B-NAME"}

        # Configure predict mock behavior
        def mock_predict(text: str, threshold: float = 0.3):
            # A simple predictable mock response
            entities = []
            redacted = text
            if "john@example.com" in text:
                entities.append(
                    Entity(
                        text="john@example.com",
                        label="EMAIL",
                        start=text.find("john@example.com"),
                        end=text.find("john@example.com") + len("john@example.com"),
                        score=0.99
                    )
                )
                redacted = text.replace("john@example.com", "[EMAIL]")

            return RedactionResponse(
                original=text,
                redacted=redacted,
                entities=entities,
                entity_count=len(entities)
            )

        mock_instance.predict.side_effect = mock_predict
        mock_class.return_value = mock_instance

        yield mock_instance

@pytest.fixture
def client(mock_redactor):
    """Create a TestClient, triggering the app lifespan events
    (which loads our mock redactor)."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def auth_client(client):
    """Client Fixture with api-key preconfigured"""
    original_key = settings.api_key
    settings.api_key = "secret_token_123"
    yield client, "secret_token_123"
    settings.api_key = original_key
    
@pytest.fixture(autouse=True)
def _restore_state_redactor():
    """Snapshot state.redactor before each test and restore it after,
    so tests that mutate state.redactor (e.g. setting it to None)
    can't leak into other tests regardless of fixture scope or order."""
    original = state.redactor
    yield
    state.redactor = original
