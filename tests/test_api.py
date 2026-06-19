from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app.main import app, settings, limiter, state
from api.app.core.config import AppSettings, AppState
from api.app.models import HealthResponse, ModelInfoResponse
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

def test_settings_defaults():
    s = AppSettings()
    assert s.model_id == "bengid/pii-redaction-deberta-small"
    assert s.threshold == 0.85
    assert s.api_key != "", "expected api-key to be changed to .env variable by default"

def test_settings_env_override():
    with patch.dict("os.environ", {"MODEL_ID": "other-model", "THRESHOLD": "0.9"}):
        s = AppSettings()
        assert s.model_id == "other-model"
        assert s.threshold == 0.9

def test_settings_loads_api_key_from_env():
    with patch.dict("os.environ", {"API_KEY": "sk-test-key-from-env"}):
        s = AppSettings()
        assert s.api_key == "sk-test-key-from-env"

def test_state_starts_empty():
    app_state = AppState()
    assert app_state.redactor is None

def test_state_load(client):
    """Test that model is loaded during lifespan startup via health endpoint."""
    response = client.get("/health")
    data = response.json()
    health = HealthResponse.model_validate(data)
    assert health.model_loaded is True
    assert health.status == "healthy"

def test_health_reflects_state_clear(client):
    """Test that /health reports model_loaded=False after state.clear()."""
    # Verify model is loaded during lifespan startup
    response = client.get("/health")
    health = HealthResponse.model_validate(response.json())
    assert health.model_loaded is True
    
    state.clear()
    assert state.redactor is None

    # Verify health reports model as not loaded after clear
    response = client.get("/health")
    health = HealthResponse.model_validate(response.json())
    assert health.model_loaded is False
    
def test_lifespan_teardown_clears_state(mock_redactor):
    """Test that exiting the TestClient context manager runs lifespan
    shutdown and clears model state."""
    with TestClient(app) as client:
        response = client.get("/health")
        health = HealthResponse.model_validate(response.json())
        assert health.model_loaded is True

    # Context manager exited -> lifespan shutdown actually ran
    assert state.redactor is None

def test_health(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    health = HealthResponse.model_validate(data)
    assert health.status == "healthy"
    assert health.model_loaded is True

def test_health_model_not_loaded(client):
    """Test health endpoint when model is not loaded."""
    state.redactor = None
    response = client.get("/health")
    health = HealthResponse.model_validate(response.json())
    assert health.model_loaded is False

def test_model_info(client):
    """Test the model information endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    info = ModelInfoResponse.model_validate(data)
    assert info.model_name == "DeBERTa-v3 PII Redaction"
    assert "EMAIL" in info.entity_types
    assert "NAME" in info.entity_types
    assert info.max_length == 512

def test_model_info_not_loaded(client):
    """Test model-info returns 503 when model is not loaded."""
    state.redactor = None
    response = client.get("/model-info")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"

def test_index_redirect(client):
    """Test that the index endpoint redirects to the demo page."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/demo"

def test_demo_page(client):
    """Test that the demo HTML page is served."""
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_demo_redact_success(client):
    """Test /demo/redact endpoint without API key (demo endpoint is open, only rate-limited)."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/demo/redact", json=payload)
    assert response.status_code == 200
    data = response.json()
    redaction = RedactionResponse.model_validate(data)
    assert redaction.redacted == "My email is [EMAIL]"
    assert redaction.entity_count == 1
    assert redaction.entities[0].text == "john@example.com"
    assert redaction.entities[0].label == "EMAIL"

def test_redact_endpoint_missing_api_key(client):
    """Test /redact endpoint fails without API key."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/redact", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated" # default msg from fastapi

def test_redact_endpoint_invalid_api_key(client):
    """Test /redact endpoint fails with an invalid API key."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

def test_redact_endpoint_valid_api_key(auth_client):
    """Test /redact endpoint succeeds with a valid API key."""
    client, key = auth_client
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    redaction = RedactionResponse.model_validate(data)
    assert redaction.redacted == "My email is [EMAIL]"

def test_redact_when_redactor_none(auth_client):
    """Test /redact returns 503 when redactor is not loaded."""
    client, key = auth_client
    state.redactor = None
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Redactor not loaded"

def test_demo_redact_rate_limit(client):
    """Test that /demo/redact enforces the 10/day rate limit per IP."""
    limiter._storage.reset()

    test_ip = "203.0.113.42"
    headers = {"X-Forwarded-For": test_ip}
    payload = {"text": "My email is john@example.com", "threshold": 0.3}

    for i in range(10):
        response = client.post("/demo/redact", json=payload, headers=headers)
        assert response.status_code == 200, (
            f"Request {i + 1}/10 should succeed but got {response.status_code}"
        )

    response = client.post("/demo/redact", json=payload, headers=headers)
    assert response.status_code == 429, (
        f"Expected 429 Too Many Requests after limit exhausted, got {response.status_code}"
    )

def test_exceeds_max_length(auth_client):
    """Test text arg exceeds 10k characters returns 422 error"""
    client, key = auth_client
    payload = {"text": "x" * 10_001, "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_empty_text(auth_client):
    """Test empty text returns 422 error"""
    client, key = auth_client
    payload = {"text": "", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_whitespace_only_text(auth_client):
    """Test whitespace-only text returns 422 error (field validator _reject_blanks)"""
    client, key = auth_client
    payload = {"text": "   ", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_missing_text(auth_client):
    """Test missing text arg returns 422 error"""
    client, key = auth_client
    payload = {"threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_missing_threshold(auth_client):
    """Test missing threshold returns ok (200) because threshold has default"""
    client, key = auth_client
    payload = {"text": "x"}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 200

def test_invalid_threshold_negative(auth_client):
    """Test negative threshold returns 422 error"""
    client, key = auth_client
    payload = {"text": "hello", "threshold": -0.1}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_invalid_threshold_above_one(auth_client):
    """Test threshold above 1.0 returns 422 error"""
    client, key = auth_client
    payload = {"text": "hello", "threshold": 1.1}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 422

def test_redact_response_model(auth_client):
    """Test that /redact response validates against RedactionResponse model."""
    client, key = auth_client
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    redaction = RedactionResponse.model_validate(data)
    assert redaction.original == payload["text"]
    assert isinstance(redaction.entities, list)
    assert redaction.entity_count == len(redaction.entities)

def test_demo_redact_response_model(client):
    """Test that /demo/redact response validates against RedactionResponse model."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/demo/redact", json=payload)
    assert response.status_code == 200
    data = response.json()
    redaction = RedactionResponse.model_validate(data)
    assert redaction.original == payload["text"]
    assert redaction.entity_count == len(redaction.entities)
