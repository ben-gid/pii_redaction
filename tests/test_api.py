from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app.main import app, settings, limiter
from pii_redaction.models import RedactionResponse, Entity

@pytest.fixture
def mock_redactor():
    """Mock the PIIRedactor class to avoid loading the real model weights during testing."""
    with patch("api.app.config.PIIRedactor") as mock_class:
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
    """Create a TestClient, triggering the app lifespan events (which loads our mock redactor)."""
    with TestClient(app) as test_client:
        yield test_client
        
@pytest.fixture
def auth_client(client):
    """Client Fixture with api-key preconfigured"""
    original_key = settings.api_key
    settings.api_key = "secret_token_123"
    yield client, "secret_token_123"
    settings.api_key = original_key

def test_health(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_model_info(client):
    """Test the model information endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "DeBERTa-v3 PII Redaction"
    assert "EMAIL" in data["entity_types"]
    assert "NAME" in data["entity_types"]

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
    assert data["redacted"] == "My email is [EMAIL]"
    assert data["entity_count"] == 1
    assert data["entities"][0]["text"] == "john@example.com"

def test_redact_endpoint_missing_api_key(client):
    """Test /redact endpoint fails without API key."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/redact", json=payload)
    assert response.status_code == 401  # APIKeyHeader returns 401 when the header is absent

def test_redact_endpoint_invalid_api_key(client):
    """Test /redact endpoint fails with an invalid API key."""
    key = "invalid_key"
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_redact_endpoint_valid_api_key(auth_client):
    """Test /redact endpoint succeeds with a valid API key."""
    client, key = auth_client
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["redacted"] == "My email is [EMAIL]"

def test_demo_redact_rate_limit(client):
    """Test that /demo/redact enforces the 10/day rate limit per IP.

    Strategy:
    - Use a unique X-Forwarded-For IP so this test doesn't share a counter
      with other tests (they use the default TestClient IP).
    - Reset the limiter's in-memory storage beforehand so the counter starts
      at zero even if the test suite is re-run within the same day.
    - Exhaust all 10 allowed requests, then assert the 11th gets a 429.
    """
    # Reset limiter storage so re-runs don't inherit a stale counter.
    limiter._storage.reset()

    # A dedicated test IP — won't collide with other tests.
    test_ip = "203.0.113.42"  # RFC 5737 TEST-NET-3, safe to use in tests
    headers = {"X-Forwarded-For": test_ip}
    payload = {"text": "My email is john@example.com", "threshold": 0.3}

    # Exhaust all 10 allowed requests — each should succeed.
    for i in range(10):
        response = client.post("/demo/redact", json=payload, headers=headers)
        assert response.status_code == 200, (
            f"Request {i + 1}/10 should succeed but got {response.status_code}"
        )

    # The 11th request from the same IP must be rejected.
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