import os
import pytest
from api.app.main import state
from pii_redaction.models import RedactionResponse

def test_redact_endpoint_missing_api_key(client):
    """Test /redact endpoint fails without API key."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/redact", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_redact_endpoint_invalid_api_key(client):
    """Test /redact endpoint fails with an invalid API key."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"

def test_redact_when_redactor_none(auth_client):
    """Test /redact returns 503 when redactor is not loaded."""
    client, key = auth_client
    state.redactor = None
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Redactor not loaded"

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

def test_redact_with_env_api_key(client):
    """Test /redact endpoint using the real API_KEY from os.environ."""
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise KeyError("os environ 'api_key' variable not set,"
                       "make sure load_dotenv is included in config.py")
        
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    headers = {"X-API-Key": api_key}
    response = client.post("/redact", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["redacted"] == "My email is [EMAIL]"
