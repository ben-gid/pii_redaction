from unittest.mock import patch
from fastapi.testclient import TestClient

from api.app.main import app, state
from api.app.core.config import AppSettings, AppState
from api.app.models import HealthResponse

def test_settings_defaults():
    s = AppSettings()
    assert s.model_id == "bengid/pii-redaction-deberta-small"
    assert s.threshold == 0.85

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

def test_lifespan_teardown_clears_state(mock_redactor):
    """Test that exiting the TestClient context manager runs lifespan
    shutdown and clears model state."""
    with TestClient(app) as client:
        response = client.get("/health")
        health = HealthResponse.model_validate(response.json())
        assert health.model_loaded is True

    # Context manager exited -> lifespan shutdown actually ran
    assert state.redactor is None
