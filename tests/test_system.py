from api.app.main import state
from api.app.models import HealthResponse, ModelInfoResponse

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
