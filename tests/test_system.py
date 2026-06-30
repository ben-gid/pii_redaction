from api.app.main import state
from api.app.models import HealthResponse, ModelInfoResponse
from api.app.dependencies import fetch_and_write_cert
from unittest.mock import patch, mock_open

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

def test_fetch_and_write_cert():
    """Test fetching and writing SSL certificates using AWS SSM Parameter Store mocks."""
    
    with patch("api.app.dependencies.boto3.client") as mock_boto3_client, \
         patch("api.app.dependencies.os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file:
         
        mock_ssm = mock_boto3_client.return_value
        
        def mock_get_parameter(Name, WithDecryption):
            if Name == "/pii-redaction/SSL_CERT":
                return {"Parameter": {"Value": "mock_cert"}}
            elif Name == "/pii-redaction/SSL_KEY":
                return {"Parameter": {"Value": "mock_key"}}
            return {}
            
        mock_ssm.get_parameter.side_effect = mock_get_parameter
        
        cert_path, key_path = fetch_and_write_cert()
        
        assert cert_path == "/tmp/certs/cert.pem"
        assert key_path == "/tmp/certs/key.pem"
        
        mock_boto3_client.assert_called_once_with("ssm", region_name="us-west-2")
        mock_makedirs.assert_called_once_with("/tmp/certs", exist_ok=True)
        
        assert mock_file.call_count == 2
        mock_file.assert_any_call("/tmp/certs/cert.pem", "w")
        mock_file.assert_any_call("/tmp/certs/key.pem", "w")
        
        mock_file().write.assert_any_call("mock_cert")
        mock_file().write.assert_any_call("mock_key")
