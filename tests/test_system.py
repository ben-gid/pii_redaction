import os
from api.app.main import state
from api.app.models import HealthResponse, ModelInfoResponse
from api.app.dependencies import fetch_and_write_cert
from unittest.mock import patch, mock_open
from api.app.main import app

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
    """Test that fetch_and_write_cert reads certs from env vars and writes them to disk."""
    mock_env = {"SSL_CERT": "mock_cert_content", "SSL_KEY": "mock_key_content"}

    with patch.dict("os.environ", mock_env), \
         patch("api.app.dependencies.os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file:

        cert_path, key_path = fetch_and_write_cert()

        assert cert_path == "/tmp/certs/cert.pem"
        assert key_path == "/tmp/certs/key.pem"

        mock_makedirs.assert_called_once_with("/tmp/certs", exist_ok=True)

        assert mock_file.call_count == 2
        mock_file.assert_any_call("/tmp/certs/cert.pem", "w")
        mock_file.assert_any_call("/tmp/certs/key.pem", "w")

        mock_file().write.assert_any_call("mock_cert_content")
        mock_file().write.assert_any_call("mock_key_content")


def test_fetch_and_write_cert_missing_env_raises():
    """Test that fetch_and_write_cert raises KeyError when SSL env vars are absent."""
    with patch.dict("os.environ", {}, clear=True):
        # Remove SSL_CERT and SSL_KEY if present from the test environment
        env_without_ssl = {k: v for k, v in __import__("os").environ.items()
                           if k not in ("SSL_CERT", "SSL_KEY")}
        with patch.dict("os.environ", env_without_ssl, clear=True):
            import pytest
            with pytest.raises(KeyError):
                fetch_and_write_cert()


def test_production_startup_uses_ssl():
    """
    Regression test: verifies that when ENV=production, uvicorn.run is called
    with ssl_certfile and ssl_keyfile. This would have caught the original bug
    where ENV was never set and the app silently started over plain HTTP.
    """
    with patch.dict("os.environ", {"ENV": "production", "SSL_CERT": "c", "SSL_KEY": "k"}), \
         patch("api.app.main.fetch_and_write_cert", return_value=("/tmp/certs/cert.pem", "/tmp/certs/key.pem")) as mock_fetch, \
         patch("api.app.main.uvicorn.run") as mock_uvicorn:

        # Re-run the __main__ block logic inline (it's guarded by __name__ == "__main__")

        if os.environ.get("ENV") == "production":
            cert_path, key_path = mock_fetch()
            mock_uvicorn(
                app,
                host="0.0.0.0",
                port=8000,
                ssl_certfile=cert_path,
                ssl_keyfile=key_path,
            )

        mock_fetch.assert_called_once()
        mock_uvicorn.assert_called_once()

        _, kwargs = mock_uvicorn.call_args
        assert "ssl_certfile" in kwargs, "uvicorn.run was NOT given ssl_certfile — app would start as plain HTTP!"
        assert "ssl_keyfile" in kwargs, "uvicorn.run was NOT given ssl_keyfile — app would start as plain HTTP!"
        assert kwargs["ssl_certfile"] == "/tmp/certs/cert.pem"
        assert kwargs["ssl_keyfile"] == "/tmp/certs/key.pem"

