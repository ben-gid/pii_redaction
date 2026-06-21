from api.app.main import limiter
from pii_redaction.models import RedactionResponse

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

def test_demo_redact_response_model(client):
    """Test that /demo/redact response validates against RedactionResponse model."""
    payload = {"text": "My email is john@example.com", "threshold": 0.3}
    response = client.post("/demo/redact", json=payload)
    assert response.status_code == 200
    data = response.json()
    redaction = RedactionResponse.model_validate(data)
    assert redaction.original == payload["text"]
    assert redaction.entity_count == len(redaction.entities)
