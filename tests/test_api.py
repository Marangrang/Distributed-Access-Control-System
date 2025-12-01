"""Test API endpoints."""
import pytest
from fastapi.testclient import TestClient
from serve.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_verify_endpoint_missing_image():
    """Test verify endpoint with missing image."""
    response = client.post("/verify")
    assert response.status_code == 422  # Validation error


def test_verify_endpoint_invalid_image():
    """Test verify endpoint with invalid image data."""
    response = client.post(
        "/verify",
        files={"image": ("test.jpg", b"invalid_data", "image/jpeg")}
    )
    assert response.status_code in [400, 422]


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_info" in response.text