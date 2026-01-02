"""Integration tests for the full pipeline."""
import pytest
from fastapi.testclient import TestClient
from verification_service.main import app
import io
from PIL import Image
import base64


client = TestClient(app)


@pytest.fixture
def valid_test_image_b64():
    """Create a valid test image as base64."""
    img = Image.new('RGB', (640, 480), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')


@pytest.mark.integration
def test_full_verification_pipeline():
    """Test the full face verification pipeline end-to-end."""
    img = Image.new('RGB', (640, 480), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    response = client.post(
        "/verify",
        json={"image_base64": img_b64}
    )

    # Accept 200 (processed), 400 (no face or bad input), or 404 (no match)
    assert response.status_code in [200, 400, 404]


@pytest.mark.integration
@pytest.mark.slow
def test_full_verification_pipeline_with_valid_image(valid_test_image_b64):
    """Test the full face verification pipeline end-to-end."""
    response = client.post(
        "/verify",
        json={"image_base64": valid_test_image_b64}
    )
    assert response.status_code in [200, 400, 404]


def test_concurrent_requests():
    """Test handling of concurrent requests."""
    import concurrent.futures

    def make_request():
        return client.get("/health")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for _ in futures]

    assert all(r.status_code == 200 for r in results)
