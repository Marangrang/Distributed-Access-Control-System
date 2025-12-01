"""Integration tests for the full pipeline."""
import pytest
from fastapi.testclient import TestClient
from serve.main import app
import io
from PIL import Image


client = TestClient(app)


@pytest.fixture
def valid_test_image():
    """Create a valid test image."""
    img = Image.new('RGB', (640, 480), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_full_verification_pipeline(valid_test_image):
    """Test complete verification pipeline."""
    response = client.post(
        "/verify",
        files={"image": ("test.jpg", valid_test_image, "image/jpeg")}
    )
    # Adjust assertion based on expected behavior
    assert response.status_code in [200, 404]  # 404 if no match found


def test_concurrent_requests():
    """Test handling of concurrent requests."""
    import concurrent.futures
    
    def make_request():
        return client.get("/health")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in results)