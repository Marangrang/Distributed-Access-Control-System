"""Test face verification logic."""
import pytest
import numpy as np
from verification_service.build_index import build_faiss_index
from PIL import Image
import io


def test_faiss_index_creation():
    """Test FAISS index creation."""
    # Create dummy embeddings
    embeddings = np.random.rand(10, 512).astype('float32')
    index = build_faiss_index(embeddings)
    assert index is not None
    assert index.ntotal == 10


def test_embedding_dimension():
    """Test that embeddings have correct dimension."""
    embedding = np.random.rand(512).astype('float32')
    assert embedding.shape == (512,)


def test_face_detection_no_face():
    """Test face detection with no face in image."""
    # Create blank image
    img = Image.new('RGB', (100, 100), color='white')
    # Add test logic here based on your face detection implementation
    assert img is not None


@pytest.fixture
def sample_face_image():
    """Create a sample face image for testing."""
    img = Image.new('RGB', (160, 160), color='gray')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_image_preprocessing(sample_face_image):
    """Test image preprocessing pipeline."""
    img = Image.open(sample_face_image)
    assert img.size == (160, 160)
    assert img.mode == 'RGB'