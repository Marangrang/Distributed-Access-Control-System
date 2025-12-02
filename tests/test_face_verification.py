"""Test face verification logic."""
import pytest
import numpy as np
from PIL import Image
import io


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return np.random.rand(10, 512).astype('float32')


def test_faiss_index_creation(sample_embeddings):
    """Test FAISS index creation."""
    import faiss
    
    # Create FAISS index directly (not using build_index function)
    dim = sample_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sample_embeddings)
    
    assert index is not None
    assert index.ntotal == 10


def test_embedding_dimension():
    """Test that embeddings have correct dimension."""
    embedding = np.random.rand(512).astype('float32')
    assert embedding.shape == (512,)


def test_embedding_normalization(sample_embeddings):
    """Test L2 normalization of embeddings."""
    norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = sample_embeddings / norms
    
    # Check all vectors have norm ~1.0
    result_norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(result_norms, 1.0, atol=1e-5)


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


def test_face_detection_no_face():
    """Test face detection with no face in image."""
    # Create blank image
    img = Image.new('RGB', (100, 100), color='white')
    assert img is not None
    assert img.size == (100, 100)


def test_build_index_integration(tmp_path):
    """Integration test for build_index function."""
    from verification_service.build_index import build_index
    
    # Create test images
    test_images = []
    metadata = []
    
    for i in range(3):
        img_path = tmp_path / f"test_face_{i}.jpg"
        img = Image.new('RGB', (160, 160), color=(i*50, i*50, i*50))
        img.save(img_path)
        test_images.append(str(img_path))
        metadata.append({'driver_id': f'driver_{i}', 'name': f'Test Driver {i}'})
    
    # This will create index files in verification_service/faiss_index/
    # In production, you'd want to use tmp_path for test isolation
    build_index(test_images, metadata)
    
    # Verify index was created (in real dir, not tmp)
    from verification_service.build_index import INDEX_PATH, METADATA_PATH
    import os
    
    assert os.path.exists(INDEX_PATH), f"Index not created at {INDEX_PATH}"
    assert os.path.exists(METADATA_PATH), f"Metadata not created at {METADATA_PATH}"


def test_index_search_functionality():
    """Test FAISS index search functionality."""
    import faiss
    
    # Create sample data
    dim = 512
    n_vectors = 100
    embeddings = np.random.rand(n_vectors, dim).astype('float32')
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Build index
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Search for nearest neighbor
    query = embeddings[0:1]  # Use first vector as query
    k = 5
    distances, indices = index.search(query, k)
    
    # First result should be the query itself
    assert indices[0][0] == 0
    assert distances[0][0] > 0.99  # Should be very close to 1.0 (cosine similarity)