"""
Build FAISS index from existing embeddings or images
Compatible with train.py and main.py (uses L2 distance)
"""
import os
import json
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import sys

INDEX_DIR = Path("verification_service/faiss_index")
INDEX_PATH = INDEX_DIR / "driver_vectors.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBEDDINGS_PATH = INDEX_DIR / "driver_vectors.npy"
IMAGE_SIZE = 160

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    keep_all=False,
    device=device,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def load_image(path):
    """Load image from path"""
    return Image.open(path).convert("RGB")


def extract_face_embedding(img_path):
    """
    Extract face embedding from image
    Returns None if no face detected (consistent with main.py)
    """
    try:
        img = load_image(img_path)
        face = mtcnn(img)

        if face is None:
            print(f"⚠ No face detected in {img_path}")
            return None

        with torch.no_grad():
            face = face.unsqueeze(0).to(device)
            emb = resnet(face).cpu().numpy()[0]

        return emb
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def encode_images(image_paths, batch_size=32):
    """
    Extract embeddings from multiple images
    Skips images where no face is detected
    """
    embeddings = []
    valid_indices = []

    for i, p in enumerate(tqdm(image_paths, desc="Embedding images")):
        emb = extract_face_embedding(p)
        if emb is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    if len(embeddings) == 0:
        return np.zeros((0, 512), dtype='float32'), []

    embeddings_array = np.vstack(embeddings).astype('float32')
    return embeddings_array, valid_indices


def build_index_from_images(image_paths, metadata_list):
    """
    Build FAISS index from image files

    Args:
        image_paths: list of image file paths
        metadata_list: list of dicts with driver_id, name, etc.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if len(image_paths) == 0:
        print("No images provided.")
        return

    print(f"Processing {len(image_paths)} images...")
    embeddings, valid_indices = encode_images(image_paths)

    if len(embeddings) == 0:
        print("No valid embeddings extracted!")
        return

    valid_metadata = [metadata_list[i] for i in valid_indices]

    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-12)

    # Save normalized embeddings
    np.save(EMBEDDINGS_PATH, embeddings_normalized)
    print(f"✓ Saved embeddings to {EMBEDDINGS_PATH}")

    # Build FAISS index using L2 distance
    dim = embeddings_normalized.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_normalized)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✓ Saved FAISS index with {index.ntotal} vectors to {INDEX_PATH}")

    # Save metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(valid_metadata, f, indent=2)
    print(f"✓ Saved metadata with {len(valid_metadata)} entries to {METADATA_PATH}")

    print(f"\n✓ Index built successfully!")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dim}")
    print(f"  Index type: L2 (Euclidean distance)")
    print(f"  Failed detections: {len(image_paths) - len(embeddings)}")


def build_index_from_embeddings(verify_only=False):
    """
    Rebuild FAISS index from existing embeddings file

    Args:
        verify_only: If True, only verify the index without rebuilding
    """
    if not EMBEDDINGS_PATH.exists():
        print(f"❌ Embeddings not found at {EMBEDDINGS_PATH}")
        print("Run train.py first to generate embeddings")
        return False

    # Load embeddings
    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)

    if len(embeddings) == 0:
        print("❌ No embeddings found")
        return False

    print(f"✓ Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Verify existing index if requested
    if verify_only and INDEX_PATH.exists():
        print(f"\nVerifying existing index at {INDEX_PATH}")
        try:
            existing_index = faiss.read_index(str(INDEX_PATH))
            if existing_index.ntotal == len(embeddings):
                print(f"✓ Index is valid with {existing_index.ntotal} vectors")
                return True
            else:
                print(f"⚠ Index mismatch: {existing_index.ntotal} vectors vs {len(embeddings)} embeddings")
                print("  Consider rebuilding with --rebuild flag")
                return False
        except Exception as e:
            print(f"✗ Index verification failed: {e}")
            return False

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-12)

    # Build FAISS index using L2 distance
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_normalized.astype('float32'))

    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✓ Saved FAISS index with {index.ntotal} vectors to {INDEX_PATH}")

    # Verify metadata exists
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"✓ Metadata exists with {len(metadata)} entries")

        # Verify metadata count matches embeddings
        if len(metadata) != len(embeddings):
            print(f"⚠ Warning: Metadata count ({len(metadata)}) doesn't match embeddings ({len(embeddings)})")
    else:
        print("⚠ Warning: metadata.json not found")

    return True


def verify_index_integrity():
    """Verify integrity of all index files"""
    print("=" * 60)
    print("Index Integrity Check")
    print("=" * 60)

    all_ok = True

    # Check embeddings
    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"✓ Embeddings: {len(embeddings)} vectors, dimension {embeddings.shape[1]}")
    else:
        print(f"✗ Embeddings not found: {EMBEDDINGS_PATH}")
        all_ok = False

    # Check index
    if INDEX_PATH.exists():
        try:
            index = faiss.read_index(str(INDEX_PATH))
            print(f"✓ FAISS Index: {index.ntotal} vectors")
        except Exception as e:
            print(f"✗ FAISS Index corrupted: {e}")
            all_ok = False
    else:
        print(f"✗ FAISS Index not found: {INDEX_PATH}")
        all_ok = False

    # Check metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        print(f"✓ Metadata: {len(metadata)} entries")
    else:
        print(f"✗ Metadata not found: {METADATA_PATH}")
        all_ok = False

    # Cross-check counts
    if all_ok and len(embeddings) == index.ntotal == len(metadata):
        print(f"\n✓ All files consistent with {len(embeddings)} entries")
    elif all_ok:
        print(f"\n⚠ Count mismatch detected:")
        print(f"  Embeddings: {len(embeddings)}")
        print(f"  Index: {index.ntotal}")
        print(f"  Metadata: {len(metadata)}")
        all_ok = False

    print("=" * 60)
    return all_ok


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build or verify FAISS index for face verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify index integrity
  python build_index.py --verify

  # Rebuild index from embeddings
  python build_index.py --rebuild

  # Rebuild and verify
  python build_index.py --rebuild --verify
        """
    )

    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Rebuild FAISS index from existing embeddings'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify index integrity without rebuilding'
    )

    parser.add_argument(
        '--from-embeddings',
        action='store_true',
        help='(Deprecated) Use --rebuild instead'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Index Builder & Verifier")
    print("=" * 60)
    print("Compatible with train.py and main.py (L2 distance)")
    print("=" * 60 + "\n")

    # Handle deprecated flag
    if args.from_embeddings:
        print("Note: --from-embeddings is deprecated, use --rebuild")
        args.rebuild = True

    # Default action if no flags provided
    if not args.rebuild and not args.verify:
        print("No action specified. Running verification...")
        args.verify = True

    success = True

    # Verify integrity
    if args.verify:
        success = verify_index_integrity()
        print()

    # Rebuild if requested
    if args.rebuild:
        print("Rebuilding index from embeddings...")
        success = build_index_from_embeddings(verify_only=False)

    if success:
        print("\n✓ Operation completed successfully!")
        return 0
    else:
        print("\n✗ Operation failed")
        print("\nTroubleshooting:")
        print("  1. Run train.py to generate fresh embeddings")
        print("  2. Check file permissions on faiss_index/ directory")
        print("  3. Verify FAISS library installation")
        return 1


if __name__ == "__main__":
    exit(main())
