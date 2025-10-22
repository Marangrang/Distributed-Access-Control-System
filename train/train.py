"""
Placeholder training/embedding generation script.

- Put original images in data/originals/
- Save model checkpoints to ./models/
- Save FAISS index and metadata to verification_service/faiss_index/
"""
import os

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("verification_service/faiss_index", exist_ok=True)
    print("Placeholder train script. Save model checkpoints to ./models/ and index to verification_service/faiss_index/")

if __name__ == "__main__":
    main()