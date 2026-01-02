"""
Test face verification API
Usage: python test_verify.py <image_path> [--url http://localhost:8080]
"""
import requests
import base64
import time
import sys
from pathlib import Path
import argparse


def encode_image(path: str) -> str:
    """Encode image to base64"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_verify(image_path: str, url: str = "http://localhost:8000/verify", driver_id: str = None):
    """Test the verify endpoint"""

    print(f"Testing verification API")
    print(f"Image: {image_path}")
    print(f"URL: {url}")
    print("-" * 60)

    try:
        # Encode image
        print("Encoding image...")
        image_base64 = encode_image(image_path)
        print(f"✓ Image encoded ({len(image_base64)} bytes)")

        # Prepare payload
        payload = {
            "image_base64": image_base64
        }
        if driver_id:
            payload["driver_id"] = driver_id

        # Send request
        print(f"\nSending POST request...")
        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        elapsed_ms = (time.time() - start) * 1000

        # Print results
        print(f"✓ Response received")
        print(f"\nStatus Code: {response.status_code}")
        print(f"Latency: {elapsed_ms:.2f} ms")
        print(f"\nResponse:")
        print("-" * 60)

        if response.status_code == 200:
            result = response.json()
            print(f"Verified: {result.get('verified')}")
            print(f"Driver ID: {result.get('driver_id')}")
            print(f"Name: {result.get('name')}")
            print(f"Similarity: {result.get('similarity', 0):.4f}")
            print(f"Distance: {result.get('distance', 0):.4f}")

            if result.get('verified'):
                print("\n✓ VERIFICATION PASSED")
            else:
                print("\n✗ VERIFICATION FAILED")
        else:
            print(f"Error: {response.text}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out after 30 seconds")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to {url}")
        print("  Is the verification service running?")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


def test_health(base_url: str):
    """Test the health endpoint"""
    url = f"{base_url}/health"
    print(f"Checking service health: {url}")

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Service is healthy")
            print(f"  Vectors loaded: {health.get('num_vectors')}")
            print(f"  Metadata entries: {health.get('num_metadata')}")
            return True
        else:
            print(f"✗ Service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face verification API")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--port", type=int, default=8000, help="API port (used to build URL if --url not specified)")
    parser.add_argument("--driver-id", help="Expected driver ID")
    parser.add_argument("--health-only", action="store_true", help="Only check health endpoint")

    args = parser.parse_args()

    # Health check
    print("=" * 60)
    print("Face Verification API Test")
    print("=" * 60)

    base_url = args.url.rstrip('/')
    if not test_health(base_url):
        print("\n✗ Service is not healthy. Exiting.")
        sys.exit(1)

    if args.health_only:
        sys.exit(0)

    # Verify test
    if not args.image:
        print("\n✗ Error: Image path required")
        print("Usage: python test_verify.py <image_path>")
        sys.exit(1)

    print("\n" + "=" * 60)
    test_verify(args.image, f"{base_url}/verify", args.driver_id)
