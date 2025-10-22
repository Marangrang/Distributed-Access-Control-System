import requests
import base64
import time

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Path to your test image (make sure this file exists!)
image_path = "data/test/face.2025.1.jpg"

url = "http://localhost:8080/verify"

payload = {
    "driver_id": "marang",
    "image_base64": encode_image(image_path)
}

print("Sending request to", url)
start = time.time()
try:
    response = requests.post(url, json=payload, timeout=10)  # 10 second timeout
    elapsed = (time.time() - start) * 1000
    print("Status code:", response.status_code)
    print("Raw text:", response.text)
    print(f"Latency: {elapsed:.2f} ms")
    try:
        print("Parsed JSON:", response.json())
    except Exception as e:
        print("Failed to parse JSON:", str(e))
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
