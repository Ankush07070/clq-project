import os
import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:10000")
TEST_IMAGE = os.getenv("TEST_IMAGE", "test.jpg")


if __name__ == "__main__":
    with open(TEST_IMAGE, "rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={"file": image_file},
            timeout=60,
        )
    print(response.status_code)
    print(response.json())
