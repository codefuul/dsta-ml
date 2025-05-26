import requests
import base64
import json
import os

# --- Configuration ---
OCR_SERVER_URL = "http://localhost:5003/ocr"
# Replace with the actual path to your test image
TEST_IMAGE_PATH = "../novice/ocr/sample_0.jpg" # e.g., "data/top_secret_report.jpg"

def test_ocr_with_single_image(image_path: str):
    """
    Sends a single image to the OCR server and prints the result.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Loading image from: {image_path}")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    # Encode the image bytes to base64
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    # Prepare the payload for the API request
    payload = {
        "instances": [
            {"b64": b64_image}
        ]
    }

    print(f"Sending request to {OCR_SERVER_URL}...")
    try:
        response = requests.post(OCR_SERVER_URL, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        
        if "predictions" in result and result["predictions"]:
            ocr_text = result["predictions"][0]
            print("\n--- OCR Result ---")
            print(ocr_text)
            print("\n------------------")
        else:
            print("No predictions found in the response.")

    except requests.exceptions.ConnectionError as e:
        print(f"Error: Could not connect to the OCR server at {OCR_SERVER_URL}.")
        print("Please ensure your Docker container is running and accessible.")
        print(f"Details: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage:
    # Make sure to replace this with the actual path to YOUR test image.
    # For instance, if you have a test image in a 'test_images' folder:
    # TEST_IMAGE_PATH = "test_images/top_secret_document.png"
    
    # Or, if you use a relative path like in your original prepare_data setup:
    # Assuming 'top_secret_report.jpg' is in your 'novice/ocr' directory.
    # Adjust this path based on where you run this script relative to your image.
    
    # You might want to copy one of your original document images (like 'doc_000000.jpg')
    # from your raw data directory to a dedicated test folder for this script.
    
    # Example: If your "TOP SECRET" report is in a 'my_test_data' folder
    # in the same directory as this script:
    # TEST_IMAGE_PATH = "my_test_data/top_secret_report.jpg"

    # Make sure to set this path correctly!
    TEST_IMAGE_PATH = "../novice/ocr/sample_0.jpg" # <--- IMPORTANT: Change this!

    print("Starting single image OCR test...")
    test_ocr_with_single_image(TEST_IMAGE_PATH)
    print("Test finished.")