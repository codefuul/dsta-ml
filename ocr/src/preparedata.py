import os
import json # Import json
from PIL import Image
from bs4 import BeautifulSoup
from typing import List, Dict, Any

# (Keep extract_line_data function mostly the same, but it will return data)
# For simplicity, let's refactor extract_line_data to return a list of dicts for one hocr/image pair
def extract_line_data_json(hocr_path: str, image_path: str, output_dir: str) -> List[Dict[str, str]]:
    """
    Extracts text lines and their corresponding image crops, and returns a list of dictionaries.
    """
    sample_id = os.path.splitext(os.path.basename(hocr_path))[0]
    extracted_data = []

    try:
        with open(hocr_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"❌ Failed to open hOCR {hocr_path}: {e}")
        return extracted_data

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open image {image_path}: {e}")
        return extracted_data

    lines = soup.find_all("span", class_="ocr_line")
    for i, line in enumerate(lines):
        title = line.get("title")
        if not title or "bbox" not in title:
            continue

        try:
            bbox_str = title.split(";")[0].replace("bbox", "").strip()
            x1, y1, x2, y2 = map(int, bbox_str.split())

            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                # print(f"⚠️ Invalid or out-of-bounds bbox in {sample_id}, line {i}: {bbox_str}. Skipping.")
                continue

            text = line.get_text(" ", strip=True)
            if not text:
                continue

            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] < 10 or crop.size[1] < 10:
                # print(f"⚠️ Skipping tiny crop in {sample_id}, line {i}: {crop.size}. (bbox: {bbox_str})")
                continue

            crop_filename = f"{sample_id}_line_{i}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            crop.save(crop_path)

            extracted_data.append({"image_path": crop_path, "text": text})
        except Exception as e:
            print(f"❌ Error processing line {i} in {sample_id}: {e}")
            continue
    return extracted_data

def prepare_data_json(
    raw_data_dir: str = "../../../novice/ocr",
    output_crops_dir: str = "output_crops",
    output_json_path: str = "line_labels.json" # Change to JSON
) -> None:
    """
    Prepares the dataset by extracting line-level images and their transcriptions,
    saving metadata to a JSON file.
    """
    print(f"🗂️ Scanning raw data directory: {raw_data_dir}")
    if not os.path.exists(raw_data_dir):
        print(f"❌ Raw data directory not found: {raw_data_dir}. Please check the path.")
        return

    os.makedirs(output_crops_dir, exist_ok=True)
    
    all_samples_metadata: List[Dict[str, str]] = [] # List to store all sample dictionaries
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".hocr"):
            sample_id = os.path.splitext(filename)[0]
            hocr_path = os.path.join(raw_data_dir, filename)
            image_path = os.path.join(raw_data_dir, f"{sample_id}.jpg")

            if not os.path.exists(image_path):
                print(f"⚠️ No corresponding image found for {sample_id}.hocr. Skipping.")
                skipped_count += 1
                continue
            
            try:
                # Get data for this document, and extend the master list
                doc_lines_data = extract_line_data_json(hocr_path, image_path, output_crops_dir)
                all_samples_metadata.extend(doc_lines_data)
                processed_count += 1
            except Exception as e:
                print(f"❌ Critical error processing {sample_id}.hocr: {e}. Skipping this document.")
                skipped_count += 1
                continue

    # Save all collected metadata to a single JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_samples_metadata, f, indent=4) # Use indent for readability

    print(f"\n✅ Data preparation complete.")
    print(f"Summary: Processed {processed_count} documents, Skipped {skipped_count} documents.")
    print(f"Crops saved to '{output_crops_dir}', labels in '{output_json_path}'.")

if __name__ == "__main__":
    prepare_data_json(
        raw_data_dir=os.path.join("..", "..", "..", "novice", "ocr"),
        output_crops_dir="output_crops",
        output_json_path="line_labels.json"
    )