# (root)/Nusurvivors/prepare_dataset.py

import json
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Adjust these paths according to your actual root directory
# Assuming this script is in (root)/Nusurvivors/
ROOT_DIR = Path(__file__).resolve().parent.parent 
NOVCE_CV_DIR = ROOT_DIR / "novice" / "cv"
NUSURVIVORS_DIR = ROOT_DIR / "Nusurvivors"

DATASET_OUTPUT_DIR = NUSURVIVORS_DIR / "cv_dataset_yolo"
ANNOTATIONS_FILE = NOVCE_CV_DIR / "annotations.json"
IMAGES_INPUT_DIR = NOVCE_CV_DIR / "images"

TRAIN_RATIO = 0.8 # 80% for training, 20% for validation
# --- End Configuration ---

def convert_coco_to_yolo(coco_annotations_file: Path, images_input_dir: Path, dataset_output_dir: Path, train_ratio: float):
    """
    Converts COCO annotations to YOLO format and splits into train/val sets.
    """
    logging.info(f"Starting dataset conversion from {coco_annotations_file}")
    logging.info(f"Images input directory: {images_input_dir}")
    logging.info(f"YOLO dataset output directory: {dataset_output_dir}")

    if dataset_output_dir.exists():
        logging.warning(f"Output directory {dataset_output_dir} already exists. It will be overwritten.")
        shutil.rmtree(dataset_output_dir)

    # Create directories
    images_train_dir = dataset_output_dir / "images" / "train"
    labels_train_dir = dataset_output_dir / "labels" / "train"
    images_val_dir = dataset_output_dir / "images" / "val"
    labels_val_dir = dataset_output_dir / "labels" / "val"

    for p in [images_train_dir, labels_train_dir, images_val_dir, labels_val_dir]:
        p.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {p}")

    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    all_image_ids = list(images_info.keys())
    if not all_image_ids:
        logging.error("No images found in COCO annotations. Exiting.")
        return

    train_ids, val_ids = train_test_split(all_image_ids, train_size=train_ratio, random_state=42)
    logging.info(f"Total images: {len(all_image_ids)}. Training images: {len(train_ids)}, Validation images: {len(val_ids)}")

    # --- Category information ---
    # Assuming category IDs in annotations.json are already 0-17 as per challenge spec
    # The coco_data['categories'] should have 18 items.
    if 'categories' not in coco_data or len(coco_data['categories']) != 18:
        logging.warning("COCO categories might not match the 18 target classes. Ensure your annotation IDs are 0-17.")
    
    challenge_categories = [
        "cargo aircraft", "commercial aircraft", "drone", "fighter jet", "fighter plane",
        "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus", "van",
        "cargo ship", "yacht", "cruise ship", "warship", "sailboat"
    ]
    
    # Create dataset.yaml file
    dataset_yaml_content = {
        'path': str(dataset_output_dir.resolve()), # Absolute path
        'train': 'images/train',
        'val': 'images/val',
        'nc': 18,
        'names': challenge_categories
    }

    with open(dataset_output_dir / "dataset.yaml", 'w') as f:
        import yaml # Ensure pyyaml is installed: pip install pyyaml
        yaml.dump(dataset_yaml_content, f, sort_keys=False)
    logging.info(f"Created dataset.yaml at {dataset_output_dir / 'dataset.yaml'}")


    for split_name, image_ids, img_out_dir, lbl_out_dir in [
        ('train', train_ids, images_train_dir, labels_train_dir),
        ('val', val_ids, images_val_dir, labels_val_dir)
    ]:
        logging.info(f"Processing {split_name} split...")
        for img_id in tqdm(image_ids, desc=f"Processing {split_name} images"):
            img_data = images_info.get(img_id)
            if not img_data:
                logging.warning(f"Image ID {img_id} not found in images_info. Skipping.")
                continue

            file_name = img_data['file_name']
            img_width = img_data['width']
            img_height = img_data['height']

            # Copy image
            source_image_path = images_input_dir / file_name
            dest_image_path = img_out_dir / file_name
            if source_image_path.exists():
                shutil.copy(source_image_path, dest_image_path)
            else:
                logging.warning(f"Source image {source_image_path} not found. Skipping.")
                continue

            # Create label file
            label_file_path = lbl_out_dir / f"{Path(file_name).stem}.txt"
            yolo_labels = []
            if img_id in annotations_by_image_id:
                for ann in annotations_by_image_id[img_id]:
                    category_id = ann['category_id'] # This should be 0-17
                    bbox = ann['bbox'] # [x_min, y_min, width, height] in COCO

                    # Convert COCO bbox to YOLO bbox
                    # x_center_norm, y_center_norm, width_norm, height_norm
                    x_min, y_min, w, h = bbox
                    x_center = x_min + w / 2
                    y_center = y_min + h / 2

                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = w / img_width
                    height_norm = h / img_height
                    
                    # Clip values to be within [0, 1] to avoid issues with slight out-of-bounds boxes
                    x_center_norm = max(0.0, min(1.0, x_center_norm))
                    y_center_norm = max(0.0, min(1.0, y_center_norm))
                    width_norm = max(0.0, min(1.0, width_norm))
                    height_norm = max(0.0, min(1.0, height_norm))


                    yolo_labels.append(f"{category_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")
            
            if yolo_labels:
                with open(label_file_path, 'w') as f_label:
                    f_label.write("\n".join(yolo_labels))
            # else: an image might have no labels, which is fine. An empty .txt file will be created.
            # However, Ultralytics prefers if such images are not present or if .txt files exist, they are empty.
            # For simplicity, if no labels, we'll just not write a .txt file, or write an empty one.
            # Ultralytics handles images without corresponding .txt files by assuming they have no objects.
            # Writing an empty .txt file is also standard.
            elif not label_file_path.exists(): # Ensure even images with no objects have a (possibly empty) label file
                 with open(label_file_path, 'w') as f_label:
                    pass # Create an empty file

    logging.info("Dataset preparation complete.")
    logging.info(f"YOLO dataset created at: {dataset_output_dir}")
    logging.info(f"YAML file for training: {dataset_output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    # Ensure you have PyYAML: pip install pyyaml
    convert_coco_to_yolo(ANNOTATIONS_FILE, IMAGES_INPUT_DIR, DATASET_OUTPUT_DIR, TRAIN_RATIO)