# """Manages the CV model."""


# from typing import Any


# class CVManager:

#     def __init__(self):
#         # This is where you can initialize your model and any static
#         # configurations.
#         pass

#     def cv(self, image: bytes) -> list[dict[str, Any]]:
#         """Performs object detection on an image.

#         Args:
#             image: The image file in bytes.

#         Returns:
#             A list of `dict`s containing your CV model's predictions. See
#             `cv/README.md` for the expected format.
#         """

#         # Your inference code goes here.

#         return []


# (root)/Nusurvivors/cv/src/cv_manager.py

from typing import Any, List, Dict
from ultralytics import YOLO
import torch
from PIL import Image
import io
import os
from pathlib import Path

class CVManager:
    def __init__(self):
        self.model_path = Path(os.getenv("MODEL_PATH", "/workspace/models/best.pt"))
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"CVManager initializing with device: {self.device}")
        print(f"Attempting to load model from: {self.model_path}")

        # --- Parameters for this attempt ---
        self.CONFIDENCE_THRESHOLD = 0.625  # From your high-scoring version
        self.IMG_SIZE_INFERENCE = 768    # TRYING A DIFFERENT INFERENCE IMAGE SIZE (e.g., 512 or 704)
                                         # TTA is OFF by default (augment=False)
        # --- End Parameters ---
        print(f"Inference settings: conf={self.CONFIDENCE_THRESHOLD}, imgsz={self.IMG_SIZE_INFERENCE}, TTA=OFF")

        if not self.model_path.exists():
            alt_model_path = Path(__file__).parent.parent / "models" / "best.pt"
            if alt_model_path.exists():
                self.model_path = alt_model_path
                print(f"Using alternative local model path: {self.model_path}")
            else:
                print(f"ERROR: Model file not found at {self.model_path} or {alt_model_path}")
                self.model = None
                return

        try:
            self.model = YOLO(self.model_path) # This should be your yolov8m model
            self.model.to(self.device)
            print(f"Model {self.model_path} loaded successfully on device {self.device}.")
            
            # Perform a dummy inference to warm up the model
            dummy_img = Image.new('RGB', (self.IMG_SIZE_INFERENCE, self.IMG_SIZE_INFERENCE), color = 'red')
            self.model(dummy_img, verbose=False, imgsz=self.IMG_SIZE_INFERENCE) # TTA is OFF
            print("Model warmup complete.")

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def cv(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        if self.model is None:
            print("ERROR: Model not loaded, returning empty predictions.")
            return []

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return []

        # Perform inference with specified conf and imgsz, TTA is OFF
        results = self.model(image, verbose=False, conf=self.CONFIDENCE_THRESHOLD, imgsz=self.IMG_SIZE_INFERENCE)

        predictions = []
        processed_boxes = None
        if results:
            if isinstance(results, list):
                if results[0].boxes:
                    processed_boxes = results[0].boxes
            elif results.boxes:
                processed_boxes = results.boxes
        
        if processed_boxes:
            for box in processed_boxes:
                xyxy_tensor = box.xyxy[0] 
                xyxy = xyxy_tensor.cpu().numpy()
                x_min, y_min, x_max, y_max = xyxy

                w = x_max - x_min
                h = y_max - y_min
                
                current_bbox = [float(x_min), float(y_min), float(w), float(h)]
                category_id = int(box.cls[0].cpu().item())

                predictions.append({
                    "bbox": current_bbox,
                    "category_id": category_id
                })
        
        return predictions
