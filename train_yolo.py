# # (root)/Nusurvivors/train_yolo.py
# from ultralytics import YOLO
# import torch
# import os
# from pathlib import Path

# def main():
#     # --- Configuration ---
#     # Assuming this script is in (root)/Nusurvivors/
#     NUSURVIVORS_DIR = Path(__file__).resolve().parent
#     DATASET_YAML = NUSURVIVORS_DIR / "cv_dataset_yolo" / "dataset.yaml"
    
#     # Model choice: yolov8s.pt (small), yolov8m.pt (medium)
#     # For T4 and balance, yolov8s.pt or yolov8m.pt are good.
#     MODEL_NAME = 'yolov8s.pt' 
#     EPOCHS = 50 # Start with 50, can increase if needed and time permits
#     BATCH_SIZE = 16 # Adjust based on GPU memory (T4 has 16GB). 16 should be fine. For larger models, reduce.
#     IMG_SIZE = 640 # Standard image size
#     PROJECT_NAME = 'Nusurvivors_CV_Training'
#     RUN_NAME = 'yolov8s_finetune_18class'
#     DEVICE = 0 if torch.cuda.is_available() else 'cpu' # Use GPU if available
#     # --- End Configuration ---

#     print(f"Using device: {DEVICE}")
#     if not torch.cuda.is_available():
#         print("WARNING: CUDA not available, training on CPU will be very slow.")

#     # Load a pre-trained YOLOv8 model
#     model = YOLO(MODEL_NAME)

#     # Train the model
#     results = model.train(
#         data=str(DATASET_YAML),
#         epochs=EPOCHS,
#         batch=BATCH_SIZE,
#         imgsz=IMG_SIZE,
#         project=PROJECT_NAME,
#         name=RUN_NAME,
#         device=DEVICE,
#         patience=10, # Early stopping patience
#         # workers=4, # Number of dataloader workers, adjust based on your CPU cores
#         # exist_ok=True, # if you want to overwrite previous runs with the same name
#     )

#     print("Training complete.")
#     print(f"Results saved to: {results.save_dir}")
    
#     # The best model is saved as best.pt in the run directory, e.g.,
#     # Nusurvivors_CV_Training/yolov8s_finetune_18class/weights/best.pt
#     # We need to copy this to a known location for the Docker image.
    
#     best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
#     # Define where to save the model for the Docker build
#     # (root)/Nusurvivors/cv/models/best.pt
#     destination_model_dir = NUSURVIVORS_DIR / "cv" / "models"
#     destination_model_dir.mkdir(parents=True, exist_ok=True)
#     destination_model_path = destination_model_dir / "best.pt"
    
#     if best_model_path.exists():
#         import shutil
#         shutil.copy(best_model_path, destination_model_path)
#         print(f"Best model copied to: {destination_model_path}")
#     else:
#         print(f"ERROR: Could not find best model at {best_model_path}")


# if __name__ == '__main__':
#     main()

# (root)/Nusurvivors/train_yolo.py
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import shutil # For copying the model

def main():
    # --- Configuration ---
    # Assuming this script is in (root)/Nusurvivors/
    NUSURVIVORS_DIR = Path(__file__).resolve().parent
    DATASET_YAML = NUSURVIVORS_DIR / "cv_dataset_yolo" / "dataset.yaml"
    
    # Model choice: Changed to yolov8m.pt for potentially better accuracy
    MODEL_NAME = 'yolov8m.pt' 
    EPOCHS = 100 # Increased epochs for better convergence
    BATCH_SIZE = 8 # Reduced batch size for yolov8m on a 16GB T4 GPU
    IMG_SIZE = 640 # Standard image size
    PROJECT_NAME = 'Nusurvivors_CV_Training' # Keep project name consistent or change as desired
    RUN_NAME = 'yolov8m_e100_b8_improved' # Updated run name
    DEVICE = 0 if torch.cuda.is_available() else 'cpu' # Use GPU if available
    
    # Hyperparameters (Ultralytics defaults are generally good starting points)
    LEARNING_RATE0 = 0.01  # Initial learning rate
    LEARNING_RATEF = 0.01  # Final OneCycleLR learning rate (lr0 * lrf)
    WEIGHT_DECAY = 0.0005 # Optimizer weight decay
    PATIENCE_EARLY_STOPPING = 20 # Increased patience for early stopping
    WORKERS = 4 # Number of dataloader workers, adjust based on your CPU cores
    # --- End Configuration ---

    print(f"Using device: {DEVICE}")
    if not torch.cuda.is_available() and DEVICE != 'cpu':
        print("WARNING: CUDA not available, forcing CPU. Training will be very slow.")
        DEVICE = 'cpu'
    elif DEVICE == 'cpu':
         print("INFO: Training on CPU. This will be very slow.")


    print(f"Dataset YAML path: {DATASET_YAML}")
    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML file not found at {DATASET_YAML}")
        print("Please ensure you have run 'prepare_dataset.py' and the path is correct.")
        return

    # Load a pre-trained YOLOv8 model
    model = YOLO(MODEL_NAME)

    # Train the model
    print(f"Starting training with model: {MODEL_NAME}, epochs: {EPOCHS}, batch_size: {BATCH_SIZE}")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        device=DEVICE,
        patience=PATIENCE_EARLY_STOPPING, 
        lr0=LEARNING_RATE0,
        lrf=LEARNING_RATEF,
        weight_decay=WEIGHT_DECAY,
        workers=WORKERS, 
        exist_ok=True, # Allows overwriting if a run with the same name exists
        # You can add more augmentation parameters here if needed, e.g.:
        # degrees=10.0, translate=0.1, scale=0.1, shear=0.1, perspective=0.001,
        # flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    )

    print("Training complete.")
    print(f"Results saved to: {results.save_dir}")
    
    # The best model is saved as best.pt in the run directory, e.g.,
    # Nusurvivors_CV_Training/yolov8m_e100_b8_improved/weights/best.pt
    
    best_model_path_source = Path(results.save_dir) / "weights" / "best.pt"
    
    # Define where to save the model for the Docker build
    # (root)/Nusurvivors/cv/models/best.pt
    destination_model_dir = NUSURVIVORS_DIR / "cv" / "models"
    destination_model_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    destination_model_path_target = destination_model_dir / "best.pt"
    
    if best_model_path_source.exists():
        shutil.copy(best_model_path_source, destination_model_path_target)
        print(f"Best model copied from {best_model_path_source} to: {destination_model_path_target}")
    else:
        print(f"ERROR: Could not find best model at {best_model_path_source}")
        print("Please check the training logs and output directory.")

if __name__ == '__main__':
    main()

# # (root)/Nusurvivors/train_yolo.py
# from ultralytics import YOLO
# import torch
# import os
# from pathlib import Path
# import shutil # For copying the model

# def main():
#     # --- Configuration ---
#     NUSURVIVORS_DIR = Path(__file__).resolve().parent
#     DATASET_YAML = NUSURVIVORS_DIR / "cv_dataset_yolo" / "dataset.yaml"
    
#     MODEL_NAME = 'yolov8s.pt'  # Switch to small model for faster training
#     EPOCHS = 50                # Reduce epochs for faster turnaround
#     BATCH_SIZE = 16            # Adjust batch size for smaller model
#     IMG_SIZE = 640             # Standard image size
#     PROJECT_NAME = 'Nusurvivors_CV_Training' 
#     RUN_NAME = 'yolov8s_e50_b16_fast_tune' # Updated run name
#     DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    
#     # Hyperparameters (Ultralytics defaults are generally good starting points)
#     LEARNING_RATE0 = 0.01  # Initial learning rate
#     LEARNING_RATEF = 0.01  # Final OneCycleLR learning rate (lr0 * lrf)
#     WEIGHT_DECAY = 0.0005  # Optimizer weight decay
#     PATIENCE_EARLY_STOPPING = 20 # Patience for early stopping
#     WORKERS = 4                # Number of dataloader workers
#     # --- End Configuration ---

#     print(f"Using device: {DEVICE}")
#     if not torch.cuda.is_available() and DEVICE != 'cpu':
#         print("WARNING: CUDA not available, forcing CPU. Training will be very slow.")
#         DEVICE = 'cpu'
#     elif DEVICE == 'cpu':
#          print("INFO: Training on CPU. This will be very slow.")

#     print(f"Dataset YAML path: {DATASET_YAML}")
#     if not DATASET_YAML.exists():
#         print(f"ERROR: Dataset YAML file not found at {DATASET_YAML}")
#         print("Please ensure you have run 'prepare_dataset.py' and the path is correct.")
#         return

#     model = YOLO(MODEL_NAME)

#     print(f"Starting training with model: {MODEL_NAME}, epochs: {EPOCHS}, batch_size: {BATCH_SIZE}")
#     results = model.train(
#         data=str(DATASET_YAML),
#         epochs=EPOCHS,
#         batch=BATCH_SIZE,
#         imgsz=IMG_SIZE,
#         project=PROJECT_NAME,
#         name=RUN_NAME,
#         device=DEVICE,
#         patience=PATIENCE_EARLY_STOPPING, 
#         lr0=LEARNING_RATE0,
#         lrf=LEARNING_RATEF,
#         weight_decay=WEIGHT_DECAY,
#         workers=WORKERS, 
#         exist_ok=True # Allows overwriting if a run with the same name exists
#     )

#     print("Training complete.")
#     print(f"Results saved to: {results.save_dir}")
    
#     best_model_path_source = Path(results.save_dir) / "weights" / "best.pt"
#     destination_model_dir = NUSURVIVORS_DIR / "cv" / "models"
#     destination_model_dir.mkdir(parents=True, exist_ok=True)
#     destination_model_path_target = destination_model_dir / "best.pt"
    
#     if best_model_path_source.exists():
#         shutil.copy(best_model_path_source, destination_model_path_target)
#         print(f"Best model copied from {best_model_path_source} to: {destination_model_path_target}")
#     else:
#         print(f"ERROR: Could not find best model at {best_model_path_source}")
#         print("Please check the training logs and output directory.")

# if __name__ == '__main__':
#     main()
