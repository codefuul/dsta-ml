# /home/jupyter/Nusurvivors/asr/download_model.py
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

MODEL_NAME = "facebook/wav2vec2-base-960h"
# Define a local path *relative to your project structure*
# This path will be used inside the Docker container as well.
# We will place it inside the asr module, e.g., asr/models/wav2vec2-base-960h
LOCAL_MODEL_PATH = "./models/wav2vec2-base-960h" # Path relative to asr/ directory

# Create the directory if it doesn't exist
# This script will be run from /home/jupyter/Nusurvivors/asr/
if not os.path.exists(LOCAL_MODEL_PATH):
    os.makedirs(LOCAL_MODEL_PATH)
    print(f"Created directory: {LOCAL_MODEL_PATH}")

try:
    print(f"Downloading processor for {MODEL_NAME} to {LOCAL_MODEL_PATH}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    processor.save_pretrained(LOCAL_MODEL_PATH)
    print("Processor downloaded and saved.")

    print(f"Downloading model {MODEL_NAME} to {LOCAL_MODEL_PATH}...")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.save_pretrained(LOCAL_MODEL_PATH)
    print("Model downloaded and saved.")
    
    print(f"Model files should now be in /home/jupyter/Nusurvivors/asr/{LOCAL_MODEL_PATH}")

except Exception as e:
    print(f"An error occurred: {e}")

