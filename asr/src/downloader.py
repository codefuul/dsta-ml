import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Define a local directory to save the model
LOCAL_WHISPER_MODEL_PATH = "../models/local_whisper_base" # This will create a folder like this

# Create the directory if it doesn't exist
os.makedirs(LOCAL_WHISPER_MODEL_PATH, exist_ok=True)

print(f"Downloading WhisperProcessor to {LOCAL_WHISPER_MODEL_PATH}...")
# Note: language and task are not needed for saving, only for actual processing.
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
processor.save_pretrained(LOCAL_WHISPER_MODEL_PATH)
print("Processor downloaded and saved.")

print(f"Downloading WhisperForConditionalGeneration model to {LOCAL_WHISPER_MODEL_PATH}...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.save_pretrained(LOCAL_WHISPER_MODEL_PATH)
print("Model downloaded and saved.")

print("Whisper model and processor successfully downloaded locally.")
print(f"Check the contents of the '{LOCAL_WHISPER_MODEL_PATH}' directory.")