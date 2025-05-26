import os
import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import WhisperForConditionalGeneration, WhisperProcessor
# If you used a specific feature extractor or tokenizer class during fine-tuning,
# you might need to import them specifically, but WhisperProcessor usually handles it.

# Initialize FastAPI app
app = FastAPI(title="Whisper ASR Manager API")

# --- Configuration ---
# Path to the models directory within the Docker container
# This path is relative to WORKDIR /workspace as defined in the Dockerfile
MODEL_BASE_PATH = "./models"
# This should be the name of your fine-tuned model's directory
MODEL_NAME_OR_PATH = "whisper-base-ft-novice-asr-from-jsonl" # Or "whisper-base-ft-novice-asr-from-jsonl/final_model" if files are in a subdir

# Global variables for the model and processor
asr_model = None
asr_processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define language and task if known from fine-tuning, otherwise, it might be in the model's config
LANGUAGE = "english"
TASK = "transcribe"

def load_asr_components():
    """Loads the ASR model and processor."""
    global asr_model, asr_processor
    
    model_full_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME_OR_PATH)
    
    if not os.path.isdir(model_full_path):
        print(f"Error: Model directory does not exist: {model_full_path}")
        # Consider raising an exception or ensuring the app doesn't start if critical
        return

    try:
        print(f"Loading Whisper processor from {model_full_path}...")
        asr_processor = WhisperProcessor.from_pretrained(model_full_path)
        print("Whisper processor loaded successfully.")

        print(f"Loading Whisper model from {model_full_path} to device: {device}...")
        asr_model = WhisperForConditionalGeneration.from_pretrained(model_full_path)
        asr_model.to(device)
        asr_model.eval() # Set model to evaluation mode
        print("Whisper model loaded successfully and moved to device.")

        # Configuration for fine-tuned Whisper models during inference
        if hasattr(asr_model.config, 'forced_decoder_ids'):
             asr_model.config.forced_decoder_ids = None # Allow model to use its learned defaults or passed ids
        if hasattr(asr_model.config, 'suppress_tokens'):
             asr_model.config.suppress_tokens = []


    except Exception as e:
        print(f"Error loading ASR model/processor from {model_full_path}: {e}")
        import traceback
        traceback.print_exc()
        asr_model = None
        asr_processor = None

@app.on_event("startup")
async def startup_event():
    print("Application startup: Loading ASR components...")
    load_asr_components()
    if asr_model is None or asr_processor is None:
        print("CRITICAL: ASR model or processor failed to load. Transcription endpoint may not work.")

@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    status = "Not Loaded"
    if asr_model and asr_processor:
        status = "Loaded"
    return {"message": "ASR Manager is running.", "model_status": status, "device": device}

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded audio file.
    """
    if not asr_model or not asr_processor:
        raise HTTPException(status_code=503, detail="ASR model/processor is not available or failed to load.")

    try:
        # Read audio file contents
        contents = await audio_file.read()
        
        # Ensure torchaudio is in requirements.txt
        import torchaudio 
        waveform, sample_rate = torchaudio.load(io.BytesIO(contents))

        # If stereo, convert to mono by averaging channels or taking the first one
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary (Whisper expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000 # Update sample_rate after resampling

        # Prepare features
        # The processor takes the raw audio array (numpy array or list of floats)
        # and the sampling rate.
        input_features = asr_processor(
            waveform.squeeze().numpy(), # Squeeze to make it 1D, convert to numpy
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(device)

        # Set forced decoder IDs for the specific task and language during generation
        # This ensures the model performs the desired operation, e.g., transcribe in English.
        # no_timestamps=True based on some Hugging Face discussions for cleaner output
        forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK, no_timestamps=True)

        # Perform inference
        with torch.no_grad():
            predicted_ids = asr_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        
        # Decode the predicted IDs to text
        transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Return the response with the "predictions" key as a list
        return {
            "filename": audio_file.filename,
            "content_type": audio_file.content_type,
            "language_used": LANGUAGE,
            "task_performed": TASK,
            "predictions": [transcription.strip()] # Key is "predictions", value is a list
        }
    except Exception as e:
        print(f"Error during transcription for {audio_file.filename}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e)}")

# For local testing (python src/asr_manager.py)
if __name__ == "__main__":
    import uvicorn
    print(f"Attempting to run ASR Manager locally on device: {device}...")
    # The startup event will handle model loading
    uvicorn.run("asr_manager:app", host="0.0.0.0", port=5001, reload=True)
