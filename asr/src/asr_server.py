# """Runs the ASR server."""

# # Unless you want to do something special with the server, you shouldn't need
# # to change anything in this file.


# import base64
# from fastapi import FastAPI, Request
# from asr_manager import ASRManager


# app = FastAPI()
# manager = ASRManager()


# @app.post("/asr")
# async def asr(request: Request) -> dict[str, list[str]]:
#     """Performs ASR on audio files.

#     Args:
#         request: The API request. Contains a list of audio files, encoded in
#             base-64.

#     Returns:
#         A `dict` with a single key, `"predictions"`, mapping to a `list` of
#         `str` transcriptions, in the same order as which appears in `request`.
#     """

#     inputs_json = await request.json()

#     predictions = []
#     for instance in inputs_json["instances"]:

#         # Reads the base-64 encoded audio and decodes it into bytes.
#         audio_bytes = base64.b64decode(instance["b64"])

#         # Performs ASR and appends the result.
#         transcription = manager.asr(audio_bytes)
#         predictions.append(transcription)

#     return {"predictions": predictions}


# @app.get("/health")
# def health() -> dict[str, str]:
#     """Health check endpoint for the server."""
#     return {"message": "health ok"}

# /home/jupyter/Nusurvivors/asr/src/asr_server.py

# import base64
# from fastapi import FastAPI, Request
# # from .asr_manager import ASRManager # Keep this if it was working, or:
# import asr_manager # Use direct import if the relative one was problematic
# import json # For pretty printing JSON
# import traceback # For detailed error logging

# app = FastAPI()
# print("[ASRServer INFO] Initializing ASRManager instance...")
# manager = asr_manager.ASRManager() # Ensure this matches your import style
# print("[ASRServer INFO] ASRManager instance created.")


# @app.post("/asr")
# async def asr_endpoint(request: Request) -> dict[str, list[str]]: # Renamed function to avoid conflict
#     print("[ASRServer INFO] /asr endpoint hit.")
#     try:
#         inputs_json = await request.json()
#         # print(f"[ASRServer DEBUG] Received request JSON: {json.dumps(inputs_json, indent=2)}") # Can be very verbose for audio data

#         predictions = []
#         if "instances" not in inputs_json or not isinstance(inputs_json["instances"], list):
#             print("[ASRServer ERROR] Invalid request format: 'instances' key missing or not a list.")
#             # You might want to return a FastAPI HTTPException here for a proper 400
#             return {"error": "Invalid request format"} # Or conform to expected error if any

#         for i, instance in enumerate(inputs_json["instances"]):
#             print(f"[ASRServer INFO] Processing instance {i+1}/{len(inputs_json['instances'])}.")
#             if "b64" not in instance:
#                 print(f"[ASRServer ERROR] Instance {i+1} missing 'b64' key.")
#                 predictions.append("ERROR: MISSING B64 KEY IN INSTANCE")
#                 continue
            
#             audio_bytes = base64.b64decode(instance["b64"])
#             print(f"[ASRServer INFO] Instance {i+1}: Decoded base64 audio, bytes length: {len(audio_bytes)}")
            
#             transcription = manager.asr(audio_bytes) # Calls your ASRManager's method
#             print(f"[ASRServer INFO] Instance {i+1}: Received transcription from manager: '{transcription}'")
#             predictions.append(transcription)

#         response_payload = {"predictions": predictions}
#         print(f"[ASRServer INFO] Sending response: {json.dumps(response_payload)}") # Log before sending
#         return response_payload

#     # /home/jupyter/Nusurvivors/asr/src/asr_server.py
# # ...
#     except Exception as e:
#         print(f"[ASRServer ERROR] Unhandled exception in /asr endpoint: {e}")
#         print(traceback.format_exc())
#         # Ensure this list has the same number of elements as expected instances
#         num_instances = 0
#         try:
#             # Attempt to get the number of instances from the request if possible
#             # This part might fail if inputs_json itself is the problem
#             if 'inputs_json' in locals() and isinstance(inputs_json.get("instances"), list):
#                 num_instances = len(inputs_json["instances"])
#             elif isinstance(await request.json().get("instances"), list): # Try to re-parse if needed (careful)
#                 num_instances = len(await request.json()["instances"])
#             else: # Fallback if we can't determine instance count
#                 num_instances = 1 # Or some default, though this might mismatch
#         except:
#             num_instances = 1 # Default if everything fails
#             print("[ASRServer WARNING] Could not determine number of instances during exception handling for error response.")

#         error_predictions = ["SERVER_ENDPOINT_EXCEPTION" for _ in range(num_instances if num_instances > 0 else 1)]
#         return {"predictions": error_predictions}



# @app.get("/health")
# def health() -> dict[str, str]:
#     print("[ASRServer INFO] /health endpoint hit.")
#     if manager and manager.model and manager.processor:
#         return {"message": "health ok", "model_status": "loaded"}
#     else:
#         print("[ASRServer WARNING] Health check: Model or processor not loaded.")
#         return {"message": "health degraded", "model_status": "error or not loaded"}

# print("[ASRServer INFO] ASR Server script loaded. FastAPI app configured.")


import base64
import time
from typing import List, Dict
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import json
import traceback # Import traceback for detailed error logging

# Import your ASRManager.
# Assuming asr_manager.py is in the same 'src' directory as asr_server.py,
# a direct import is usually sufficient when running via `python -m uvicorn`.
import asr_manager

app = FastAPI(
    title="Nusurvivors ASR Server",
    description="ASR service for transcribing audio from base64 input.",
    version="1.0.0",
    docs_url="/docs", # Provides Swagger UI for testing
    redoc_url="/redoc" # Provides ReDoc UI
)

# Initialize ASRManager globally (once when the server starts).
# This will load the model into memory.
print("[ASRServer INFO] Initializing ASRManager instance...")
try:
    manager = asr_manager.ASRManager()
    print("[ASRServer INFO] ASRManager instance created.")
except Exception as e:
    print(f"[ASRServer ERROR] Failed to initialize ASRManager: {e}")
    print(f"[ASRServer ERROR] Detailed traceback:\n{traceback.format_exc()}")
    print("[ASRServer ERROR] Exiting application due to ASR manager initialization failure.")
    # For a robust server, you might set a flag here and return 503 on requests.
    # For a contest, crashing early if core component fails is often acceptable.
    manager = None # Ensure manager is None if initialization fails

# Pydantic models for request and response
class Instance(BaseModel):
    key: int # The 'key' from the input JSON
    b64: str = Field(..., example="data:audio/wav;base64,UklGRi... (base64 encoded WAV audio)")

class Prediction(BaseModel):
    key: int # The 'key' from the input, mapped to its prediction
    transcription: str # The predicted text transcription

class RequestPayload(BaseModel):
    instances: List[Instance] # A list of audio instances

class ResponsePayload(BaseModel):
    predictions: List[Prediction] # A list of predictions matching the input order

print("[ASRServer INFO] ASR Server script loaded. FastAPI app configured.")

@app.get("/")
async def root():
    """Root endpoint for basic server information."""
    return {"message": "Nusurvivors ASR API. Visit /docs for more information."}

@app.post("/asr", response_model=ResponsePayload)
async def asr_endpoint(payload: RequestPayload):
    """
    Handles ASR transcription requests for multiple audio instances in a single batch.
    """
    print("[ASRServer INFO] /asr endpoint hit.")

    # Check if ASRManager initialized successfully
    if not manager:
        print("[ASRServer ERROR] ASRManager is not initialized. Cannot process request.")
        raise HTTPException(status_code=503, detail="ASR service is not ready. Model failed to load.")

    # Handle empty instances list
    if not payload.instances:
        print("[ASRServer WARNING] No instances provided in the request payload. Returning empty predictions.")
        return ResponsePayload(predictions=[])

    start_total_time = time.time()

    # Prepare lists to hold decoded audio bytes and their original keys, maintaining order
    audio_data_for_batch: List[bytes] = []
    original_keys: List[int] = []
    # This list will store placeholders for errors during base64 decoding,
    # so we can maintain the correct order in the final response.
    pre_processing_errors: Dict[int, str] = {} # {original_key: error_message}

    # Step 1: Decode all base64 audio inputs from the request payload
    for i, instance in enumerate(payload.instances):
        try:
            # Handle potential data URI prefix (e.g., "data:audio/wav;base64,")
            b64_string = instance.b64
            if "," in b64_string:
                header, b64_string = b64_string.split(",", 1)
                # print(f"[ASRServer INFO] Instance {i+1}: Data URI prefix detected: {header}") # Too verbose for production

            decoded_audio_bytes = base64.b64decode(b64_string)
            print(f"[ASRServer INFO] Instance {i+1}: Decoded base64 audio, bytes length: {len(decoded_audio_bytes)}")

            audio_data_for_batch.append(decoded_audio_bytes)
            original_keys.append(instance.key)

        except Exception as e:
            # If base64 decoding fails for an instance, record the error and use a placeholder.
            # This allows the batch processing to continue for other valid instances.
            error_msg = f"ERROR_DECODING_BASE64: {str(e)}"
            print(f"[ASRServer ERROR] Instance {i+1}: Error decoding base64 audio for key {instance.key}: {e}")
            audio_data_for_batch.append(b"") # Append empty bytes or a special marker for the manager to handle
            original_keys.append(instance.key)
            pre_processing_errors[instance.key] = error_msg

    # Step 2: Call the ASRManager's batch processing method
    transcriptions_from_batch: List[str] = []
    try:
        # The manager.asr_batch method expects a list of bytes and returns a list of strings.
        # It handles its own internal error messages for individual audio files.
        transcriptions_from_batch = manager.asr_batch(audio_data_for_batch)

    except Exception as e:
        # This catches unexpected errors that occur during the entire batch call within ASRManager.
        print(f"[ASRServer ERROR] Unhandled error during ASR batch processing in manager: {e}")
        print(f"[ASRServer ERROR] Detailed traceback:\n{traceback.format_exc()}")
        # If the entire batch call fails, return a 500 error for the whole request.
        raise HTTPException(status_code=500, detail=f"Internal ASR batch processing error: {e}")

    # Step 3: Construct the final response payload, mapping transcriptions back to original keys
    predictions_list: List[Prediction] = []
    for i, key in enumerate(original_keys):
        # Prioritize pre-processing errors if they occurred for this key
        if key in pre_processing_errors:
            predictions_list.append(Prediction(key=key, transcription=pre_processing_errors[key]))
        elif i < len(transcriptions_from_batch):
            # Use the transcription from the batch output
            transcription_text = transcriptions_from_batch[i]
            predictions_list.append(Prediction(key=key, transcription=transcription_text))
        else:
            # Fallback for unexpected mismatches (shouldn't happen if lengths match)
            predictions_list.append(Prediction(key=key, transcription="ERROR_UNEXPECTED_BATCH_MISMATCH"))

    end_total_time = time.time()
    print(f"[ASRServer INFO] Total processing time for {len(payload.instances)} instances: {end_total_time - start_total_time:.2f} seconds.")

    # Return the structured response
    return ResponsePayload(predictions=predictions_list)

@app.get("/health")
def health() -> dict[str, str]:
    print("[ASRServer INFO] /health endpoint hit.")
    if manager and manager.model and manager.processor:
        return {"message": "health ok", "model_status": "loaded"} # <-- Fixed line
    else:
        print("[ASRServer WARNING] Health check: Model or processor not loaded.")
        return {"message": "health degraded", "model_status": "error or not loaded"}
