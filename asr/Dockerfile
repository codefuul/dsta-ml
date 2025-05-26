# # ~/Nusurvivors/asr/Dockerfile

# FROM python:3.12-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_ROOT_USER_ACTION=ignore
# WORKDIR /workspace

# RUN pip install -U pip
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copies your Python source files from local 'src/' to '/workspace/'
# COPY src/ /workspace/

# # --- ASR Model Setup (NEW SECTION) ---
# # Create the target directory for the ASR model within the container
# RUN mkdir -p /app/models/wav2vec2-base-ft-novice-asr/

# # Copies your fine-tuned ASR model from local 'models/wav2vec2-base-ft-novice-asr/'
# # into '/app/models/wav2vec2-base-ft-novice-asr/' in the container.
# COPY models/wav2vec2-base-ft-novice-asr/ /app/models/wav2vec2-base-ft-novice-asr/

# # Set the environment variable that asr_manager.py will use
# ENV ASR_TRAINED_MODEL_PATH=/app/models/wav2vec2-base-ft-novice-asr
# # --- End of ASR Model Setup ---

# # Starts your ASR model server.
# CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]



# /home/jupyter/Nusurvivors/asr/Dockerfile

# FROM python:3.12-slim
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-3.py310

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Copies your source files from asr/src to /workspace
COPY src .

# Copy the locally downloaded models directory from asr/models to /workspace/models
#COPY models ./models/
RUN mkdir -p /app/models/local_whisper_base # Ensure target directory exists
COPY models/local_whisper_base /app/models/local_whisper_base/

# Starts your model server.
CMD ["python", "-m", "uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]

# Dockerfile for building the ASR image.

# FROM python:3.12-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_ROOT_USER_ACTION=ignore

# WORKDIR /workspace

# # Install system dependencies if any (e.g., for audio processing libs)
# # RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY src/ .
# COPY models/ ./models/

# EXPOSE 5001

# CMD ["uvicorn", "asr_manager:app", "--host", "0.0.0.0", "--port", "5001"]

