# Dockerfile for building the CV image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# # Configures settings for the image.
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_ROOT_USER_ACTION=ignore
# WORKDIR /workspace

# # Installs your dependencies.
# RUN pip install -U pip
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copies your source files.
# COPY src .

# # Starts your model server.
# CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0


# (root)/Nusurvivors/cv/Dockerfile

# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Install system dependencies if any (e.g., for opencv)
# RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
# Usually not needed for opencv-python-headless with this base image

# Installs your Python dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files from Nusurvivors/cv/src into /workspace/
COPY src/ ./src/

# Copies the fine-tuned model weights
# Assuming best.pt is in Nusurvivors/cv/models/
COPY models/best.pt ./models/best.pt 
# This makes the model available at /workspace/models/best.pt inside the container

# Set environment variable for model path, can be overridden at runtime if needed
ENV MODEL_PATH=/workspace/models/best.pt

# Starts your model server.
# The cv_server.py and cv_manager.py are expected to be in ./src/ by this point
# CMD uvicorn cv_server:app --port 5002 --host 0.0.0.0
# The server should be started from within the src directory or Python path adjusted
# Given WORKDIR /workspace and COPY src . -> files are in /workspace/src
# So, the command should be relative to /workspace
CMD ["uvicorn", "src.cv_server:app", "--port", "5002", "--host", "0.0.0.0"]
