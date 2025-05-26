# Dockerfile for a simplified PyTesseract-only OCR image.

# --- Base Image ---
# Using a lightweight Python 3.10 image.
# This is sufficient for PyTesseract (which runs on CPU) and Flask/FastAPI.
FROM python:3.10-slim-buster

# --- Configures settings for the image ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    OMP_NUM_THREADS=8  

# --- Working Directory ---
# Set the working directory inside the container to /app.
WORKDIR /app

# --- System Dependencies ---
# Update apt-get and install Tesseract OCR engine and its English language data.
# Also install libglib2.0-0, which is a common dependency for Pillow/image processing.
# `rm -rf /var/lib/apt/lists/*` cleans up apt caches to reduce image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# Upgrade pip to the latest version.
# Copy requirements.txt into the container.
# Install Python dependencies from requirements.txt.
# `--no-cache-dir` prevents pip from storing cache, reducing image size.
COPY requirements.txt .
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Application Source Code ---
# Create the /app/src directory.
RUN mkdir -p /app/src/

# Copy your application's source files into the container.
# Ensure your local project structure has `src/ocr_manager.py`, `src/ocr_server.py`, `src/__init__.py`.
COPY src/ocr_manager.py /app/src/ocr_manager.py
COPY src/ocr_server.py /app/src/ocr_server.py
COPY src/__init__.py /app/src/__init__.py

# --- Port Exposure ---
# Expose port 5003, indicating that the application inside the container listens on this port.
EXPOSE 5003

# --- Container Startup Command ---
# Define the command to run when the container starts.
# This uses Uvicorn to serve your FastAPI/Flask application.
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]