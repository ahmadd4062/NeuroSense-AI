# Python 3.10 on Debian Bookworm — matches your local env
FROM python:3.10-slim-bookworm

# Install system libraries needed by OpenCV, MediaPipe, TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    libgles2-mesa \
    libegl1-mesa \
    libglvnd-dev \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Collect static files at build time
RUN python manage.py collectstatic --noinput || true

# Expose port
EXPOSE 8000

# Default command (overridden by startup.sh in Azure)
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout", "600", "--workers", "2", "Neurosense.wsgi:application"]