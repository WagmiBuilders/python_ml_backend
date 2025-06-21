FROM python:3.10-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl ca-certificates libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set work directory
WORKDIR /app

# Copy requirements file first for layer caching
COPY requirements.txt .

# Pre-install TensorFlow & Keras to cache it separately
RUN pip install --upgrade pip && \
    pip install --no-cache-dir tensorflow

# Install Python packages using pip
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Create data directory with write permissions
RUN mkdir -p /app/data && \
    touch /app/data/data.csv && \
    chown -R appuser:appuser /app/data && \
    chmod -R 777 /app/data

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]