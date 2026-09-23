FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-airflow.txt ./

# Install Python dependencies (Airflow is included in the shared image so
# airflow-webserver/airflow-scheduler compose services can run it directly)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-airflow.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external data/interim \
    models/trained models/artifacts \
    logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for FastAPI
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
