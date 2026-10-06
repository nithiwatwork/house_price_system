FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY config/ /app/config/
COPY src/ /app/src/
COPY serving/ /app/serving/
COPY monitoring/ /app/monitoring/
COPY artifacts/ /app/artifacts/

# Expose HTTP port for FastAPI Serving
EXPOSE 8000

# Healthcheck to monitor container status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to start FastAPI application via Uvicorn
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
