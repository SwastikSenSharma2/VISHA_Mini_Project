FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (if any) and Python packages
RUN pip install --no-cache-dir -q fastapi uvicorn[standard] python-multipart torch

# Expose the API port
EXPOSE 8000

# Environment variable for taxonomy checkpoint (relative to /app)
ENV SNAKE_TAXONOMY_CHECKPOINT=ml/artifacts/taxonomy_classifier_latest.pt

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
