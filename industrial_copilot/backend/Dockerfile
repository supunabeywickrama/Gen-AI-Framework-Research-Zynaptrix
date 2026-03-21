FROM python:3.10-slim

WORKDIR /app

# Install native dependencies required by psycopg2, pandas, and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create physical directories to mount if necessary
RUN mkdir -p data/mock_data data/processed

# Copy local source code and weights
COPY . .

# Set native PYTHONPATH for internal absolute imports
ENV PYTHONPATH=/app

# Command is intended to be overridden by docker-compose
CMD ["python", "-m", "uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "8000"]
