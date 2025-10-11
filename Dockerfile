# Use official Python slim image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy all files into container (include your models)
COPY . /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["python3", "app.py"]