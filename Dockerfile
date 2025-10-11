# Use official Python image
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on (for Render or Docker)
EXPOSE 8000

# Command to run the app
CMD ["python3", "app.py"]