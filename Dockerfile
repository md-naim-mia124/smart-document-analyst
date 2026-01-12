# Dockerfile

# 1. Base Image: Using Python 3.9 to match your development environment
FROM python:3.9-slim

# 2. Install system dependencies
# Added 'curl' because it is required for the HEALTHCHECK command below
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements file
COPY requirements.txt .

# 5. Upgrade pip and install Python dependencies
# --no-cache-dir is used to keep the image size small
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Create empty directories for volume mounting
# These will be mapped to your local folders when running the container
RUN mkdir -p models indexes uploads data

# 8. Expose the port Streamlit runs on
EXPOSE 8501

# 9. Healthcheck to ensure the app is running correctly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]