# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# [GOD-LEVEL FIX] 
# Updated for Debian Bookworm/Trixie compliance.
# Replaced 'libgl1-mesa-glx' with 'libgl1' and 'libglx-mesa0'.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run the nervous system (Bind to 0.0.0.0 for Docker Networking)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]