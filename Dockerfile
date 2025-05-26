FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV GRADIO_ANALYTICS_ENABLED=False
ENV GRADIO_TELEMETRY_ENABLED=False
ENV GRADIO_SHARE=False
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"] 