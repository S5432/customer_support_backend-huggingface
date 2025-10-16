# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies first (for faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 7860 (Hugging Face default for Spaces)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "7860"]
