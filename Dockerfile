# api/Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the API with Uvicorn
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000"]
