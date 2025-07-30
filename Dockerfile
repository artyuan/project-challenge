FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the API
CMD ["uvicorn", "model_endpoint:app", "--host", "0.0.0.0", "--port", "8000"]