FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python requirements (make sure scikit-learn version matches!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW: Copy the Model directly into the image ---
COPY models/global_best_model_classification.pkl /app/models/global_best_model_classification.pkl

# Copy code
COPY housing_pipeline.py .
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]