from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from housing_pipeline import build_preprocessing, make_estimator_for_name

app = FastAPI(title="Churn Prediction API")

# Load Model
MODEL_PATH = "/app/models/global_best_model_classification.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict
    try:
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
