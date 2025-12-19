# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import pandas as pd
from pathlib import Path

# Import pipeline logic
from housing_pipeline import build_preprocessing, make_estimator_for_name

app = FastAPI(title="Churn API")
model = joblib.load("/app/models/global_best_model_classification.pkl")

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.instances)
    preds = model.predict(df)
    return {"predictions": [int(p) for p in preds]}