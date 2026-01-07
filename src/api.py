# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Dict
import logging
from app.model import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digits Classification API", version="1.0.0")

class DigitFeatures(BaseModel):
    # fixed-length list of 64 floats
    features: conlist(float, min_items=64, max_items=64) = Field(..., description="64 float features (8x8 image flattened)")

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float = None
    probabilities: Dict[str, float] = None

@app.get("/", tags=["health"])
def root():
    return {"message": "Digits classification API", "docs": "/docs"}

@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}

@app.get("/model/info", tags=["model"])
def get_model_info():
    return model.get_model_info()

@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: DigitFeatures):
    try:
        result = model.predict(payload.features)
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
