# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import Dict
import logging
from app.model import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Digits Classification API", version="1.0.0")

# Fixed for Pydantic v2: use min_length/max_length
class DigitFeatures(BaseModel):
    features: conlist(float, min_length=64, max_length=64)

@app.get("/")
def root():
    return {"message": "Digits classification API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model/info")
def model_info():
    return model.get_model_info()

@app.post("/predict")
def predict(payload: DigitFeatures):
    try:
        return model.predict(payload.features)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
