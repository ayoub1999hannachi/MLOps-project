# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api import app  # src must have __init__.py

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "task" in data
    assert "n_features" in data

def test_predict_valid():
    sample = {"features": [0.0]*64}
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "class_name" in data
    assert "confidence" in data
    assert "probabilities" in data

def test_predict_invalid_length():
    # Send too few features
    sample = {"features": [0.0]*63}
    response = client.post("/predict", json=sample)
    assert response.status_code == 422  # <-- change from 400 to 422

