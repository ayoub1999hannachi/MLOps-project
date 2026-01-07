# app/model.py
import joblib
import json
import numpy as np
from typing import List, Dict, Any
import os

class DigitsModel:
    """Load and predict digits using RandomForest"""

    def __init__(self, model_path="app/digits_model.joblib", metadata_path="app/model_metadata.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run src/train.py first.")
        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.n_features = int(self.model.n_features_in_)

    def predict(self, features: List[float]) -> Dict[str, Any]:
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
        arr = np.array(features).reshape(1, -1)
        pred = int(self.model.predict(arr)[0])
        probs = self.model.predict_proba(arr)[0] if hasattr(self.model, "predict_proba") else None
        confidence = float(max(probs)) if probs is not None else None
        probabilities = {str(i): float(probs[i]) for i in range(len(probs))} if probs is not None else {}
        return {
            "prediction": pred,
            "class_name": self.metadata.get("target_names", [str(i) for i in range(10)])[pred],
            "confidence": confidence,
            "probabilities": probabilities
        }

    def get_model_info(self) -> Dict[str, Any]:
        info = dict(self.metadata)
        info.update({"n_features": self.n_features})
        return info

# Single global instance
model = DigitsModel()
