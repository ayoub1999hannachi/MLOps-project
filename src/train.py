# src/train.py
import os
import json
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_model(output_dir="../app"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading digits dataset...")
    digits = load_digits()
    X = digits.data           # shape (n_samples, 64)
    y = digits.target         # 0..9

    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=None)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(output_dir, "digits_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    metadata = {
        "task": "Digits classification",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_names": digits.target_names.tolist(),
        "accuracy": float(acc)
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata

if __name__ == "__main__":
    train_model()
