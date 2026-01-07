# src/train.py
import os
import json
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(output_dir="app"):
    """Train a RandomForest on digits dataset and save model and metadata"""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading digits dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model inside app/
    model_path = os.path.join(output_dir, "digits_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    metadata = {
        "task": "Digits classification",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "target_names": digits.target_names.tolist(),
        "accuracy": float(acc)
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata

if __name__ == "__main__":
    train_model()
