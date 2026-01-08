# 🧠 Digits Classification MLOps Project

![CI](https://github.com/USERNAME/MLOps-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A production‑ready **MLOps project** for handwritten digits classification using **scikit‑learn**, **FastAPI**, **Docker**, and **GitHub Actions**.
This project demonstrates the full ML lifecycle: **training → testing → API serving → CI/CD → containerization**.

---

## 🚀 Project Overview

* **Task**: Handwritten digits (0–9) classification
* **Dataset**: `sklearn.datasets.load_digits`
* **Model**: RandomForestClassifier
* **API**: FastAPI
* **Testing**: Pytest
* **CI/CD**: GitHub Actions
* **Containerization**: Docker

---

## 📂 Project Structure

```text
mlops-project/
│
├── app/
│   ├── __init__.py
│   └── model.py              # Model loading, prediction logic
│
├── src/
│   ├── __init__.py
│   ├── train.py               # Model training script
│   └── api.py                 # FastAPI application
│
├── tests/
│   ├── test_api.py            # API endpoint tests
│   └── test_model.py          # Model unit tests
│
├── .github/workflows/
│   └── ci.yml                 # GitHub Actions CI pipeline
│
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore
```

---

## ⚙️ Installation (Local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python src/train.py
```

This will generate:

* `app/digits_model.joblib`
* `app/model_metadata.json`

---

## 🌐 Run the API

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

| Method | Endpoint      | Description      |
| ------ | ------------- | ---------------- |
| GET    | `/`           | API info         |
| GET    | `/health`     | Health check     |
| GET    | `/model/info` | Model metadata   |
| POST   | `/predict`    | Digit prediction |

Example request:

```json
{
  "features": [0.0, 0.1, ..., 16.0]
}
```

---

## 🧪 Run Tests

```bash
PYTHONPATH=. pytest -v tests/
```

Tests cover:

* Model loading & prediction
* API endpoints
* Input validation

---

## 🐳 Docker

### Build image

```bash
docker build -t digits-ml-api:1.0 .
```

### Run container

```bash
docker run -p 8000:8000 digits-ml-api:1.0
```

The model is **trained automatically during image build**.

---

## 🔁 CI/CD (GitHub Actions)

On every **push or pull request**, GitHub Actions will:

1. Install dependencies
2. Train the model
3. Run all tests

Workflow file:

```
.github/workflows/ci.yml
```

---

## 📦 Dependencies

* Python 3.10
* fastapi
* uvicorn
* scikit-learn
* joblib
* pydantic
* pytest

---

## 🎯 MLOps Best Practices Applied

* Reproducible training pipeline
* Automated testing
* CI validation before merge
* Dockerized deployment
* Clean project structure

---

## 👤 Author

**Hannachi Ayoub**
MLOps / Data Science Student

---

## 📜 License

This project is for educational and academic purposes.
