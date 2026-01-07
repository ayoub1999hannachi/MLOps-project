# tests/test_model.py
import pytest
from app.model import DigitsModel

@pytest.fixture(scope="module")
def model_instance():
    return DigitsModel()

def test_model_info(model_instance):
    info = model_instance.get_model_info()
    assert "task" in info
    assert "accuracy" in info
    assert info["n_features"] == 64

def test_predict_valid(model_instance):
    sample_features = [0.0]*64
    result = model_instance.predict(sample_features)
    assert "prediction" in result
    assert "class_name" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == 10

def test_predict_invalid_length(model_instance):
    with pytest.raises(ValueError):
        model_instance.predict([0.0]*63)
