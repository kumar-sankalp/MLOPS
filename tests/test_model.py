"""Unit tests for the trained model and inference helper."""
import pytest

from src import config
from src.predict import load_model, predict_one

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Model not trained yet; run `python -m src.train` first.",
)


def test_model_loads():
    model = load_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_one_returns_valid_schema(sample_record):
    result = predict_one(sample_record)
    assert set(result) == {"prediction", "label", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in ("Heart Disease", "No Heart Disease")


def test_prediction_is_deterministic(sample_record):
    r1 = predict_one(sample_record)
    r2 = predict_one(sample_record)
    assert r1 == r2


def test_high_risk_record_scores_higher(sample_record):
    """A clearly higher-risk profile should not score lower than a low-risk one."""
    model = load_model()
    low_risk = dict(sample_record)
    low_risk.update({"age": 29, "cp": 0, "thalach": 190, "oldpeak": 0.0,
                     "exang": 0, "ca": 0})
    high_risk = dict(sample_record)
    high_risk.update({"age": 67, "cp": 3, "thalach": 100, "oldpeak": 4.0,
                      "exang": 1, "ca": 3})
    p_low = predict_one(low_risk, model)["probability"]
    p_high = predict_one(high_risk, model)["probability"]
    assert p_high >= p_low
