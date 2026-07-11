"""Integration tests for the FastAPI service."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="Model not trained yet; run `python -m src.train` first.",
)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict_valid(sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("Heart Disease", "No Heart Disease")


def test_predict_missing_field_returns_422(sample_record):
    bad = dict(sample_record)
    del bad["age"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # pydantic validation error


def test_metrics_endpoint():
    client.post("/predict", json={
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    })
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "predictions_total" in r.text
