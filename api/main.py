"""FastAPI service for heart-disease risk prediction.

Endpoints
---------
GET  /            - service metadata
GET  /health      - liveness/readiness probe
POST /predict     - JSON patient record -> prediction + confidence
GET  /metrics     - Prometheus metrics (request counts + latency)

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src import config
from src.predict import load_model, predict_one

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")

# --- Prometheus metrics ------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (seconds)",
    ["endpoint"],
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions by predicted class",
    ["predicted_label"],
)

# --- App ---------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the model cache on startup so the first request isn't slow."""
    try:
        load_model()
        logger.info("Model loaded successfully from %s", config.MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load model at startup: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart-disease risk from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    """Input schema. Field descriptions mirror the UCI dataset."""

    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="1 = male, 0 = female")
    cp: int = Field(..., description="Chest pain type (0-3/1-4)")
    trestbps: float = Field(..., description="Resting BP (mm Hg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar >120 (1/0)")
    restecg: int = Field(..., description="Resting ECG result (0-2)")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise-induced angina (1/0)")
    oldpeak: float = Field(..., description="ST depression")
    slope: int = Field(..., description="Slope of peak ST segment")
    ca: float = Field(..., description="Major vessels (0-3)")
    thal: float = Field(..., description="Thalassemia (3/6/7)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request latency, counts and log every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        http_status=response.status_code,
    ).inc()
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method, endpoint, response.status_code, elapsed * 1000,
    )
    return response


@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    """Return 200 if the model is loadable."""
    try:
        load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False, "error": str(exc)},
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    """Predict heart-disease risk for one patient record."""
    result = predict_one(features.model_dump())
    PREDICTION_COUNT.labels(predicted_label=result["label"]).inc()
    logger.info("Prediction: %s (p=%.4f)", result["label"], result["probability"])
    return PredictionResponse(**result)


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
