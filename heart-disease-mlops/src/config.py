"""Central configuration: paths, feature groups and constants.

Keeping these in one place makes the preprocessing, training and serving
code consistent and reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
NOTEBOOK_FIG_DIR = ROOT / "notebooks" / "figures"

DATA_PATH = DATA_DIR / "heart_disease_clean.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"          # full sklearn Pipeline
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# --- Columns -----------------------------------------------------------------
TARGET = "target"

# Continuous features -> imputed (median) + standard-scaled.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete / categorical features -> imputed (most frequent) + one-hot encoded.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- MLflow ------------------------------------------------------------------
# MLflow 3.x deprecated the bare file store, so we use a SQLite backend for
# metadata and a local directory for artifacts. View the UI with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
MLFLOW_EXPERIMENT = "heart-disease-classification"
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (ROOT / "mlartifacts").as_uri()
