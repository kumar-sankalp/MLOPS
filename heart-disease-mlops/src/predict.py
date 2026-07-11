"""Inference helper: load the packaged pipeline and score patient records.

The saved artifact is a full sklearn Pipeline (preprocessing + classifier),
so raw feature values can be passed straight in - no manual preprocessing.
"""
from __future__ import annotations

import functools
from typing import Dict

import joblib
import pandas as pd

from src import config


@functools.lru_cache(maxsize=1)
def load_model(model_path: str | None = None):
    """Load and cache the serialised pipeline."""
    path = model_path or config.MODEL_PATH
    return joblib.load(path)


def predict_one(record: Dict, model=None) -> Dict:
    """Score a single patient record.

    Parameters
    ----------
    record : dict
        Mapping of feature name -> value (see config.ALL_FEATURES).
    model : optional
        A pre-loaded pipeline; if None the cached model is used.

    Returns
    -------
    dict with keys: prediction (0/1), label, probability (of disease).
    """
    model = model or load_model()
    X = pd.DataFrame([{f: record.get(f) for f in config.ALL_FEATURES}])

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": "Heart Disease" if pred == 1 else "No Heart Disease",
        "probability": round(proba, 4),
    }
