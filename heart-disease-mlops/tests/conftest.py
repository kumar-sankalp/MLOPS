"""Shared pytest fixtures."""
import pandas as pd
import pytest


@pytest.fixture
def sample_record() -> dict:
    """One valid patient record covering every feature."""
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A tiny raw frame (with a missing value) for preprocessing tests."""
    return pd.DataFrame(
        [
            {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
             "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
             "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
            {"age": 45, "sex": 0, "cp": 1, "trestbps": 130, "chol": 250,
             "fbs": 0, "restecg": 1, "thalach": 170, "exang": 1,
             "oldpeak": 0.0, "slope": 1, "ca": None, "thal": 3},
        ]
    )
