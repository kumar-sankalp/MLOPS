"""Data loading and the reusable preprocessing pipeline.

The same ``ColumnTransformer`` is used for training and inference, which
guarantees that the transformations applied at serving time exactly match
those seen during training (a core reproducibility requirement).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    df = pd.read_csv(path)
    missing = set(config.ALL_FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (feature frame) and y (target series)."""
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used for both training and inference.

    Numeric features  -> median imputation + standard scaling.
    Categorical feats -> most-frequent imputation + one-hot encoding.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
