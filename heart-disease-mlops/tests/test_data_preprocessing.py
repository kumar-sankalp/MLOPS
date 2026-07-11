"""Unit tests for data loading and the preprocessing pipeline."""
import numpy as np

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)


def test_load_data_has_expected_columns():
    df = load_data()
    for col in config.ALL_FEATURES + [config.TARGET]:
        assert col in df.columns


def test_target_is_binary():
    df = load_data()
    assert set(df[config.TARGET].unique()).issubset({0, 1})


def test_split_features_target_shapes():
    df = load_data()
    X, y = split_features_target(df)
    assert X.shape[0] == y.shape[0]
    assert list(X.columns) == config.ALL_FEATURES


def test_preprocessor_handles_missing_values(raw_frame):
    """ColumnTransformer must impute the NaN in 'ca' without error."""
    assert raw_frame["ca"].isna().any()  # sanity: there IS a missing value
    pre = build_preprocessor()
    transformed = pre.fit_transform(raw_frame)
    # No NaNs should remain after imputation.
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()


def test_preprocessor_output_is_2d(raw_frame):
    pre = build_preprocessor()
    out = pre.fit_transform(raw_frame)
    arr = out.toarray() if hasattr(out, "toarray") else out
    assert arr.ndim == 2
    assert arr.shape[0] == len(raw_frame)


def test_numeric_features_are_scaled(raw_frame):
    """Numeric block should be standard-scaled (roughly zero-mean per column)."""
    pre = build_preprocessor()
    pre.fit(raw_frame)
    # The numeric transformer is the first (index 0).
    num_out = pre.named_transformers_["num"].transform(
        raw_frame[config.NUMERIC_FEATURES]
    )
    assert np.allclose(num_out.mean(axis=0), 0, atol=1e-6)
