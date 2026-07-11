"""
Download and clean the Heart Disease UCI dataset.

Source : UCI Machine Learning Repository
         https://archive.ics.uci.edu/dataset/45/heart+disease
Primary : processed.cleveland.data (303 rows, 14 attributes)

The script:
  1. Downloads the raw Cleveland data (with an ``ucimlrepo`` fallback).
  2. Assigns proper column names.
  3. Replaces the UCI missing-value marker '?' with NaN.
  4. Binarises the target (0 = no disease, 1..4 -> 1 = disease present).
  5. Writes a cleaned CSV to ``data/heart_disease_clean.csv``.

Run:  python data/download_data.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

# --- Configuration -----------------------------------------------------------
RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# 13 features + raw target ("num")
COLUMN_NAMES = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1/0)
    "restecg",   # resting ECG results (0-2)
    "thalach",   # max heart rate achieved
    "exang",     # exercise-induced angina (1/0)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment (1-3)
    "ca",        # number of major vessels (0-3) coloured by fluoroscopy
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # raw diagnosis (0-4)
]

OUT_PATH = Path(__file__).resolve().parent / "heart_disease_clean.csv"


def _download_raw() -> pd.DataFrame:
    """Fetch raw Cleveland data, trying a direct URL then ucimlrepo."""
    # 1) Direct download from the UCI file server.
    try:
        import urllib.request

        print(f"[download] Fetching from {RAW_URL}")
        with urllib.request.urlopen(RAW_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), header=None, names=COLUMN_NAMES)
        print(f"[download] OK - {len(df)} rows via direct URL")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download] Direct URL failed ({exc}). Trying ucimlrepo ...")

    # 2) Fallback: ucimlrepo package (dataset id 45).
    try:
        from ucimlrepo import fetch_ucirepo

        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = COLUMN_NAMES
        print(f"[download] OK - {len(df)} rows via ucimlrepo")
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[download] ucimlrepo failed ({exc}).", file=sys.stderr)
        raise SystemExit(
            "Could not download the dataset. Check your internet connection, "
            "or `pip install ucimlrepo` and retry."
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Cleveland data into a modelling-ready frame."""
    # UCI marks missing values with '?'.
    df = df.replace("?", pd.NA)

    # Every column is numeric in this dataset; coerce.
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarise target: 0 -> 0 (no disease), 1-4 -> 1 (disease present).
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df


def main() -> None:
    df = _download_raw()
    df = clean(df)

    n_missing = int(df.isna().sum().sum())
    print(f"[clean] shape={df.shape}, missing cells={n_missing}")
    print(f"[clean] target balance:\n{df['target'].value_counts().to_string()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[save] Wrote cleaned dataset -> {OUT_PATH}")


if __name__ == "__main__":
    main()
