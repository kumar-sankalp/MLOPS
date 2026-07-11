"""Train, tune, track and package heart-disease classifiers.

For each candidate model we:
  * build a full sklearn Pipeline (preprocessor + classifier),
  * tune hyper-parameters with GridSearchCV (5-fold CV),
  * evaluate on a held-out test set,
  * log params/metrics/plots/model to MLflow.

The best model (by test ROC-AUC) is serialised with joblib to models/.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve

# Optional: XGBoost (falls back gracefully if not installed).
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # noqa: BLE001
    _HAS_XGB = False


def get_model_space():
    """Return {name: (estimator, param_grid)} for each candidate model.

    Grid keys are prefixed with ``clf__`` because the classifier is the
    ``clf`` step of the full pipeline.
    """
    space = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
            },
        ),
    }

    if _HAS_XGB:
        space["xgboost"] = (
            XGBClassifier(
                random_state=config.RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        )
    return space


def train_one(name, estimator, param_grid, X_train, y_train, X_test, y_test):
    """Train + tune one model, log to MLflow, return (pipeline, metrics)."""
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("clf", estimator)]
    )

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=config.CV_FOLDS,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Cross-validated score on the training folds (mean ROC-AUC).
        cv_scores = cross_val_score(
            best, X_train, y_train, cv=config.CV_FOLDS, scoring="roc_auc"
        )

        # Held-out test evaluation.
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["cv_roc_auc_mean"] = float(np.mean(cv_scores))
        metrics["cv_roc_auc_std"] = float(np.std(cv_scores))

        # --- MLflow logging ---
        mlflow.log_param("model", name)
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            roc_path = plot_roc_curve(y_test, y_proba, name, tmp / "roc_curve.png")
            cm_path = plot_confusion_matrix(
                y_test, y_pred, name, tmp / "confusion_matrix.png"
            )
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

        mlflow.sklearn.log_model(
            best, name="model", serialization_format="cloudpickle"
        )

        print(
            f"[{name}] test ROC-AUC={metrics['roc_auc']:.3f} "
            f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
            f"cv_auc={metrics['cv_roc_auc_mean']:.3f}"
        )
        return best, metrics


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=config.MLFLOW_ARTIFACT_LOCATION,
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    results = {}
    fitted = {}
    for name, (estimator, grid) in get_model_space().items():
        model, metrics = train_one(
            name, estimator, grid, X_train, y_train, X_test, y_test
        )
        results[name] = metrics
        fitted[name] = model

    # Select the best model by test ROC-AUC.
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted[best_name]
    print(f"\n[select] Best model: {best_name} "
          f"(ROC-AUC={results[best_name]['roc_auc']:.3f})")

    # Persist the full pipeline + metadata.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)
    metadata = {
        "best_model": best_name,
        "metrics": results[best_name],
        "all_results": results,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"[save] Model  -> {config.MODEL_PATH}")
    print(f"[save] Meta   -> {config.METADATA_PATH}")


if __name__ == "__main__":
    main()
