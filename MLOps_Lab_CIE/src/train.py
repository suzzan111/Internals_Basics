"""
Task 1: Train LinearRegression and Ridge with MLflow tracking; select best by RMSE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError as e:
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(
        "Missing Python packages (e.g. mlflow). From the MLOPs_Lab_CIE folder run:\n"
        f"  {sys.executable} -m pip install -r requirements.txt\n"
        "or use the project virtualenv:\n"
        f"  {root / '.venv' / 'bin' / 'python'} src/train.py\n"
    ) from e
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

FEATURE_COLS = [
    "rock_hardness",
    "drill_bit_age_hours",
    "mud_pressure_psi",
    "depth_m",
]
TARGET_COL = "drilling_rate_m_per_hr"
EXPERIMENT_NAME = "drillsense-drilling-rate-m-per-hr"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_registry_uri(db_uri)

    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        art = (ROOT / "mlartifacts").resolve()
        art.mkdir(parents=True, exist_ok=True)
        client.create_experiment(EXPERIMENT_NAME, artifact_location=art.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_DIR / "training_data.csv")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models_spec = [
        ("LinearRegression", LinearRegression, {"fit_intercept": True}),
        ("Ridge", Ridge, {"alpha": 1.0, "fit_intercept": True}),
    ]

    summaries: list[dict] = []
    best_name: str | None = None
    best_rmse = float("inf")
    best_run_id: str | None = None
    best_estimator = None

    for name, model_cls, params in models_spec:
        estimator = model_cls()
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("experiment_type", "baseline_comparison")
            mlflow.log_params(params)
            estimator.set_params(**{k: v for k, v in params.items() if hasattr(estimator, k)})
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            mae = float(mean_absolute_error(y_test, pred))
            rmse = _rmse(y_test.values, pred)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(estimator, name="model")

            summaries.append({"name": name, "mae": mae, "rmse": rmse})

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_run_id = run.info.run_id
                best_estimator = estimator

    assert best_name is not None and best_run_id is not None and best_estimator is not None

    joblib.dump(best_estimator, MODELS_DIR / "champion.joblib")
    meta = {
        "model_name": best_name,
        "run_id": best_run_id,
        "rmse": best_rmse,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    (MODELS_DIR / "champion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out = {
        "experiment_name": EXPERIMENT_NAME,
        "models": summaries,
        "best_model": best_name,
        "best_metric_name": "rmse",
        "best_metric_value": best_rmse,
    }
    (RESULTS_DIR / "step1_s1.json").write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
