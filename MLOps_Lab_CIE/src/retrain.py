"""
Task 4: Combine datasets, retrain winning model type, compare RMSE on same test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
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
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_IMPROVEMENT = 0.3


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_model(name: str):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "Ridge":
        r = Ridge()
        r.set_params(alpha=1.0, fit_intercept=True)
        return r
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((MODELS_DIR / "champion_meta.json").read_text(encoding="utf-8"))
    champion_name = meta["model_name"]

    train_df = pd.read_csv(DATA_DIR / "training_data.csv")
    new_df = pd.read_csv(DATA_DIR / "new_data.csv")

    n_orig = len(train_df)
    n_new = len(new_df)
    combined = pd.concat([train_df, new_df], axis=0, ignore_index=True)

    X_all = train_df[FEATURE_COLS]
    y_all = train_df[TARGET_COL]
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    champion = joblib.load(MODELS_DIR / "champion.joblib")
    champ_pred = champion.predict(X_test)
    champion_rmse = _rmse(y_test.values, champ_pred)

    X_extra = new_df[FEATURE_COLS]
    y_extra = new_df[TARGET_COL]
    X_retrain = pd.concat([X_train_orig, X_extra], axis=0, ignore_index=True)
    y_retrain = pd.concat([y_train_orig, y_extra], axis=0, ignore_index=True)

    retrained = _make_model(champion_name)
    retrained.fit(X_retrain, y_retrain)
    retr_pred = retrained.predict(X_test)
    retrained_rmse = _rmse(y_test.values, retr_pred)

    improvement = champion_rmse - retrained_rmse
    action = "promoted" if improvement >= MIN_IMPROVEMENT else "kept_champion"

    out = {
        "original_data_rows": n_orig,
        "new_data_rows": n_new,
        "combined_data_rows": len(combined),
        "champion_rmse": champion_rmse,
        "retrained_rmse": retrained_rmse,
        "improvement": improvement,
        "min_improvement_threshold": MIN_IMPROVEMENT,
        "action": action,
        "comparison_metric": "rmse",
    }
    (RESULTS_DIR / "step4_s8.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
