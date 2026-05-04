"""
Task 3: Register champion model in MLflow Model Registry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError as e:
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(
        "Missing Python packages (e.g. mlflow). From the MLOPs_Lab_CIE folder run:\n"
        f"  {sys.executable} -m pip install -r requirements.txt\n"
        "or use the project virtualenv:\n"
        f"  {root / '.venv' / 'bin' / 'python'} src/register_model.py\n"
    ) from e

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

REGISTERED_NAME = "drillsense-drilling-rate-m-per-hr-predictor"
EXPERIMENT_NAME = "drillsense-drilling-rate-m-per-hr"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_registry_uri(db_uri)

    meta = json.loads((MODELS_DIR / "champion_meta.json").read_text(encoding="utf-8"))
    run_id = meta["run_id"]
    rmse = float(meta["rmse"])

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)

    client = MlflowClient()
    client.set_registered_model_tag(REGISTERED_NAME, "source_run_id", run_id)

    out = {
        "registered_model_name": REGISTERED_NAME,
        "version": int(mv.version),
        "run_id": run_id,
        "source_metric": "rmse",
        "source_metric_value": rmse,
    }
    (RESULTS_DIR / "step3_s6.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
