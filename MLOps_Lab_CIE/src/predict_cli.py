"""
CLI predictor for Task 2 (Docker entrypoint).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "champion.joblib"

FEATURE_COLS = [
    "rock_hardness",
    "drill_bit_age_hours",
    "mud_pressure_psi",
    "depth_m",
]


def _frame(row: dict) -> pd.DataFrame:
    return pd.DataFrame([row], columns=FEATURE_COLS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DrillSense drilling rate predictor")
    p.add_argument("--rock_hardness", type=float, required=True)
    p.add_argument("--drill_bit_age_hours", type=float, required=True)
    p.add_argument("--mud_pressure_psi", type=float, required=True)
    p.add_argument("--depth_m", type=float, required=True)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--write-step2-result",
        action="store_true",
        help="Write results/step2_s3.json for the lab fixed test input.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    model = joblib.load(args.model_path)
    X = _frame(
        {
            "rock_hardness": args.rock_hardness,
            "drill_bit_age_hours": args.drill_bit_age_hours,
            "mud_pressure_psi": args.mud_pressure_psi,
            "depth_m": args.depth_m,
        }
    )
    pred = float(model.predict(X)[0])
    print(pred)

    if args.write_step2_result:
        fixed = {
            "rock_hardness": 4,
            "drill_bit_age_hours": 81,
            "mud_pressure_psi": 1457.3,
            "depth_m": 1891.2,
        }
        pred_fixed = float(model.predict(_frame(fixed))[0])
        out = {
            "image_name": "drillsense-predictor",
            "image_tag": "v1",
            "base_image": "python:3.12-slim",
            "test_input": fixed,
            "prediction": pred_fixed,
        }
        out_path = ROOT / "results" / "step2_s3.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
