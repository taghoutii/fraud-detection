"""
Loads the Stage 5 model and the Stage 6 decision threshold, once per process
(module-level, so every importer shares the same objects). Read-only: nothing
here writes to models/ or reports/results/.

Paths can be overridden with MODEL_PATH / REPORT_PATH env vars.
"""
import json
import os
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "final_pipeline.joblib"))
REPORT_PATH = Path(os.environ.get("REPORT_PATH", ROOT / "reports" / "results" / "final_classification_report.json"))


def load_threshold(report_path: Path) -> float:
    #the threshold Stage 6 chose on validation (recall >= 60% rule); never hardcoded here
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    threshold = float(report["threshold_selection"]["chosen_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{report_path}: chosen_threshold {threshold} is not a probability in (0, 1)")
    return threshold


PIPELINE = joblib.load(MODEL_PATH)
#the exact columns, in the exact order, the pipeline was fit on -- the API schema is derived from this
FEATURE_NAMES: list[str] = list(PIPELINE.named_steps["clf"].feature_name_)
THRESHOLD: float = load_threshold(REPORT_PATH)
