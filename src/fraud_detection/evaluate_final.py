"""
evaluation + explainability. 
Everything is saved under reports/results/ for the dashboard to load:
    test_predictions.parquet          one row per test txn: ids, true label, probability, label at 0.5
    threshold_search.json             precision/recall/F1/flag counts across thresholds (test set)
    final_classification_report.json  chosen threshold, the assumption behind it, test metrics at it
    shap_values.parquet               SHAP value per feature for a fixed test sample (log-odds units)
    shap_sample.parquet               the same rows' feature values + label/probability, same row order
    shap_meta.json                    SHAP base value and how to read the two parquet files
    feature_importances.json          LightGBM native importances and mean |SHAP|, two ranked lists
    calibration.json                  predicted-probability bins vs observed fraud rate
"""
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.fraud_detection.dataset import NON_FEATURE_COLS, load_or_build_split
from src.fraud_detection.modeling import make_cv_splits
from src.fraud_detection.run_experiments import MIN_POS, to_jsonable, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "reports" / "results"
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"

MIN_RECALL = 0.60
THRESHOLD_ASSUMPTION = (
    f"Catch at least {MIN_RECALL:.0%} of fraud (recall >= {MIN_RECALL}); among thresholds that do, "
    "pick the one with the highest precision (fewest false alarms per flagged transaction)."
)
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2) #0.01-0.99: the tuned model's 0.5 sits deep in the high-precision end
SHAP_SAMPLE_N = 2000


# ---- pure helpers (unit-tested) ----

def threshold_table(y_true, proba, thresholds) -> list[dict]:
    #precision/recall/F1 + confusion counts at each threshold; flagged = proba >= threshold
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    n_pos = int(y_true.sum())
    rows = []
    for t in thresholds:
        flagged = proba >= t
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = n_pos - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1,
                     "n_flagged": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": int(len(y_true) - tp - fp - fn)})
    return rows


def pick_threshold(table: list[dict], min_recall: float = MIN_RECALL) -> dict:
    #highest precision among thresholds with recall >= min_recall; ties -> higher recall
    eligible = [r for r in table if r["recall"] >= min_recall]
    if not eligible:
        raise ValueError(f"no threshold reaches recall >= {min_recall}")
    return max(eligible, key=lambda r: (r["precision"], r["recall"]))


def binary_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row = threshold_table(y_true, proba, [threshold])[0]
    return {
        "threshold": float(threshold),
        "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "n_flagged": int(tp + fp), "flag_rate": float((tp + fp) / len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibration_bins(y_true, proba, n_bins: int = 10, strategy: str = "uniform") -> list[dict]:
    #reliability-curve data: per bin, mean predicted probability vs observed fraud rate
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    idx = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2) #last bin includes 1.0
    bins = []
    for b in range(len(edges) - 1):
        mask = idx == b
        count = int(mask.sum())
        bins.append({
            "bin_lower": float(edges[b]), "bin_upper": float(edges[b + 1]), "count": count,
            "mean_predicted": float(proba[mask].mean()) if count else None,
            "observed_fraud_rate": float(y_true[mask].mean()) if count else None,
        })
    return bins


def ranked_importances(names, values, key: str) -> list[dict]:
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    return [{"rank": i + 1, "feature": str(names[j]), key: float(values[j])} for i, j in enumerate(order)]


# ---- evaluation ----

def _log(msg: str, t_start: float) -> None:
    print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)


def _positive_class(values):
    #shap returns [class0, class1] for some model/version combos and the positive class directly for others
    return values[1] if isinstance(values, list) else values


def main() -> None:
    t_start = time.time()
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed, n_splits = cfg["seed"], cfg["cv"]["n_splits"]
    time_col, target_col = cfg["data"]["time_col"], cfg["data"]["target_col"]

    train, test = load_or_build_split(ROOT / cfg["data"]["processed_dir"], ROOT / cfg["data"]["raw_dir"],
                                      time_col, cfg["split"]["test_size"])
    model = joblib.load(MODEL_PATH)
    clf = model.named_steps["clf"]
    feature_cols = [c for c in test.columns if c not in [target_col] + NON_FEATURE_COLS]
    if list(clf.feature_name_) != feature_cols:
        raise ValueError("test.parquet columns don't match the features final_pipeline.joblib was trained on")
    X_test, y_test = test[feature_cols], test[target_col].to_numpy()
    _log(f"test set: {len(test):,} rows, fraud rate {y_test.mean():.2%} | model: {type(clf).__name__}", t_start)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "n_test_rows": int(len(test)), "n_features": len(feature_cols),
        "test_fraud_rate": float(y_test.mean()),
        "test_time_range": [int(test[time_col].min()), int(test[time_col].max())],
        "seed": seed,
    }

    # 1. predictions
    proba = model.predict_proba(X_test)[:, 1]
    preds = pd.DataFrame({
        "TransactionID": test["TransactionID"].to_numpy(),
        "TransactionDT": test[time_col].to_numpy(),
        "isFraud": y_test,
        "fraud_proba": proba,
        "pred_label_at_0_5": (proba >= 0.5).astype(int),
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)
    _log("saved test_predictions.parquet", t_start)

    # 2. threshold search on the test set (descriptive: shows the trade-off; NOT used to choose the threshold)
    test_table = threshold_table(y_test, proba, THRESHOLDS)
    write_json({"meta": meta, "set": "test", "thresholds": test_table}, RESULTS_DIR / "threshold_search.json")
    _log("saved threshold_search.json", t_start)

    # 3. choose the threshold on VALIDATION data, then report it on test. picking it on the
    #    test set and scoring it there would be the same selection bias as tuning on test.
    #    validation = the last (most recent) time-ordered CV fold of the training data,
    #    predicted by a clone of the final pipeline refit on the rows before it.
    y_train = train[target_col].to_numpy()
    fit_idx, val_idx = make_cv_splits(y_train, train[time_col].to_numpy(), n_splits, MIN_POS)[-1]
    _log(f"refitting final pipeline on {len(fit_idx):,} earlier train rows to pick the threshold "
         f"on the last {len(val_idx):,}-row validation fold...", t_start)
    val_model = clone(model).fit(train.iloc[fit_idx][feature_cols], y_train[fit_idx])
    val_proba = val_model.predict_proba(train.iloc[val_idx][feature_cols])[:, 1]
    val_table = threshold_table(y_train[val_idx], val_proba, THRESHOLDS)
    chosen = pick_threshold(val_table, MIN_RECALL)
    test_would_pick = pick_threshold(test_table, MIN_RECALL)["threshold"]
    del train

    report = {
        "meta": meta,
        "threshold_rule": {"assumption": THRESHOLD_ASSUMPTION, "min_recall": MIN_RECALL,
                           "objective": "max precision subject to recall >= min_recall"},
        "threshold_selection": {
            "chosen_threshold": chosen["threshold"],
            "selected_on": "validation: last time-ordered CV fold of the training data (test set not used)",
            "validation_rows": int(len(val_idx)), "validation_fraud_rate": float(y_train[val_idx].mean()),
            "validation_metrics_at_threshold": {k: chosen[k] for k in ("precision", "recall", "f1", "n_flagged")},
            "threshold_test_set_would_pick": test_would_pick, #for transparency only
        },
        "test_metrics_at_chosen_threshold": binary_metrics(y_test, proba, chosen["threshold"]),
        "test_metrics_at_0_5": binary_metrics(y_test, proba, 0.5),
        "classification_report_at_chosen_threshold": classification_report(
            y_test, (proba >= chosen["threshold"]).astype(int), labels=[0, 1],
            target_names=["legit", "fraud"], output_dict=True, zero_division=0),
    }
    write_json(to_jsonable(report), RESULTS_DIR / "final_classification_report.json")
    _log(f"saved final_classification_report.json (threshold {chosen['threshold']} chosen on validation)", t_start)

    # 4. SHAP for a fixed, reproducible test sample
    sample_idx = np.sort(np.random.default_rng(seed).choice(len(test), size=min(SHAP_SAMPLE_N, len(test)), replace=False))
    X_sample = X_test.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_vals = np.asarray(_positive_class(explainer.shap_values(X_sample)), dtype=np.float32)
    base_value = float(np.ravel(_positive_class(explainer.expected_value))[-1])
    #sanity check: base + sum of contributions must reproduce the model's raw log-odds output
    raw = clf.predict(X_sample, raw_score=True)
    max_additivity_err = float(np.max(np.abs(base_value + shap_vals.sum(axis=1) - raw)))
    if max_additivity_err > 1e-3:
        raise ValueError(f"SHAP values don't add up to the model output (max error {max_additivity_err:.2e})")

    ids = test["TransactionID"].to_numpy()[sample_idx]
    pd.DataFrame(shap_vals, columns=feature_cols).assign(TransactionID=ids)[["TransactionID"] + feature_cols] \
        .to_parquet(RESULTS_DIR / "shap_values.parquet", index=False)
    pd.concat([preds.iloc[sample_idx].reset_index(drop=True)
                   .assign(pred_label_at_chosen=lambda d: (d["fraud_proba"] >= chosen["threshold"]).astype(int)),
               X_sample.reset_index(drop=True)], axis=1) \
        .to_parquet(RESULTS_DIR / "shap_sample.parquet", index=False)
    write_json({
        "meta": meta,
        "sample": {"n_rows": int(len(sample_idx)), "selection": f"uniform random over the test set, seed={seed}",
                   "n_fraud": int(y_test[sample_idx].sum())},
        "base_value": base_value,
        "units": "log-odds (LightGBM raw margin); fraud_proba = sigmoid(base_value + sum of a row's SHAP values)",
        "max_additivity_error": max_additivity_err,
        "files": {
            "shap_values.parquet": "TransactionID + one SHAP value column per feature",
            "shap_sample.parquet": "same rows, same order: TransactionID, TransactionDT, isFraud, fraud_proba, "
                                   "pred_label_at_0_5, pred_label_at_chosen + the raw feature values",
        },
        "chosen_threshold": chosen["threshold"],
    }, RESULTS_DIR / "shap_meta.json")
    _log(f"saved SHAP for {len(sample_idx)} test rows (additivity error {max_additivity_err:.1e})", t_start)

    # 5. feature importances: native (gain + split) and mean |SHAP|, two separately ranked lists
    booster = clf.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    native = ranked_importances(feature_cols, gain, "gain")
    split_by_name = dict(zip(feature_cols, split))
    for row in native:
        row["split"] = int(split_by_name[row["feature"]])
    write_json({
        "meta": meta,
        "lightgbm_native": {"ranked_by": "gain (total loss reduction from splits on the feature); "
                                         "split = number of times it was used", "features": native},
        "mean_abs_shap": {"ranked_by": f"mean |SHAP| over the {len(sample_idx)}-row test sample (log-odds)",
                          "features": ranked_importances(feature_cols, np.abs(shap_vals).mean(axis=0), "mean_abs_shap")},
    }, RESULTS_DIR / "feature_importances.json")
    _log("saved feature_importances.json", t_start)

    # 6. calibration
    write_json({
        "meta": meta,
        "brier_score": float(brier_score_loss(y_test, proba)),
        "mean_predicted_proba": float(proba.mean()), "observed_fraud_rate": float(y_test.mean()),
        "uniform_bins": calibration_bins(y_test, proba, 10, "uniform"),
        "quantile_bins": calibration_bins(y_test, proba, 10, "quantile"),
    }, RESULTS_DIR / "calibration.json")
    _log("saved calibration.json -- done", t_start)


if __name__ == "__main__":
    main()
