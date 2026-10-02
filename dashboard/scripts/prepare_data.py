"""
One-time conversion of the Stage 5/6 result files into frontend-friendly JSON.

    .venv/Scripts/python.exe dashboard/scripts/prepare_data.py     (from the repo root)

Reads reports/results/ (never writes there) and writes dashboard/public/data/:
    results/*.json        Stage 5/6 JSON files, copied byte-for-byte
    overview.json         headline numbers for the Overview, read from those files' meta
    predictions.json      every test prediction (columnar): probability + true label
    transactions.json     the 2,000-row SHAP sample: ids, probability, label, amount, chunk
    shap_global.json      top-20 features by mean |SHAP|: every sample row's SHAP value + value percentile
    shap_local/NN.json    full per-row SHAP (all 426 features) + feature values, 100 rows per file
Pure reformatting: no metric is recomputed here except as a fidelity check against the source files.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "results"
OUT = ROOT / "dashboard" / "public" / "data"

COPY_AS_IS = ["cv_results.json", "tuning_results.json", "threshold_search.json", "calibration.json",
              "final_classification_report.json", "feature_importances.json", "shap_meta.json"]
MISSING_SENTINEL = -999  #handle_missing()'s fill value: shown as "missing" in the UI, not as a number
GLOBAL_TOP_N = 20
CHUNK_SIZE = 100
SHAP_DECIMALS = 6  #log-odds; 6dp keeps every waterfall within ~4e-6 of the saved probability (4dp drifted 2e-4)
PROBA_DECIMALS = 10  #fewer creates artificial ties that shift PR-AUC (6dp: 0.57833 vs saved 0.57842)


def dump(obj, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return path.stat().st_size / 1e6


def value_or_none(x):
    #feature value for display: None for missing (NaN or the -999 sentinel), else 6 significant digits
    if pd.isna(x) or x == MISSING_SENTINEL:
        return None
    return float(f"{x:.6g}")


def percentile_ranks(col: np.ndarray) -> list:
    #per-feature color scale for the beeswarm: rank among this feature's non-missing sample values (0-1)
    present = ~(np.isnan(col) | (col == MISSING_SENTINEL))
    out = np.full(len(col), np.nan)
    if present.sum() > 1:
        out[present] = pd.Series(col[present]).rank(pct=True).to_numpy()
    return [None if np.isnan(v) else round(float(v), 3) for v in out]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    sizes = {}

    # 1. JSON results, untouched
    for name in COPY_AS_IS:
        (OUT / "results").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / name, OUT / "results" / name)
        sizes[f"results/{name}"] = (OUT / "results" / name).stat().st_size / 1e6
    cv = json.loads((SRC / "cv_results.json").read_text(encoding="utf-8"))
    report = json.loads((SRC / "final_classification_report.json").read_text(encoding="utf-8"))
    shap_meta = json.loads((SRC / "shap_meta.json").read_text(encoding="utf-8"))
    importances = json.loads((SRC / "feature_importances.json").read_text(encoding="utf-8"))

    # 2. Overview headline numbers (all read from saved meta, nothing recomputed)
    test_meta = report["meta"]
    t0, t1 = test_meta["test_time_range"]
    n_train, n_test = cv["meta"]["n_rows"], test_meta["n_test_rows"]
    overview = {
        "model": "LightGBM",
        "test_pr_auc": report["test_metrics_at_chosen_threshold"]["pr_auc"],
        "test_roc_auc": report["test_metrics_at_chosen_threshold"]["roc_auc"],
        "chosen_threshold": report["threshold_selection"]["chosen_threshold"],
        "n_transactions": n_train + n_test, "n_train": n_train, "n_test": n_test,
        "train_fraud_rate": cv["meta"]["fraud_rate"], "test_fraud_rate": test_meta["test_fraud_rate"],
        "test_span_days": round((t1 - t0) / 86400, 1),
        "n_features": test_meta["n_features"],
    }
    sizes["overview.json"] = dump(overview, OUT / "overview.json")

    # 3. Every test prediction, columnar. Fidelity check: curves rebuilt from the rounded
    #    probabilities must reproduce Stage 6's saved PR-AUC / ROC-AUC
    preds = pd.read_parquet(SRC / "test_predictions.parquet")
    proba = preds["fraud_proba"].round(PROBA_DECIMALS).to_numpy()
    label = preds["isFraud"].astype(int).to_numpy()
    fidelity = {
        "pr_auc_saved": report["test_metrics_at_chosen_threshold"]["pr_auc"],
        "pr_auc_from_json": float(average_precision_score(label, proba)),
        "roc_auc_saved": report["test_metrics_at_chosen_threshold"]["roc_auc"],
        "roc_auc_from_json": float(roc_auc_score(label, proba)),
    }
    sizes["predictions.json"] = dump({"n": int(len(preds)), "proba": proba.tolist(), "label": label.tolist(),
                                      "fidelity_check": fidelity}, OUT / "predictions.json")

    # 4. SHAP sample: transaction index, global beeswarm, full per-row local detail
    shap_df = pd.read_parquet(SRC / "shap_values.parquet")
    sample = pd.read_parquet(SRC / "shap_sample.parquet")
    if not (shap_df["TransactionID"].to_numpy() == sample["TransactionID"].to_numpy()).all():
        raise ValueError("shap_values and shap_sample rows are not aligned")
    features = [c for c in shap_df.columns if c != "TransactionID"]
    shap = shap_df[features].to_numpy(dtype=np.float64)
    values = sample[features].to_numpy(dtype=np.float64)

    transactions = []
    for i, row in sample.iterrows():
        transactions.append({
            "id": int(row["TransactionID"]), "proba": round(float(row["fraud_proba"]), PROBA_DECIMALS),
            "is_fraud": int(row["isFraud"]), "flagged": int(row["pred_label_at_chosen"]),
            "amount": value_or_none(row["TransactionAmt"]), "chunk": i // CHUNK_SIZE,
        })
    sizes["transactions.json"] = dump({"base_value": shap_meta["base_value"], "units": shap_meta["units"],
                                       "chosen_threshold": shap_meta["chosen_threshold"],
                                       "transactions": transactions}, OUT / "transactions.json")

    top = [f["feature"] for f in importances["mean_abs_shap"]["features"][:GLOBAL_TOP_N]]
    sizes["shap_global.json"] = dump({
        "features": [{"feature": f, "mean_abs_shap": float(np.abs(shap[:, features.index(f)]).mean()),
                      "shap": np.round(shap[:, features.index(f)], SHAP_DECIMALS).tolist(),
                      "value_pct": percentile_ranks(values[:, features.index(f)])} for f in top],
        "n_rows": len(sample),
    }, OUT / "shap_global.json")

    rounded = np.round(shap, SHAP_DECIMALS)
    max_logit_err = 0.0
    for c in range(int(np.ceil(len(sample) / CHUNK_SIZE))):
        rows = range(c * CHUNK_SIZE, min((c + 1) * CHUNK_SIZE, len(sample)))
        chunk = {"features": features, "rows": {}}
        for i in rows:
            chunk["rows"][str(int(sample["TransactionID"].iat[i]))] = {
                "shap": rounded[i].tolist(), "values": [value_or_none(v) for v in values[i]]}
            #fidelity: base + rounded contributions must still give the saved probability
            logit = shap_meta["base_value"] + rounded[i].sum()
            max_logit_err = max(max_logit_err, abs(logit - np.log(sample["fraud_proba"].iat[i] / (1 - sample["fraud_proba"].iat[i]))))
        sizes[f"shap_local/{c:02d}.json"] = dump(chunk, OUT / "shap_local" / f"{c:02d}.json")

    print(f"wrote {len(sizes)} files to {OUT.relative_to(ROOT)}, {sum(sizes.values()):.2f} MB total")
    for name, mb in sizes.items():
        if not name.startswith("shap_local/") or name.endswith("00.json"):
            print(f"  {name:<44} {mb:6.3f} MB")
    print(f"  shap_local/00-{c:02d}.json  ({c + 1} files, {sum(v for k, v in sizes.items() if k.startswith('shap_local')):.2f} MB)")
    print("fidelity:", {k: round(v, 6) for k, v in fidelity.items()},
          f"| max |logit error| from rounding SHAP to {SHAP_DECIMALS}dp: {max_logit_err:.2e}")


if __name__ == "__main__":
    main()
