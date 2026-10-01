"""
loads the cached dataset, trains and cross-validates all 5 model types using cv.py splitter, 
runs hyperparameter tuning on the winner, and saves the results (metrics, best parameters, 
the final trained model) to disk as files and saves the final pipeline for deploment
"""
import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))
warnings.filterwarnings("ignore") #before the imports: mlflow 2.13 warns about pkg_resources on import

import joblib
import mlflow
import numpy as np
import yaml
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from src.fraud_detection.dataset import NON_FEATURE_COLS, load_or_build_train_set
from src.fraud_detection.modeling import (
    METRICS,
    build_lgbm_pipeline,
    build_logreg_pipeline,
    build_logreg_smote_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
    compute_scale_pos_weight,
    make_cv_splits,
    run_cv,
    summarize_cv,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "reports" / "results"
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"

MIN_POS = 30        
THRESHOLD = 0.5     

LABELS = {
    "logreg": "LogReg (class_weight)",
    "logreg_smote": "LogReg + SMOTE",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

PARAM_DISTRIBUTIONS = {
    "lgbm": {
        "clf__n_estimators": randint(200, 1000),
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__num_leaves": randint(31, 256),
        "clf__max_depth": [-1, 8, 12, 16],
        "clf__min_child_samples": randint(20, 200),
        "clf__subsample": uniform(0.6, 0.4),          #0.6 - 1.0
        "clf__colsample_bytree": uniform(0.4, 0.6),   #0.4 - 1.0
        "clf__reg_lambda": loguniform(1e-3, 10),
    },
    "xgb": {
        "clf__n_estimators": randint(200, 1000),
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__max_depth": randint(4, 12),
        "clf__min_child_weight": randint(1, 20),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.4, 0.6),
        "clf__gamma": loguniform(1e-3, 5),
    },
    "rf": {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": [None, 12, 20, 30],
        "clf__min_samples_leaf": randint(1, 20),
        "clf__max_features": ["sqrt", "log2", 0.2],
    },
    "logreg": {"clf__C": loguniform(1e-3, 10)},
    "logreg_smote": {"clf__C": loguniform(1e-3, 10)},
}


# ---- JSON formatting (pure, unit-tested) ----

def to_jsonable(value):
    #numpy scalars/arrays (e.g. from best_params_ or run_cv) -> plain python for json.dump
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def format_metrics(results: dict) -> dict:
    #run_cv output -> {metric: {"mean", "std", "folds"}}
    summary = summarize_cv(results)
    return {
        m: {"mean": summary[f"{m}_mean"], "std": summary[f"{m}_std"], "folds": to_jsonable(results[m])}
        for m in METRICS
    }


def format_cv_entry(name: str, results: dict, seconds: float) -> dict:
    return {"label": LABELS.get(name, name), "cv_seconds": round(float(seconds), 1), "metrics": format_metrics(results)}


def format_tuning_results(family: str, untuned: dict, tuned: dict, best_params: dict,
                          best_search_score: float, search_seconds: float, meta: dict) -> dict:
    return {
        "meta": meta,
        "family": family,
        "label": LABELS.get(family, family),
        "best_params": to_jsonable({k.replace("clf__", ""): v for k, v in best_params.items()}),
        "best_search_pr_auc": float(best_search_score),
        "search_seconds": round(float(search_seconds), 1),
        "untuned": format_metrics(untuned),
        "tuned": format_metrics(tuned),
    }


def write_json(data: dict, path: Path) -> None:
    #write to a temp file then rename, so stopping mid-write never leaves a truncated JSON behind
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def metrics_from_json(entry_metrics: dict) -> dict:
    #inverse of format_metrics, for --resume: {metric: np.ndarray of per-fold scores}
    return {m: np.array(v["folds"]) for m, v in entry_metrics.items()}


# ---- experiment ----

def _log(msg: str, t_start: float) -> None:
    print(f"[{time.time() - t_start:7.0f}s] {msg}", flush=True)


def _log_mlflow_run(run_name: str, params: dict, metrics: dict) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def main(sample_n: int | None = None, n_iter: int = 20, resume: bool = False) -> None:
    t_start = time.time()
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed, n_splits = cfg["seed"], cfg["cv"]["n_splits"]
    time_col, target_col = cfg["data"]["time_col"], cfg["data"]["target_col"]

    #dry runs go to their own experiment so they never sit next to full-data runs
    experiment = cfg["mlflow"]["experiment_name"] + ("-dryrun" if sample_n else "")
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)

    _log("loading processed train set", t_start)
    train = load_or_build_train_set(ROOT / cfg["data"]["processed_dir"] / "train.parquet",
                                    ROOT / cfg["data"]["raw_dir"], time_col, cfg["split"]["test_size"])
    if sample_n:
        train = train.sample(n=sample_n, random_state=seed)
    X = train.drop(columns=[target_col] + NON_FEATURE_COLS)
    y = train[target_col].to_numpy()
    time_values = train[time_col].to_numpy()
    del train

    cv_splits = make_cv_splits(y, time_values, n_splits, MIN_POS)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_n": sample_n,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "fraud_rate": float(y.mean()),
        "seed": seed, "n_splits": n_splits, "min_pos": MIN_POS, "threshold": THRESHOLD,
        "folds": [{"train_rows": len(tr), "val_rows": len(va), "val_frauds": int(y[va].sum())}
                  for tr, va in cv_splits],
    }
    _log(f"X={X.shape[0]:,}x{X.shape[1]}  fraud rate={y.mean():.2%}  folds={len(cv_splits)}"
         + (f"  [DRY RUN sample_n={sample_n}]" if sample_n else ""), t_start)

    spw = compute_scale_pos_weight(y)
    pipelines = {
        "logreg": build_logreg_pipeline(seed),
        "logreg_smote": build_logreg_smote_pipeline(seed),
        "rf": build_rf_pipeline(seed),
        "xgb": build_xgb_pipeline(spw, seed),
        "lgbm": build_lgbm_pipeline(seed),
    }

    cv_path = RESULTS_DIR / "cv_results.json"
    cv_json = {"meta": meta, "models": {}}
    if resume and cv_path.exists():
        previous = json.loads(cv_path.read_text(encoding="utf-8"))
        same_setup = all(previous["meta"].get(k) == meta[k]
                         for k in ("sample_n", "n_rows", "seed", "n_splits", "min_pos", "threshold"))
        if same_setup:
            cv_json["models"] = previous["models"]
        else:
            _log("--resume: saved cv_results.json was produced with different settings, starting over", t_start)

    base_params = {"n_splits": n_splits, "min_pos": MIN_POS, "threshold": THRESHOLD, "seed": seed,
                   "n_rows": len(y), "n_features": X.shape[1], "sample_n": sample_n}
    cv_results = {}
    for name, pipe in pipelines.items():
        if name in cv_json["models"]:
            cv_results[name] = metrics_from_json(cv_json["models"][name]["metrics"])
            _log(f"cv {name:<13} skipped (already in cv_results.json)", t_start)
            continue
        _log(f"cv {name:<13} running...", t_start)
        t0 = time.time()
        cv_results[name] = run_cv(pipe, X, y, time_values, n_splits=n_splits, min_pos=MIN_POS, threshold=THRESHOLD)
        seconds = time.time() - t0
        _log_mlflow_run(f"cv_{name}", {**base_params, "model": name},
                        {**summarize_cv(cv_results[name]), "cv_seconds": seconds})
        cv_json["models"][name] = format_cv_entry(name, cv_results[name], seconds)
        write_json(cv_json, cv_path) #saved after every model, so a stopped run keeps finished models for --resume
        _log(f"cv {name:<13} done in {seconds / 60:.1f} min  pr_auc={cv_results[name]['pr_auc'].mean():.4f}", t_start)

    best_family = max(cv_results, key=lambda n: cv_results[n]["pr_auc"].mean())
    _log(f"search {best_family}: RandomizedSearchCV n_iter={n_iter} x {len(cv_splits)} folds...", t_start)
    search = RandomizedSearchCV(
        pipelines[best_family],
        param_distributions=PARAM_DISTRIBUTIONS[best_family],
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv_splits,          
        refit=True,            #best config refit on the full training set -> the pipeline we save
        random_state=seed,
        n_jobs=1,              #the boosters already use every core; parallel candidates would oversubscribe
        verbose=2,             #one line per fit, so progress is visible during the long search
    )
    t0 = time.time()
    with mlflow.start_run(run_name=f"search_{best_family}"):
        search.fit(X, y)
        search_seconds = time.time() - t0
        best_params_logged = {k.replace("clf__", "best_"): v for k, v in search.best_params_.items()}
        mlflow.log_params({"model": best_family, "n_iter": n_iter, "n_splits": n_splits, "min_pos": MIN_POS,
                           "seed": seed, "scoring": "average_precision", "sample_n": sample_n,
                           **best_params_logged})
        mlflow.log_metrics({"pr_auc_mean": search.best_score_,
                            "pr_auc_std": search.cv_results_["std_test_score"][search.best_index_],
                            "search_seconds": search_seconds})
    _log(f"search done in {search_seconds / 60:.1f} min  best pr_auc={search.best_score_:.4f}", t_start)

    #best_score_ only covers PR-AUC; rerun the tuned config through run_cv for every metric on the same folds
    _log(f"cv {best_family}_tuned running...", t_start)
    tuned_results = run_cv(search.best_estimator_, X, y, time_values, n_splits=n_splits, min_pos=MIN_POS, threshold=THRESHOLD)
    _log_mlflow_run(f"cv_{best_family}_tuned", {**base_params, "model": f"{best_family}_tuned", **best_params_logged},
                    summarize_cv(tuned_results))

    write_json(format_tuning_results(best_family, cv_results[best_family], tuned_results, search.best_params_,
                                     search.best_score_, search_seconds, {**meta, "n_iter": n_iter}),
               RESULTS_DIR / "tuning_results.json")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    _log(f"saved {cv_path.relative_to(ROOT)}, {(RESULTS_DIR / 'tuning_results.json').relative_to(ROOT)}, "
         f"{MODEL_PATH.relative_to(ROOT)}", t_start)


def _env_int(name: str):
    value = os.environ.get(name)
    return int(value) if value else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: CV all candidate pipelines, tune the PR-AUC winner, save results.")
    parser.add_argument("--sample-n", type=int, default=_env_int("SAMPLE_N"),
                        help="random subsample size for a dry run (default: full training set; env SAMPLE_N)")
    parser.add_argument("--n-iter", type=int, default=_env_int("N_ITER") or 20,
                        help="RandomizedSearchCV candidates (default 20; env N_ITER)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse models already in reports/results/cv_results.json from a stopped run with the same settings")
    args = parser.parse_args()
    main(sample_n=args.sample_n, n_iter=args.n_iter, resume=args.resume)
