import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.fraud_detection.cv import TimeAwareStratifiedSplit

#every builder names its estimator step "clf", so search spaces are always "clf__<param>"
METRICS = ("precision", "recall", "f1", "roc_auc", "pr_auc")


def build_logreg_pipeline(seed: int = 42) -> Pipeline:
    #baseline: imbalance handled by reweighting the loss, no resampling
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def build_logreg_smote_pipeline(seed: int = 42) -> ImbPipeline:
    """
    Comparison experiment against build_logreg_pipeline, NOT something we intend
    to adopt by default: it exists to check whether synthetic oversampling beats
    plain class_weight reweighting on this data.

    Must be imblearn's Pipeline: sklearn's Pipeline doesn't support samplers
    (SMOTE has fit_resample, not transform). imblearn's applies the sampler during
    fit only and skips it at predict time, so validation rows are never resampled.
    SMOTE runs after scaling so its k-NN distances aren't dominated by
    large-range columns.
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)), #no class_weight: SMOTE already balances the classes
    ])


def build_rf_pipeline(seed: int = 42) -> Pipeline:
    #trees are scale-invariant, so no scaler
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5, #caps tree size; fully grown trees on ~470k rows get very large in memory
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def compute_scale_pos_weight(y) -> float:
    #XGBoost's imbalance knob: neg_count / pos_count of the data it'll be trained on
    y = np.asarray(y)
    pos = int((y == 1).sum())
    if pos == 0:
        raise ValueError("compute_scale_pos_weight: y has no positive cases")
    return float((y == 0).sum() / pos)


def build_xgb_pipeline(scale_pos_weight: float, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def build_lgbm_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("clf", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1, #lightgbm ignores subsample unless bagging frequency is set
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )),
    ])


def make_cv_splits(y, time_values, n_splits: int = 5, min_pos: int = 30) -> list:
    #materialized folds, e.g. to pass as RandomizedSearchCV(cv=...) so tuning uses the exact same folds as run_cv
    splitter = TimeAwareStratifiedSplit(n_splits=n_splits, min_pos=min_pos)
    return list(splitter.split(np.zeros(len(y)), y, time_values))


def _take(data, idx):
    #positional row selection for both DataFrames/Series and numpy arrays
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def run_cv(pipeline, X, y, time_values, n_splits: int = 5, min_pos: int = 30, threshold: float = 0.5) -> dict:
    """
    Expanding-window CV with TimeAwareStratifiedSplit. Returns
    {metric: np.ndarray of per-fold scores} for every name in METRICS, all
    computed for the fraud class (label 1). precision/recall/f1 use
    `threshold` on predict_proba; roc_auc/pr_auc are threshold-free.
    """
    scores = {m: [] for m in METRICS}
    for train_idx, val_idx in make_cv_splits(y, time_values, n_splits, min_pos):
        model = clone(pipeline) #fresh unfitted copy per fold, so scaler/SMOTE only ever see this fold's training rows
        model.fit(_take(X, train_idx), _take(y, train_idx))

        y_val = np.asarray(_take(y, val_idx))
        proba = model.predict_proba(_take(X, val_idx))[:, 1]
        pred = (proba >= threshold).astype(int)

        scores["precision"].append(precision_score(y_val, pred, pos_label=1, zero_division=0))
        scores["recall"].append(recall_score(y_val, pred, pos_label=1, zero_division=0))
        scores["f1"].append(f1_score(y_val, pred, pos_label=1, zero_division=0))
        scores["roc_auc"].append(roc_auc_score(y_val, proba))
        scores["pr_auc"].append(average_precision_score(y_val, proba, pos_label=1))
    return {m: np.array(v) for m, v in scores.items()}


def summarize_cv(results: dict) -> dict:
    #{"pr_auc_mean": ..., "pr_auc_std": ..., ...} -- flat, so it can go straight into mlflow.log_metrics
    summary = {}
    for m, v in results.items():
        summary[f"{m}_mean"] = float(np.mean(v))
        summary[f"{m}_std"] = float(np.std(v))
    return summary
