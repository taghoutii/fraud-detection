import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.fraud_detection.cv import TimeAwareStratifiedSplit
from src.fraud_detection.modeling import (
    METRICS,
    build_lgbm_pipeline,
    build_logreg_pipeline,
    build_logreg_smote_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
    compute_scale_pos_weight,
    run_cv,
)


@pytest.fixture
def toy_data():
    #~10% positives, small enough that every builder fits in well under a second
    X, y = make_classification(n_samples=600, n_features=8, n_informative=4,
                               weights=[0.9, 0.1], random_state=0)
    time_values = np.arange(len(y))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y, time_values


def _all_pipelines(y):
    pipes = {
        "logreg": build_logreg_pipeline(),
        "logreg_smote": build_logreg_smote_pipeline(),
        "rf": build_rf_pipeline(),
        "xgb": build_xgb_pipeline(scale_pos_weight=compute_scale_pos_weight(y)),
        "lgbm": build_lgbm_pipeline(),
    }
    #shrink the tree ensembles so the test suite stays fast
    pipes["rf"].set_params(clf__n_estimators=20)
    pipes["xgb"].set_params(clf__n_estimators=20)
    pipes["lgbm"].set_params(clf__n_estimators=20)
    return pipes


@pytest.mark.parametrize("name", ["logreg", "logreg_smote", "rf", "xgb", "lgbm"])
def test_builder_fits_and_predicts_proba(toy_data, name):
    X, y, _ = toy_data
    pipe = _all_pipelines(y)[name]
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_smote_pipeline_is_imblearn_others_are_sklearn(toy_data):
    _, y, _ = toy_data
    pipes = _all_pipelines(y)
    assert isinstance(pipes["logreg_smote"], ImbPipeline)
    for name in ["logreg", "rf", "xgb", "lgbm"]:
        #imblearn's Pipeline subclasses sklearn's, so check the exact type
        assert type(pipes[name]) is Pipeline, name


def test_compute_scale_pos_weight():
    assert compute_scale_pos_weight([0, 0, 0, 1]) == 3.0
    with pytest.raises(ValueError):
        compute_scale_pos_weight([0, 0, 0])


def test_run_cv_returns_one_score_per_fold_for_each_metric(toy_data):
    X, y, time_values = toy_data
    n_splits, min_pos = 3, 5
    expected_folds = len(list(
        TimeAwareStratifiedSplit(n_splits=n_splits, min_pos=min_pos).split(X, y, time_values)
    ))

    results = run_cv(build_logreg_pipeline(), X, y, time_values, n_splits=n_splits, min_pos=min_pos)

    assert set(results) == set(METRICS)
    for metric, scores in results.items():
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (expected_folds,), metric
        assert np.all((scores >= 0) & (scores <= 1)), metric


def test_run_cv_does_not_fit_the_pipeline_passed_in(toy_data):
    #each fold must fit its own clone, so the caller's pipeline stays untouched
    X, y, time_values = toy_data
    pipe = build_logreg_smote_pipeline()
    run_cv(pipe, X, y, time_values, n_splits=3, min_pos=5)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipe.named_steps["clf"])


def test_run_cv_handles_rows_not_sorted_by_time(toy_data):
    #the splitter returns positional indices into time-sorted order; shuffling
    #the rows (with their timestamps) must give the same scores
    X, y, time_values = toy_data
    perm = np.random.default_rng(0).permutation(len(y))
    kwargs = dict(n_splits=3, min_pos=5)

    sorted_res = run_cv(build_logreg_pipeline(), X, y, time_values, **kwargs)
    shuffled_res = run_cv(build_logreg_pipeline(), X.iloc[perm], y[perm], time_values[perm], **kwargs)

    for m in METRICS:
        np.testing.assert_allclose(sorted_res[m], shuffled_res[m], rtol=1e-6)
