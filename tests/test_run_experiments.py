import json

import numpy as np
import pytest

from src.fraud_detection.modeling import METRICS
from src.fraud_detection.run_experiments import (
    format_cv_entry,
    format_tuning_results,
    metrics_from_json,
    write_json,
)

#JSON-writing logic only -- no training, these use hand-made run_cv-shaped dicts


@pytest.fixture
def fake_cv():
    #same shape run_cv returns: {metric: np.ndarray of per-fold scores}
    return {m: np.array([0.2, 0.4, 0.6], dtype=np.float32) for m in METRICS}


def test_format_cv_entry_has_mean_std_and_folds_per_metric(fake_cv):
    entry = format_cv_entry("lgbm", fake_cv, seconds=12.34)

    assert entry["label"] == "LightGBM"
    assert entry["cv_seconds"] == 12.3
    assert set(entry["metrics"]) == set(METRICS)
    for m in METRICS:
        assert entry["metrics"][m]["mean"] == pytest.approx(0.4)
        assert entry["metrics"][m]["std"] == pytest.approx(np.std([0.2, 0.4, 0.6]))
        assert entry["metrics"][m]["folds"] == pytest.approx([0.2, 0.4, 0.6])


def test_format_tuning_results_strips_prefix_and_converts_numpy(fake_cv):
    best_params = {"clf__n_estimators": np.int64(400), "clf__learning_rate": np.float64(0.03),
                   "clf__max_depth": -1}
    out = format_tuning_results("lgbm", fake_cv, fake_cv, best_params, np.float64(0.5),
                                search_seconds=60.0, meta={"sample_n": None})

    assert out["family"] == "lgbm"
    assert out["best_params"] == {"n_estimators": 400, "learning_rate": 0.03, "max_depth": -1}
    assert set(out["untuned"]) == set(out["tuned"]) == set(METRICS)
    json.dumps(out)  # no numpy types left anywhere


def test_write_json_round_trips_and_resume_restores_fold_arrays(tmp_path, fake_cv):
    path = tmp_path / "results" / "cv_results.json"
    write_json({"meta": {}, "models": {"rf": format_cv_entry("rf", fake_cv, 1.0)}}, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert not path.with_suffix(".json.tmp").exists()

    restored = metrics_from_json(loaded["models"]["rf"]["metrics"])
    for m in METRICS:
        np.testing.assert_allclose(restored[m], fake_cv[m], rtol=1e-6)
