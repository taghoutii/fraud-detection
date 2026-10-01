import json

import numpy as np
import pytest

from src.fraud_detection.evaluate_final import (
    binary_metrics,
    calibration_bins,
    pick_threshold,
    ranked_importances,
    threshold_table,
)

#threshold/JSON logic only -- no model, no SHAP, hand-made labels and probabilities


@pytest.fixture
def toy():
    #4 frauds, 6 legit; probabilities chosen so each threshold's counts are easy to check by hand
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.2, 0.7, 0.3, 0.1, 0.1, 0.05, 0.0])
    return y, p


def test_threshold_table_counts_and_metrics(toy):
    y, p = toy
    rows = {r["threshold"]: r for r in threshold_table(y, p, [0.5, 0.25])}

    # >= 0.5 flags 0.9, 0.8 (fraud) and 0.7 (legit)
    r = rows[0.5]
    assert (r["tp"], r["fp"], r["fn"], r["tn"], r["n_flagged"]) == (2, 1, 2, 5, 3)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    # >= 0.25 adds 0.4 (fraud) and 0.3 (legit)
    assert (rows[0.25]["tp"], rows[0.25]["fp"]) == (3, 2)


def test_threshold_table_flagging_nothing_gives_zero_not_nan(toy):
    y, p = toy
    r = threshold_table(y, p, [0.99])[0]
    assert r["n_flagged"] == 0 and r["precision"] == 0.0 and r["f1"] == 0.0


def test_pick_threshold_maximizes_precision_subject_to_recall(toy):
    y, p = toy
    table = threshold_table(y, p, [0.15, 0.25, 0.5, 0.75])
    # recall >= 0.6 rules out 0.5 and 0.75 (recall 0.5 each). that leaves 0.15 (4 fraud + 2 legit
    # flagged: precision 4/6, recall 1.0) and 0.25 (3 + 2: precision 3/5, recall 0.75) -> 0.15 wins
    best = pick_threshold(table, min_recall=0.6)
    assert best["threshold"] == 0.15
    assert best["recall"] >= 0.6

    with pytest.raises(ValueError):
        pick_threshold(table, min_recall=1.01)


def test_pick_threshold_breaks_precision_ties_toward_higher_recall():
    table = [{"threshold": 0.3, "precision": 0.5, "recall": 0.9},
             {"threshold": 0.6, "precision": 0.5, "recall": 0.7}]
    assert pick_threshold(table, min_recall=0.6)["threshold"] == 0.3


def test_binary_metrics_confusion_matrix_matches_table(toy):
    y, p = toy
    m = binary_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 2, "tp": 2}
    assert m["n_flagged"] == 3 and m["flag_rate"] == pytest.approx(0.3)
    json.dumps(m)  # plain python types only


def test_calibration_bins_cover_every_row_and_compute_observed_rate(toy):
    y, p = toy
    for strategy in ("uniform", "quantile"):
        bins = calibration_bins(y, p, n_bins=5, strategy=strategy)
        assert sum(b["count"] for b in bins) == len(y), strategy
        assert bins[0]["bin_lower"] == 0.0 and bins[-1]["bin_upper"] == 1.0
    uniform = calibration_bins(y, p, n_bins=5, strategy="uniform")
    # [0.8, 1.0] holds 0.9 and 0.8, both fraud; the 0.0 row lands in the first bin
    assert uniform[-1]["count"] == 2 and uniform[-1]["observed_fraud_rate"] == 1.0
    assert uniform[0]["count"] == 4


def test_calibration_includes_probability_exactly_one():
    bins = calibration_bins([1], [1.0], n_bins=10)
    assert bins[-1]["count"] == 1


def test_ranked_importances_sorted_descending_with_ranks():
    out = ranked_importances(["a", "b", "c"], np.array([0.1, 3.0, 2.0]), "gain")
    assert [r["feature"] for r in out] == ["b", "c", "a"]
    assert [r["rank"] for r in out] == [1, 2, 3]
    json.dumps(out)
