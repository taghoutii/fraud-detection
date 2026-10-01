import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.features import add_amount_features, add_time_since_last_txn

def test_time_since_last_txn_no_leakage():
    df = pd.DataFrame({
        "card1": [1, 1, 1],
        "TransactionDT": [100, 200, 500],
    })
    out = add_time_since_last_txn(df, "card1")
    assert out.iloc[0][f"card1_time_since_last"] == -1   # first txn has no history
    assert out.iloc[1][f"card1_time_since_last"] == 100

def test_amount_features_no_leakage():
    df = pd.DataFrame({
        "card1": [1, 1, 1],
        "TransactionDT": [100, 200, 300],
        "TransactionAmt": [10.0, 20.0, 100.0],
    })
    out = add_amount_features(df, entity_col="card1")

    # first txn: no history at all -> mean/std default to 0 (cold start)
    assert out.iloc[0]["card1_amt_mean"] == 0
    assert out.iloc[0]["card1_amt_std"] == 0

    # second txn: history is only the first txn's amount (10.0) -- a single
    # point has no defined std, so std defaults to 0 too (cold start)
    assert out.iloc[1]["card1_amt_mean"] == 10.0
    assert out.iloc[1]["card1_amt_std"] == 0

    # third txn: history is [10.0, 20.0] only -- its OWN amount (100.0) must
    # never contribute to its own baseline, which is exactly the leak this
    # test guards against
    assert out.iloc[2]["card1_amt_mean"] == 15.0
    assert out.iloc[2]["card1_amt_std"] == pytest.approx(np.std([10.0, 20.0], ddof=1))
    expected_z = (100.0 - 15.0) / np.std([10.0, 20.0], ddof=1)
    assert out.iloc[2]["card1_amt_zscore"] == pytest.approx(expected_z)

def test_tied_timestamps_keep_input_order_regardless_of_later_rows():
    #same card, same second: which tx counts as "earlier" must not depend on what else is
    #in the frame (unstable sorts reshuffled ties once test-period rows were appended)
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "card1": rng.integers(0, 5, n),
        "TransactionDT": rng.integers(0, 40, n),   # heavy timestamp ties
        "TransactionAmt": rng.uniform(1, 100, n).round(2),
    })
    later = df.assign(TransactionID=df["TransactionID"] + n, TransactionDT=df["TransactionDT"] + 1000)

    for fn in (add_time_since_last_txn, add_amount_features):
        alone = fn(df.copy(), "card1").set_index("TransactionID").sort_index()
        full = fn(pd.concat([df, later], ignore_index=True), "card1").set_index("TransactionID").sort_index()
        pd.testing.assert_frame_equal(alone, full.loc[alone.index])
