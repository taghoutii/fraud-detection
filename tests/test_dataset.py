import numpy as np
import pandas as pd
import pytest

from src.fraud_detection.dataset import add_behavioral_features, apply_encoders, fit_encoders


def _train_test():
    #train is fully observed in time before test, like the real time-based split
    train = pd.DataFrame({
        "TransactionDT": [100, 200, 300, 400],
        "card1": [1, 1, 2, 2],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        "addr1": [10.0, 10.0, 20.0, np.nan],
        "card4": ["visa", "visa", "visa", "mastercard"],
        "sparse": [np.nan, np.nan, np.nan, np.nan],   # 100% missing in train
        "dense": [1.0, 2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "TransactionDT": [500, 600, 700, 800],
        "card1": [1, 2, 3, 3],
        "TransactionAmt": [50.0, 60.0, 70.0, 80.0],
        "addr1": [20.0, 20.0, 30.0, np.nan],
        "card4": ["mastercard", "mastercard", "mastercard", "amex"],
        "sparse": [1.0, 2.0, 3.0, 4.0],               # fully observed in test
        "dense": [np.nan, np.nan, np.nan, np.nan],    # fully missing in test
    })
    return train, test


def test_categorical_frequency_encoding_comes_from_train_only():
    train, test = _train_test()
    out = apply_encoders(test, fit_encoders(train))

    # mastercard is 1/4 of TRAIN. a map refit on test would give 3/4, one fit on
    # train+test would give 4/8 -- either would be the leak this guards against
    assert out["card4_freq_enc"].iloc[:3].tolist() == [0.25, 0.25, 0.25]
    # amex never appears in train: frequency 0, not test's own 1/4
    assert out["card4_freq_enc"].iloc[3] == 0.0
    assert "card4" not in out.columns


def test_merchant_frequency_map_comes_from_train_only():
    train, test = _train_test()
    out = apply_encoders(test, fit_encoders(train))

    # addr1=20 occurs once in TRAIN (twice in test -- must not be counted)
    assert out["addr1_merchant_freq"].iloc[0] == 1
    assert out["addr1_merchant_freq"].iloc[1] == 1
    # addr1=30 never seen in train -> count 0; missing addr1 -> -999, same as in train
    assert out["addr1_merchant_freq"].iloc[2] == 0
    assert out["addr1_merchant_freq"].iloc[3] == -999


def test_missing_value_column_drop_is_decided_on_train_only():
    train, test = _train_test()
    enc = fit_encoders(train)
    out_test = apply_encoders(test, enc)

    # "sparse" is >90% missing in train, so it's dropped from test even though test has it fully
    assert "sparse" not in out_test.columns
    # "dense" is complete in train, so test keeps it (filled with the -999 sentinel) even though
    # test alone would have dropped it
    assert out_test["dense"].tolist() == [-999] * 4


def test_train_and_test_get_identical_columns_in_identical_order():
    train, test = _train_test()
    enc = fit_encoders(train)
    assert apply_encoders(train, enc).columns.tolist() == apply_encoders(test, enc).columns.tolist()


def test_applying_encoders_to_train_matches_fitting_on_train():
    #fit/apply split must reproduce what the old one-shot train pipeline produced
    train, _ = _train_test()
    out = apply_encoders(train, fit_encoders(train))
    assert out["card4_freq_enc"].tolist() == [0.75, 0.75, 0.75, 0.25]
    assert out["addr1_merchant_freq"].tolist() == [2, 2, 1, -999]


def test_behavioral_features_of_earlier_rows_ignore_later_rows():
    #why behavioral features may be computed before the split: appending later (test-period)
    #rows must not change any earlier (train-period) row's values
    train, test = _train_test()
    alone = add_behavioral_features(train.copy()).sort_values("TransactionDT").reset_index(drop=True)
    full = add_behavioral_features(pd.concat([train, test], ignore_index=True))
    full = full.sort_values("TransactionDT").reset_index(drop=True)

    feature_cols = [c for c in alone.columns if c not in train.columns]
    pd.testing.assert_frame_equal(alone[feature_cols], full.loc[: len(train) - 1, feature_cols],
                                  check_dtype=False)


def test_test_rows_see_card_history_from_train_period():
    #card 1's first TEST transaction (t=500) follows its last TRAIN transaction (t=200);
    #computing features on test alone would wrongly cold-start it at -1
    train, test = _train_test()
    full = add_behavioral_features(pd.concat([train, test], ignore_index=True))
    row = full[full["TransactionDT"] == 500].iloc[0]
    assert row["card1_time_since_last"] == 300
    assert row["card1_amt_mean"] == pytest.approx(15.0) # mean of card 1's train amounts [10, 20]
