from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.fraud_detection.data_loader import load_raw, reduce_memory
from src.fraud_detection.features import (
    add_amount_features,
    add_time_features,
    add_time_since_last_txn,
    add_velocity_features,
)
from src.fraud_detection.preprocessing import handle_missing, time_based_split

#builds the processed train AND test sets from the Stage 1-4 functions (unchanged).
#two kinds of transformation, handled differently on purpose:
#  1. behavioral features (velocity, time since last txn, amount z-score) only look at
#     each card's EARLIER transactions and never at labels, so they're computed once on
#     the full time-ordered data before splitting: a test row then sees its card's history
#     from the train period, exactly as it would in production. a train row's values can't
#     change because later rows exist (tested in test_dataset.py).
#  2. anything FITTED on data (which columns to drop for missingness, the addr1 merchant
#     frequency map, categorical frequency maps) is fit on the train split only via
#     fit_encoders(), then applied unchanged to both splits via apply_encoders().

#absolute time position: with a time-based split these would only teach the model
#to extrapolate a trend, so they're not features (TransactionDT is kept as the CV time axis)
NON_FEATURE_COLS = ["TransactionID", "TransactionDT", "txn_day"]
MERCHANT_COL = "addr1"


@dataclass
class FittedEncoders:
    kept_columns: list            #columns surviving handle_missing's drop rule, decided on train
    merchant_freq_map: pd.Series  #addr1 value -> transaction count in train
    cat_freq_maps: dict           #categorical column -> normalized value frequency in train


def add_behavioral_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    df = add_time_features(df, time_col)
    df = add_velocity_features(df, "card1", time_col)
    df = add_time_since_last_txn(df, "card1", time_col)
    df = add_amount_features(df, "card1", time_col)
    return df


def fit_encoders(train: pd.DataFrame, num_thresh: float = 0.9) -> FittedEncoders:
    #every statistic here is computed from `train` only
    merchant_freq_map = train[MERCHANT_COL].value_counts()
    with_merchant = train.assign(**{f"{MERCHANT_COL}_merchant_freq": train[MERCHANT_COL].map(merchant_freq_map)})
    filled = handle_missing(with_merchant, num_thresh) #drop decision + fills, on train rows only
    cat_cols = filled.select_dtypes(include="object").columns
    return FittedEncoders(
        kept_columns=filled.columns.tolist(),
        merchant_freq_map=merchant_freq_map,
        cat_freq_maps={col: filled[col].value_counts(normalize=True) for col in cat_cols},
    )


def apply_encoders(df: pd.DataFrame, enc: FittedEncoders) -> pd.DataFrame:
    #applies train-fitted maps via .map(); nothing in here looks at df's own value distribution
    df = df.copy()
    merchant_freq = df[MERCHANT_COL].map(enc.merchant_freq_map)
    merchant_freq[df[MERCHANT_COL].notna() & merchant_freq.isna()] = 0 #addr1 never seen in train: its train count really is 0
    df[f"{MERCHANT_COL}_merchant_freq"] = merchant_freq #missing addr1 stays NaN -> -999 below, same as in train

    df = df[enc.kept_columns] #same columns as train, even if test's own missingness would say otherwise
    df = handle_missing(df, num_thresh=1.0) #fills only: no column can be >100% missing, so none is dropped here

    for col, freq in enc.cat_freq_maps.items():
        df[col + "_freq_enc"] = df[col].map(freq).astype(float).fillna(0.0) #category never seen in train -> frequency 0
    return df.drop(columns=list(enc.cat_freq_maps))


def build_train_test_sets(raw_dir: str | Path, time_col: str = "TransactionDT",
                          test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = reduce_memory(load_raw(raw_dir))
    df = add_behavioral_features(df, time_col) #backward-looking only, see module comment
    train, test = time_based_split(df, time_col=time_col, test_size=test_size)
    del df

    enc = fit_encoders(train)
    train = apply_encoders(train, enc)
    test = apply_encoders(test, enc)
    return (train.sort_values(time_col, kind="stable").reset_index(drop=True),
            test.sort_values(time_col, kind="stable").reset_index(drop=True))


def load_or_build_split(processed_dir: str | Path, raw_dir: str | Path, time_col: str = "TransactionDT",
                        test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    #train.parquet and test.parquet are always built and saved together, so they share one set of encoders
    processed_dir = Path(processed_dir)
    train_path, test_path = processed_dir / "train.parquet", processed_dir / "test.parquet"
    if train_path.exists() and test_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(test_path)
    train, test = build_train_test_sets(raw_dir, time_col, test_size)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train, test


def load_or_build_train_set(processed_path: str | Path, raw_dir: str | Path,
                            time_col: str = "TransactionDT", test_size: float = 0.2) -> pd.DataFrame:
    #kept for run_experiments.py (Stage 5), which must only ever see the train split
    processed_path = Path(processed_path)
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    train, _ = load_or_build_split(processed_path.parent, raw_dir, time_col, test_size)
    return train
