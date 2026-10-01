from pathlib import Path

import pandas as pd

from src.fraud_detection.data_loader import load_raw, reduce_memory
from src.fraud_detection.features import (
    add_amount_features,
    add_merchant_frequency,
    add_time_features,
    add_time_since_last_txn,
    add_velocity_features,
)
from src.fraud_detection.preprocessing import encode_categoricals, handle_missing, time_based_split

#chains the Stage 1-4 functions (unchanged) into the processed TRAIN set used for CV.
#the test split is deliberately not built here -- it needs train-fitted encodings (Stage 6)

#absolute time position: with a time-based split these would only teach the model
#to extrapolate a trend, so they're not features (TransactionDT is kept as the CV time axis)
NON_FEATURE_COLS = ["TransactionID", "TransactionDT", "txn_day"]


def build_train_set(raw_dir: str | Path, time_col: str = "TransactionDT", test_size: float = 0.2) -> pd.DataFrame:
    df = reduce_memory(load_raw(raw_dir))
    train, _ = time_based_split(df, time_col=time_col, test_size=test_size)
    del df
    train = train.copy() #own the slice so the in-place feature functions don't warn

    #behavioral features first, on raw values -- handle_missing's -999 sentinel would
    #otherwise be counted as a real addr1 "merchant"
    train = add_time_features(train, time_col)
    train = add_velocity_features(train, "card1", time_col)
    train = add_time_since_last_txn(train, "card1", time_col)
    train = add_amount_features(train, "card1", time_col)
    train, _ = add_merchant_frequency(train, "addr1")

    train = handle_missing(train) #drop decision made on train rows only
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    train = encode_categoricals(train, cat_cols)
    return train.sort_values(time_col).reset_index(drop=True)


def load_or_build_train_set(processed_path: str | Path, raw_dir: str | Path,
                            time_col: str = "TransactionDT", test_size: float = 0.2) -> pd.DataFrame:
    processed_path = Path(processed_path)
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    train = build_train_set(raw_dir, time_col, test_size)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(processed_path, index=False)
    return train
