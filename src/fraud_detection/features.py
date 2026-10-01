import pandas as pd
import numpy as np
#engineering behavioral features

def add_time_features(df: pd.DataFrame, time_col: str = "TransactionDT") -> pd.DataFrame:
    df["txn_day"] = (df[time_col] // (24 * 3600)) #get the day of the transaction
    df["txn_hour"] = (df[time_col] // 3600) % 24 #get the hour of the transaction
    return df

def add_velocity_features(df: pd.DataFrame, entity_col: str, time_col: str = "TransactionDT") -> pd.DataFrame:
    """
    Transaction velocity: count of transactions by the same entity (e.g. card1)
    in rolling time windows. Must be computed strictly on PAST data per row
    """
    df = df.sort_values(time_col).copy()
    df[f"{entity_col}_txn_count_1h"] = (
        df.groupby(entity_col)[time_col]
        .transform(lambda s: s.rolling(window=len(s), min_periods=1)
                   .apply(lambda x: ((x[-1] - x) <= 3600).sum() - 1, raw=True)) # -1 to exclude the current transaction
    )
    return df

def add_time_since_last_txn(df: pd.DataFrame, entity_col: str, time_col: str = "TransactionDT") -> pd.DataFrame:
    #how long has it been since this card's previous transaction?
    df = df.sort_values(time_col).copy()
    df[f"{entity_col}_time_since_last"] = (
        df.groupby(entity_col)[time_col].diff().fillna(-1)
    )
    return df

def add_merchant_frequency(df: pd.DataFrame, merchant_col: str = "addr1") -> tuple[pd.DataFrame, pd.Series]:
    """How often this merchant/address appears in the data. a static
    feature, safe to compute on train and MAP onto test (no leakage)."""
    freq_map = df[merchant_col].value_counts()
    df[f"{merchant_col}_merchant_freq"] = df[merchant_col].map(freq_map)
    return df, freq_map

def add_amount_features(df: pd.DataFrame, entity_col: str = "card1", time_col: str = "TransactionDT") -> pd.DataFrame:
    """
    Is this purchase unusual for this card? Z-score of this transaction's amount
    vs the entity's historical mean/std, computed strictly from that entity's
    PAST transactions: shift(1) before expanding() excludes the current row
    from its own baseline, same idea as add_time_since_last_txn's diff().
    Cold start (fewer than 2 prior transactions) leaves mean/std undefined by
    pandas (NaN); we set those to 0 and fall back to the raw amount difference
    for the z-score (std replaced with 1) rather than leaking future data in.
    """
    df = df.sort_values(time_col).copy()
    grp = df.groupby(entity_col)["TransactionAmt"]
    expanding_mean = grp.transform(lambda s: s.shift(1).expanding().mean())
    expanding_std = grp.transform(lambda s: s.shift(1).expanding().std())

    df[f"{entity_col}_amt_mean"] = expanding_mean.fillna(0)
    df[f"{entity_col}_amt_std"] = expanding_std.fillna(0)
    df[f"{entity_col}_amt_zscore"] = (
        (df["TransactionAmt"] - df[f"{entity_col}_amt_mean"])
        / df[f"{entity_col}_amt_std"].replace(0, 1)
    )
    return df