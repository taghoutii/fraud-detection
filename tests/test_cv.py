import numpy as np
from src.fraud_detection.cv import TimeAwareStratifiedSplit

def test_no_future_leakage_into_train():
    rng = np.random.default_rng(0)
    n = 1000
    time_values = np.arange(n)                    
    y = (rng.random(n) < 0.04).astype(int)        # ~4% positive rate

    splitter = TimeAwareStratifiedSplit(n_splits=4, min_pos=5)
    for train_idx, val_idx in splitter.split(np.zeros(n), y, time_values):
        assert time_values[train_idx].max() < time_values[val_idx].min() #training has to finish before validation begins

def test_validation_folds_meet_min_positive_count():
    rng = np.random.default_rng(1)
    n = 2000
    time_values = np.arange(n)
    y = (rng.random(n) < 0.02).astype(int)

    splitter = TimeAwareStratifiedSplit(n_splits=5, min_pos=10)
    for _, val_idx in splitter.split(np.zeros(n), y, time_values):
        assert y[val_idx].sum() >= 10