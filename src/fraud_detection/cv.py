import warnings

import numpy as np

#keep transactions in chronological order
#make sure validation sets contain enough fraud examples

class TimeAwareStratifiedSplit:
    """
    Expanding-window CV splitter for imbalanced, time-ordered data.

    Splits the (already time-ordered) training data into `n_splits + 1` sequential
    blocks. Fold k trains on everything up to block k and validates ONLY on block
    k+1 so the model never trains on data that is temporally after its validation
    set, unlike random StratifiedKFold.

    Because fraud is rare (~3.5%), a validation block can end up with very few or
    zero positive cases. If a block has fewer than `min_pos` fraud rows, it's
    merged forward into the next block until the minimum is met, so every fold
    stays statistically usable.
    """
    def __init__(self, n_splits: int = 5, min_pos: int = 30):
        self.n_splits = n_splits
        self.min_pos = min_pos #min nb of frauds per split

    def split(self, X, y, time_values): #time_values : transaction times
        order = np.argsort(np.asarray(time_values), kind="stable") #stable so ties (equal timestamps) keep a deterministic order across runs
        y_arr = np.asarray(y)
        n = len(order) #nb of transactions
        block_size = n // (self.n_splits + 1)

        blocks = [order[i * block_size:(i + 1) * block_size] for i in range(self.n_splits + 1)] #divide the dataset into equally sized time blocks
        leftover = order[(self.n_splits + 1) * block_size:]
        if len(leftover):      #extra rows that don't fit into the blocks, add them to the last block
            blocks[-1] = np.concatenate([blocks[-1], leftover])

        folds = []
        train_idx = list(blocks[0])
        i = 1
        while i <= self.n_splits:
            val_idx = list(blocks[i])
            while y_arr[val_idx].sum() < self.min_pos and i < self.n_splits: #merge the next block into the validation set if it has too few positive cases
                i += 1
                val_idx += list(blocks[i])
            folds.append((list(train_idx), val_idx))
            train_idx += val_idx #expanding window: add the current validation block to the training set for the next fold
            i += 1
        if len(folds) >= 2 and y_arr[folds[-1][1]].sum() < self.min_pos: #last validation block has too few positive cases, merge it into the previous fold's validation set
            prev_train, prev_val = folds[-2]
            _, last_val = folds[-1]
            folds[-2] = (prev_train, prev_val + last_val)
            folds.pop()
        elif len(folds) == 1 and y_arr[folds[-1][1]].sum() < self.min_pos:
            #n_splits=1 means there's only ever one fold, so there's no earlier
            #fold to backward-merge into -- warn instead of silently shipping
            #an underpowered validation set
            warnings.warn(
                f"TimeAwareStratifiedSplit: the only fold has "
                f"{int(y_arr[folds[-1][1]].sum())} positive cases, below "
                f"min_pos={self.min_pos}, and there is no other fold to merge "
                "into (n_splits=1). Returning it anyway.",
                stacklevel=2,
            )

        for train_idx, val_idx in folds:
            yield np.array(train_idx), np.array(val_idx) #return, Instead of returning all folds at once, it produces them one at a time