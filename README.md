# fraud-detection-pipeline

Production-style fraud detection pipeline built on the IEEE-CIS Fraud Detection dataset (Kaggle).

## Milestones

- **v0.1-data-pipeline** — covers Stages 1, 2, 4, and 4b: the raw data loader
  and memory-reduction utility, missing-value handling and frequency encoding
  for categoricals, the time-based train/test split, and the
  `TimeAwareStratifiedSplit` expanding-window CV splitter. The name reflects
  when the tag was cut, not just the data-loading/preprocessing stages — by
  the time it was tagged, the time-based split and CV splitter work had
  already landed on the same commit.
