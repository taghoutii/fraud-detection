# Fraud Detection Pipeline

An end-to-end fraud detection system for card transactions, built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset: 590,540
transactions, 3.5% fraud. It covers the full path from raw CSVs to a served model:
- **A leakage-aware data pipeline.** The train/test split is by time, not at random. Encoders are
  fitted on the training period only. Behavioral features (transaction velocity, time since the
  card's last transaction, how unusual the amount is for the card) look only backward in time.
- **A custom cross-validation splitter** that keeps every fold in time order while guaranteeing
  each validation fold has enough fraud cases to score.
- **Model selection and tuning** across five candidates, tracked in MLflow.
- **A one-time evaluation on a held-out test set**, with SHAP explanations.
- **A FastAPI service in Docker, and a React dashboard** that reports every result.

## Results

All numbers are from the held-out test set: the most recent 20% of transactions (118,108 rows,
about 42 days). It was used once, after every modeling decision had been made.

| | Test set |
|---|---|
| **PR-AUC** (average precision) | **0.578**, about 17× the 0.034 no-skill baseline |
| ROC-AUC | 0.911 |
| At the production threshold (0.13) | catches 61% of fraud; 45% of flagged transactions are fraud; 4.7% of traffic flagged |

The model is a tuned LightGBM. In time-ordered cross-validation it beat XGBoost, random forest
and two logistic-regression baselines on PR-AUC (0.584 vs 0.555 for the runner-up), and tuning
raised it to 0.621. PR-AUC is the headline metric because at a 3.5% fraud rate, ROC-AUC and
accuracy look good even for weak models.

For context, the winning solutions to the original Kaggle competition reached about 0.95
ROC-AUC. They used extensive feature engineering, most importantly reconstructing card-holder
identities across transactions, and were scored on a different test set. This project
prioritizes a sound, leakage-free evaluation and a deployable system over leaderboard feature
mining. The 0.911 here is an honest estimate of performance on future transactions.

## Key findings

**Fraud patterns drift over time.** The threshold was chosen on the most recent validation
period and then applied unchanged to the later test period. Recall held (60% → 61%), but
precision dropped from 67% to 45%: the model still finds the fraud, yet more of what it flags
is legitimate. In production, alert precision is the metric to watch for retraining.

**Scores rank well but overstate the odds above about 0.2.** Transactions scored around 0.85
turn out to be fraud only 49% of the time. Ranking quality is strong (ROC-AUC 0.911), so
threshold decisions are sound. Any use of the score as a literal probability, such as
expected-loss estimates, needs a calibration step first.

## Running it

**Environment** (Python 3.12, managed with [uv](https://github.com/astral-sh/uv)):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-dev.txt
pytest                                   # 46 tests, no data needed
```

**Pipeline.** Retraining needs the Kaggle data in `data/raw/`. The saved model and results are
already committed, so this step is optional.

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw && unzip data/raw/ieee-fraud-detection.zip -d data/raw
python -m src.fraud_detection.run_experiments       # builds processed data, CV, tuning, saves the model
python -m src.fraud_detection.evaluate_final        # one-time test-set evaluation + SHAP
```

**API.** It takes the 426 processed features of one transaction and returns
`{fraud_probability, is_fraud, threshold_used}`. Schema at `/docs`.

```bash
uvicorn app.main:app --port 8000                     # locally (needs requirements-serve.txt)
docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
```

**Dashboard.** It runs from committed data, with no Python needed:

```bash
cd dashboard && npm install && npm run dev           # http://localhost:5173
```

## Repository layout

```
src/fraud_detection/    data loading, features, preprocessing, CV splitter, models,
                        run_experiments.py (training) and evaluate_final.py (test + SHAP)
app/                    FastAPI service (model and threshold loaded once at startup)
dashboard/              React + Vite dashboard and the script that prepares its data
models/                 final_pipeline.joblib, the tuned model the API serves
reports/results/        every saved result: CV, tuning, predictions, thresholds, SHAP, calibration
tests/                  unit tests, including leakage guards for features and encoders
notebooks/              exploratory analysis and the time-vs-random split comparison
config.yaml             seed, split size, CV folds, paths, MLflow experiment
```

## What I'd do next

- **Serve raw transactions, not processed features.** Persist the train-fitted encoders and add
  a per-card history store, so the API can compute velocity and amount features itself.
- **Calibrate the scores** (isotonic or Platt scaling, fitted on validation data), so they can
  be read as probabilities.
- **Monitor drift and retrain on a schedule.** Track alert precision on newly labeled
  transactions, given the drop seen on the test set.
- **Close part of the gap to the Kaggle leaders** with card-holder identity features and
  aggregations over them.
- **Set the threshold on dollar cost rather than counts**, weighting missed fraud by transaction
  amount.
