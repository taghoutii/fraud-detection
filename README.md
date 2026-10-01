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

## Running the service

The FastAPI app in `app/` serves `models/final_pipeline.joblib` (tuned LightGBM) and flags
a transaction as fraud when its probability is >= the threshold chosen in Stage 6, which
is read from `reports/results/final_classification_report.json` at startup.

- `GET /health` returns status, feature count and the threshold in use.
- `POST /predict` takes one transaction and returns `{"fraud_probability", "is_fraud", "threshold_used"}`.

**Input** is the 426 *processed* features the model was trained on (see `dataset.py`), not raw
IEEE-CIS columns. Every field is required and must be a number, or `null` for "missing" (sent to the
model as -999, as in training). Missing, non-numeric or unknown fields return a 422 that lists each
failing field. The full schema is at `/docs`.

**Locally:**

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --port 8000
```

**With Docker:**

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

The image installs `requirements-serve.txt` (only what `app/` imports or needs to unpickle the model),
not the full `requirements.txt`. Training-only packages like shap, mlflow and xgboost aren't needed to
serve predictions and would add several hundred MB to the image.

**Example request.** This builds a payload from a real processed test transaction in the committed SHAP sample:

```bash
python -c "import json, pandas as pd; from app.artifacts import FEATURE_NAMES; r = pd.read_parquet('reports/results/shap_sample.parquet').iloc[0]; json.dump({k: None if pd.isna(r[k]) else float(r[k]) for k in FEATURE_NAMES}, open('payload.json', 'w'))"
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @payload.json
# {"fraud_probability":0.0390...,"is_fraud":false,"threshold_used":0.13}
```
