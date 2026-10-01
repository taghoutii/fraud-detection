import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.artifacts import FEATURE_NAMES, REPORT_PATH
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def stage6_rows():
    #real processed test transactions + Stage 6's saved probabilities for them (committed file)
    return pd.read_parquet("reports/results/shap_sample.parquet").head(20)


def _payload(row) -> dict:
    return {name: (None if pd.isna(row[name]) else float(row[name])) for name in FEATURE_NAMES}


@pytest.fixture
def valid_payload(stage6_rows):
    return _payload(stage6_rows.iloc[0])


def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_well_formed_prediction(valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold_used"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold_used"])


def test_threshold_used_comes_from_stage6_report(valid_payload):
    #read the report independently, the same file the app loads -- no hardcoded value in either place
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    expected = report["threshold_selection"]["chosen_threshold"]
    assert client.post("/predict", json=valid_payload).json()["threshold_used"] == expected
    assert client.get("/health").json()["threshold"] == expected


def test_predict_reproduces_stage6_probabilities(stage6_rows):
    #serving must give the same answer the model gave during evaluation
    for _, row in stage6_rows.iterrows():
        body = client.post("/predict", json=_payload(row)).json()
        assert body["fraud_probability"] == pytest.approx(row["fraud_proba"], abs=1e-9)
        assert body["is_fraud"] == bool(row["pred_label_at_chosen"])


def test_null_means_missing_and_is_accepted(valid_payload):
    assert client.post("/predict", json={**valid_payload, "dist1": None}).status_code == 200


@pytest.mark.parametrize("change, field, error_type", [
    ({"TransactionAmt": "DROP"}, "TransactionAmt", "missing"),
    ({"TransactionAmt": "abc"}, "TransactionAmt", "float_type"),
    ({"TransactionAmt": "12.5"}, "TransactionAmt", "float_type"),   # numeric strings aren't coerced
    ({"TransactionAmt": True}, "TransactionAmt", "float_type"),
    ({"TransactionAmount": 1.0}, "TransactionAmount", "extra_forbidden"),  # typo'd name
])
def test_predict_rejects_invalid_input_with_422(valid_payload, change, field, error_type):
    payload = dict(valid_payload)
    for name, value in change.items():
        if value == "DROP":
            payload.pop(name)
        else:
            payload[name] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    errors = r.json()["detail"]
    assert {"field": field, "type": error_type} in [{"field": e["field"], "type": e["type"]} for e in errors]
    assert all("input" not in e for e in errors)  # the request body is not echoed back


def test_predict_rejects_empty_and_malformed_bodies():
    assert client.post("/predict", json={}).status_code == 422
    r = client.post("/predict", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "(request body)"
