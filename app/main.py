import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.artifacts import FEATURE_NAMES, PIPELINE, THRESHOLD
from app.schema import FraudPrediction, TransactionInput

MISSING_SENTINEL = -999 #what handle_missing() filled numeric NaNs with during training

app = FastAPI(
    title="IEEE-CIS fraud detection API",
    description="Serves the Stage 5 tuned LightGBM pipeline at the Stage 6 decision threshold.",
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    #FastAPI's default 422 echoes the whole request back (426 fields); report only what failed and why
    errors = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"] if part != "body")
        if err["type"] == "json_invalid" or not field: #malformed JSON: loc is a character position, not a field
            field = "(request body)"
        errors.append({"field": field, "type": err["type"], "message": err["msg"]})
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict:
    #model and threshold load at import, so if this answers they're both in memory
    return {"status": "ok", "n_features": len(FEATURE_NAMES), "threshold": THRESHOLD}


@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: TransactionInput) -> FraudPrediction:
    row = pd.DataFrame([transaction.model_dump()], columns=FEATURE_NAMES).astype(float)
    row = row.fillna(MISSING_SENTINEL)
    probability = float(PIPELINE.predict_proba(row)[0, 1])
    return FraudPrediction(
        fraud_probability=probability,
        is_fraud=probability >= THRESHOLD, #same >= rule Stage 6 used to flag transactions
        threshold_used=THRESHOLD,
    )
