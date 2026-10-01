from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.artifacts import FEATURE_NAMES

#One field per feature the trained pipeline expects, i.e. the PROCESSED columns from
#dataset.py (behavioral features, -999 missing sentinel, *_freq_enc encodings), not raw
#IEEE-CIS columns. Built from the pipeline's own feature list so the schema can't drift
#from the model.
#  - every field is required: omitting one is a 422
#  - null is allowed and means "missing"; it's sent to the model as -999, the same
#    sentinel handle_missing() used in training
#  - strict numbers: strings like "1.5", booleans, NaN/inf are rejected (422)
#  - unknown fields are rejected (422), so a typo'd name can't silently become a missing value
FeatureValue = Annotated[Optional[float], Field(strict=True, allow_inf_nan=False)]

TransactionInput = create_model(
    "TransactionInput",
    __config__=ConfigDict(extra="forbid"),
    **{name: (FeatureValue, ...) for name in FEATURE_NAMES},
)


class FraudPrediction(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0)
    is_fraud: bool = Field(description="fraud_probability >= threshold_used")
    threshold_used: float = Field(description="Stage 6 threshold from final_classification_report.json")
