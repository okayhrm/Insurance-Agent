# models/schema.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from enum import Enum


# -------------------------------------------------
# Public request / response models (API contract)
# -------------------------------------------------

class RunRequest(BaseModel):
    # The organisers’ JSON uses the key "documents", but we prefer
    # the attribute name document_url inside our code.  The alias
    # parameter keeps both worlds happy.
    document_url: HttpUrl = Field(
        ...,
        alias="documents",
        description="Publicly accessible PDF URL (e.g., Azure Blob SAS URL)",
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="One or more natural-language questions",
    )

    class Config:
        validate_by_name = True   # lets us access .document_url


class RunResponse(BaseModel):
    answers: List[str] = Field(
        ...,
        description="Direct string answers in the same order as the questions",
    )


# -------------------------------------------------
# Optional internal models (not returned via API)
# -------------------------------------------------

class Decision(str, Enum):
    covered = "covered"
    not_covered = "not covered"
    waiting_period = "waiting period"
    unclear = "unclear"


class DetailedAnswer(BaseModel):
    """Richer structure we may store internally for logs / audits."""
    question: str
    answer: str
    decision: Decision = Decision.unclear
    justification: str
    clause_reference: str
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="0.0 – 1.0 confidence score"
    )
