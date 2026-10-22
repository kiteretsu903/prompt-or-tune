import pytest
from pydantic import ValidationError

from fireworks_lab.schemas import Prediction


def test_prediction_accepts_a_valid_decision() -> None:
    prediction = Prediction(
        category="billing_refund",
        priority="P1",
        escalate=True,
        reason_code="duplicate_charge_over_500",
    )

    assert prediction.priority == "P1"


def test_prediction_rejects_an_unknown_category() -> None:
    with pytest.raises(ValidationError):
        Prediction(
            category="mystery",
            priority="P1",
            escalate=True,
            reason_code="duplicate_charge_over_500",
        )
