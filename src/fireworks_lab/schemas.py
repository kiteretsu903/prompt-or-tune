"""Validated data contracts shared by inference, evaluation, and training."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

Category = Literal[
    "security_incident",
    "billing_refund",
    "bug_report",
    "feature_request",
    "account_access",
    "general_question",
]
Priority = Literal["P0", "P1", "P2", "P3"]
ReasonCode = Literal[
    "credential_theft",
    "data_loss",
    "duplicate_charge_over_500",
    "ordinary_refund",
    "service_blocking_bug",
    "cosmetic_bug",
    "product_enhancement",
    "locked_account",
    "general_information",
]


class Prediction(BaseModel):
    """The exact structured decision the model must produce."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    escalate: bool
    reason_code: ReasonCode
