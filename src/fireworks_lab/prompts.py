"""Prompt versions used in the controlled optimization experiment."""

from typing import Literal

from fireworks_lab.policy import POLICY_TEXT

PromptVersion = Literal["baseline", "optimized", "optimized_v2"]

BASELINE_SYSTEM = """
Classify the support ticket. Return only JSON with exactly four fields.

Allowed values:
- category: security_incident, billing_refund, bug_report, feature_request,
  account_access, or general_question
- priority: P0, P1, P2, or P3
- escalate: true or false
- reason_code: credential_theft, data_loss, duplicate_charge_over_500,
  ordinary_refund, service_blocking_bug, cosmetic_bug, product_enhancement,
  locked_account, or general_information

Do not add fields or prose outside the JSON object.
""".strip()

OPTIMIZED_SYSTEM = f"""
You are a deterministic enterprise support-routing engine.

{POLICY_TEXT}

Apply the first matching rule. Return only the four required JSON fields.
Do not invent labels. Do not add prose outside the JSON object.
""".strip()

OPTIMIZED_V2_SYSTEM = f"""
{OPTIMIZED_SYSTEM}

Boundary clarification for bug reports:
- If previously saved user work is deleted, corrupted, or disappears after
  synchronization, reconnection, or restart, use data_loss with priority P0,
  even if the triggering event also blocks a workflow.
- Use service_blocking_bug with priority P1 only when a core workflow cannot
  continue and the user's previously saved work remains intact.
""".strip()


def build_messages(ticket: str, version: PromptVersion) -> list[dict[str, str]]:
    if version == "baseline":
        system = BASELINE_SYSTEM
    elif version == "optimized":
        system = OPTIMIZED_SYSTEM
    elif version == "optimized_v2":
        system = OPTIMIZED_V2_SYSTEM
    else:
        raise ValueError(f"Unknown prompt version: {version}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": ticket},
    ]
