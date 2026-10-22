from fireworks_lab.prompts import build_messages


def test_baseline_prompt_contains_output_contract_without_policy() -> None:
    messages = build_messages("A support ticket.", "baseline")
    system_text = messages[0]["content"]

    allowed_values = (
        "security_incident",
        "billing_refund",
        "bug_report",
        "feature_request",
        "account_access",
        "general_question",
        "P0",
        "P1",
        "P2",
        "P3",
        "credential_theft",
        "data_loss",
        "duplicate_charge_over_500",
        "ordinary_refund",
        "service_blocking_bug",
        "cosmetic_bug",
        "product_enhancement",
        "locked_account",
        "general_information",
    )

    for value in allowed_values:
        assert value in system_text

    assert "first matching rule" not in system_text.lower()
    assert "where each charge is over $500" not in system_text.lower()


def test_optimized_prompt_contains_decision_rules() -> None:
    messages = build_messages("Someone stole my password and logged in.", "optimized")

    system_text = messages[0]["content"]
    assert "credential theft" in system_text.lower()
    assert "P0" in system_text
    assert "duplicate charge" in system_text.lower()


def test_optimized_v2_clarifies_data_loss_boundary_without_eval_leakage() -> None:
    messages = build_messages("A support ticket.", "optimized_v2")
    system_text = messages[0]["content"].lower()

    assert "previously saved" in system_text
    assert "reconnection" in system_text
    assert "remains intact" in system_text

    assert (
        "edits keep being erased whenever the desktop client reconnects"
        not in system_text
    )
