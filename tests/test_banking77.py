import json

import pytest
from pydantic import ValidationError

from fireworks_lab.banking77 import (
    BankingPrediction,
    assess_banking_improvement,
    build_banking_messages,
    build_sft_records,
    evaluate_banking_rows,
    prepare_banking77_splits,
)


def _row(text: str, category: str) -> dict[str, str]:
    return {"text": text, "category": category}


def test_prepare_splits_is_balanced_deterministic_and_leakage_safe() -> None:
    train_rows = [
        _row("activate one", "activate_my_card"),
        _row("activate two", "activate_my_card"),
        _row("activate three", "activate_my_card"),
        _row("activate four", "activate_my_card"),
        _row("  shared   query ", "activate_my_card"),
        _row("cash one", "cash_withdrawal_charge"),
        _row("cash two", "cash_withdrawal_charge"),
        _row("cash three", "cash_withdrawal_charge"),
        _row("cash four", "cash_withdrawal_charge"),
        _row("cash four", "cash_withdrawal_charge"),
    ]
    test_rows = [
        _row("Shared query", "activate_my_card"),
        _row("test cash", "cash_withdrawal_charge"),
    ]

    kwargs = {
        "train_per_intent": 2,
        "dev_per_intent": 1,
        "seed": 42,
        "required_intents": ("activate_my_card", "cash_withdrawal_charge"),
    }
    first = prepare_banking77_splits(train_rows, test_rows, **kwargs)
    second = prepare_banking77_splits(train_rows, test_rows, **kwargs)

    assert first == second
    assert len(first["train"]) == 4
    assert len(first["dev"]) == 2
    assert len(first["test"]) == 2
    assert first["metadata"]["test_overlaps_removed_from_training"] == 1
    assert first["metadata"]["duplicate_training_texts_removed"] == 1

    test_texts = {row["text"].casefold().strip() for row in first["test"]}
    selected_texts = {
        " ".join(row["text"].casefold().split())
        for split in ("train", "dev")
        for row in first[split]
    }
    assert selected_texts.isdisjoint(test_texts)

    for split_name, expected_per_intent in (("train", 2), ("dev", 1)):
        counts = {
            intent: sum(
                row["expected"]["intent"] == intent for row in first[split_name]
            )
            for intent in kwargs["required_intents"]
        }
        assert counts == {
            "activate_my_card": expected_per_intent,
            "cash_withdrawal_charge": expected_per_intent,
        }


def test_sft_records_use_only_exact_train_targets() -> None:
    train_rows = [
        {
            "id": "banking77-train-0001",
            "text": "Why was I charged for cash withdrawal?",
            "expected": {"intent": "cash_withdrawal_charge"},
        }
    ]

    records = build_sft_records(train_rows)

    assert len(records) == 1
    assert [message["role"] for message in records[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert records[0]["messages"][1]["content"] == train_rows[0]["text"]
    assert json.loads(records[0]["messages"][2]["content"]) == {
        "intent": "cash_withdrawal_charge"
    }


def test_banking_prompt_has_contract_but_no_demonstrations() -> None:
    messages = build_banking_messages("Where is my card?")
    system_text = messages[0]["content"]

    assert messages[1] == {"role": "user", "content": "Where is my card?"}
    assert "activate_my_card" in system_text
    assert "wrong_exchange_rate_for_cash_withdrawal" in system_text
    assert '"intent"' in system_text
    assert "Example:" not in system_text


def test_banking_prediction_rejects_unknown_intents_and_extra_fields() -> None:
    assert BankingPrediction.model_validate(
        {"intent": "cash_withdrawal_charge"}
    ).model_dump(mode="json") == {"intent": "cash_withdrawal_charge"}

    with pytest.raises(ValidationError):
        BankingPrediction.model_validate({"intent": "made_up_intent"})

    with pytest.raises(ValidationError):
        BankingPrediction.model_validate(
            {"intent": "cash_withdrawal_charge", "explanation": "extra"}
        )


def test_banking_metrics_count_invalid_outputs_as_incorrect() -> None:
    rows = [
        {
            "expected": {"intent": "activate_my_card"},
            "prediction": {"intent": "activate_my_card"},
            "latency_ms": 10,
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
        {
            "expected": {"intent": "activate_my_card"},
            "prediction": {"intent": "cash_withdrawal_charge"},
            "latency_ms": 20,
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
        {
            "expected": {"intent": "cash_withdrawal_charge"},
            "prediction": None,
            "latency_ms": 30,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "error": "invalid output",
        },
    ]

    report = evaluate_banking_rows(rows)

    assert report["count"] == 3
    assert report["valid_output_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["intent_accuracy"] == pytest.approx(1 / 3, abs=0.0001)
    assert report["macro_intent_accuracy"] == pytest.approx(0.25, abs=0.0001)
    assert report["errors"] == 1
    assert report["total_tokens"] == 14


def test_improvement_gate_requires_delta_quality_and_validity() -> None:
    base = {"intent_accuracy": 0.50, "valid_output_rate": 1.0}
    tuned = {"intent_accuracy": 0.75, "valid_output_rate": 0.995}

    passing = assess_banking_improvement(base, tuned)
    assert passing["passed"] is True
    assert passing["accuracy_delta"] == 0.25

    failing = assess_banking_improvement(
        base,
        {"intent_accuracy": 0.75, "valid_output_rate": 0.98},
    )
    assert failing["passed"] is False
    assert "valid_output_rate" in failing["failed_gates"]
