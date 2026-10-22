"""Leakage-safe BANKING77 preparation, prompting, validation, and metrics."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

BANKING77_INTENTS = (
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)

BankingIntent = Enum(
    "BankingIntent",
    {intent: intent for intent in BANKING77_INTENTS},
    type=str,
)


class BankingPrediction(BaseModel):
    """Strict one-intent output contract for BANKING77."""

    model_config = ConfigDict(extra="forbid")

    intent: BankingIntent


BANKING77_SYSTEM = """
You are a deterministic banking-support intent classifier.

Return exactly one JSON object with exactly one field named "intent". Its value
must be one of these allowed intent labels:

{labels}

Do not invent labels. Do not add fields, explanations, or prose outside the JSON
object. Classify the user's message by its primary intent.
""".strip().format(labels="\n".join(f"- {intent}" for intent in BANKING77_INTENTS))


def canonical_intent(value: str) -> str:
    """Normalize the two inconsistent publisher label spellings."""

    return value.strip().casefold().rstrip("?")


def normalize_text(value: str) -> str:
    """Normalize whitespace and case for exact-text leakage checks."""

    return " ".join(value.casefold().split())


def build_banking_messages(
    text: str, *, version: str = "original"
) -> list[dict[str, str]]:
    system = BANKING77_SYSTEM
    if version == "definitions-v1":
        from fireworks_lab.banking77_definitions import DEFINITIONS, GUIDANCE

        system += "\n\nLabel definitions:\n" + DEFINITIONS + "\n\n" + GUIDANCE
    elif version != "original":
        raise ValueError(f"Unknown banking prompt version: {version}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def _validate_rows(
    rows: Iterable[dict[str, str]],
    *,
    split_name: str,
    required_intents: set[str],
) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        text = row.get("text", "").strip()
        intent = canonical_intent(row.get("category", ""))
        if not text:
            raise ValueError(f"{split_name} row {index} has blank text")
        if intent not in required_intents:
            raise ValueError(f"{split_name} row {index} has unknown intent {intent!r}")
        validated.append({"text": text, "intent": intent})

    observed = {row["intent"] for row in validated}
    missing = required_intents - observed
    if missing:
        raise ValueError(f"{split_name} is missing intents: {sorted(missing)}")
    return validated


def _with_ids(split_name: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"banking77-{split_name}-{index:04d}",
            "text": row["text"],
            "expected": {"intent": row["intent"]},
        }
        for index, row in enumerate(rows, start=1)
    ]


def prepare_banking77_splits(
    train_rows: Iterable[dict[str, str]],
    test_rows: Iterable[dict[str, str]],
    *,
    train_per_intent: int = 25,
    dev_per_intent: int = 5,
    seed: int = 42,
    required_intents: Sequence[str] = BANKING77_INTENTS,
) -> dict[str, Any]:
    """Create balanced train/dev splits while keeping official test untouched."""

    if train_per_intent <= 0 or dev_per_intent <= 0:
        raise ValueError("train_per_intent and dev_per_intent must be positive")

    required = set(required_intents)
    if len(required) != len(required_intents):
        raise ValueError("required_intents contains duplicates")

    validated_test = _validate_rows(
        test_rows,
        split_name="test",
        required_intents=required,
    )
    validated_train = _validate_rows(
        train_rows,
        split_name="train",
        required_intents=required,
    )
    test_texts = {normalize_text(row["text"]) for row in validated_test}

    overlap_count = 0
    duplicate_count = 0
    seen_training_text: dict[str, str] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in validated_train:
        normalized = normalize_text(row["text"])
        if normalized in test_texts:
            overlap_count += 1
            continue
        previous_intent = seen_training_text.get(normalized)
        if previous_intent is not None:
            if previous_intent != row["intent"]:
                raise ValueError(
                    "same normalized training text has conflicting intents: "
                    f"{previous_intent!r} and {row['intent']!r}"
                )
            duplicate_count += 1
            continue
        seen_training_text[normalized] = row["intent"]
        grouped[row["intent"]].append(row)

    rng = random.Random(seed)
    train_selected: list[dict[str, str]] = []
    dev_selected: list[dict[str, str]] = []
    reserve: list[dict[str, str]] = []
    required_count = train_per_intent + dev_per_intent
    for intent in sorted(required):
        candidates = sorted(
            grouped[intent], key=lambda row: normalize_text(row["text"])
        )
        rng.shuffle(candidates)
        if len(candidates) < required_count:
            raise ValueError(
                f"intent {intent!r} has {len(candidates)} eligible rows; "
                f"needs {required_count}"
            )
        train_selected.extend(candidates[:train_per_intent])
        dev_selected.extend(candidates[train_per_intent:required_count])
        reserve.extend(candidates[required_count:])

    random.Random(seed + 1).shuffle(train_selected)
    random.Random(seed + 2).shuffle(dev_selected)
    random.Random(seed + 3).shuffle(reserve)

    return {
        "train": _with_ids("train", train_selected),
        "dev": _with_ids("dev", dev_selected),
        "reserve": _with_ids("reserve", reserve),
        "test": _with_ids("test", validated_test),
        "metadata": {
            "seed": seed,
            "intent_count": len(required),
            "train_per_intent": train_per_intent,
            "dev_per_intent": dev_per_intent,
            "test_overlaps_removed_from_training": overlap_count,
            "duplicate_training_texts_removed": duplicate_count,
            "train_count": len(train_selected),
            "dev_count": len(dev_selected),
            "reserve_count": len(reserve),
            "test_count": len(validated_test),
        },
    }


def build_sft_records(train_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize only selected training rows into OpenAI-compatible chat SFT."""

    records = []
    for row in train_rows:
        target = BankingPrediction.model_validate(row["expected"]).model_dump(
            mode="json"
        )
        records.append(
            {
                "messages": [
                    {"role": "system", "content": BANKING77_SYSTEM},
                    {"role": "user", "content": row["text"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(target, separators=(",", ":")),
                    },
                ]
            }
        )
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 2)


def evaluate_banking_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate intent accuracy with invalid outputs counted as incorrect."""

    count = len(rows)
    per_intent_total: Counter[str] = Counter()
    per_intent_correct: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    valid_count = 0
    correct_count = 0

    for row in rows:
        expected = BankingPrediction.model_validate(row["expected"]).intent.value
        per_intent_total[expected] += 1
        prediction = row.get("prediction")
        try:
            validated = BankingPrediction.model_validate(prediction)
        except ValueError:
            validated = None
        if validated is None or row.get("error") is not None:
            confusions[(expected, "<invalid>")] += 1
            continue
        valid_count += 1
        predicted = validated.intent.value
        if predicted == expected:
            correct_count += 1
            per_intent_correct[expected] += 1
        else:
            confusions[(expected, predicted)] += 1

    denominator = count or 1
    per_intent_accuracy = {
        intent: round(per_intent_correct[intent] / total, 4)
        for intent, total in sorted(per_intent_total.items())
    }
    macro_accuracy = (
        sum(per_intent_accuracy.values()) / len(per_intent_accuracy)
        if per_intent_accuracy
        else 0.0
    )
    latencies = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    ]
    prompt_tokens = sum(
        int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in rows
    )
    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows
    )
    attempt_history_known = bool(rows) and all(row.get("attempts") for row in rows)
    attempts = [attempt for row in rows for attempt in row.get("attempts", [])]

    return {
        "count": count,
        "valid_output_rate": round(valid_count / denominator, 4),
        "intent_accuracy": round(correct_count / denominator, 4),
        "macro_intent_accuracy": round(macro_accuracy, 4),
        "per_intent_accuracy": per_intent_accuracy,
        "top_confusions": [
            {"expected": expected, "predicted": predicted, "count": amount}
            for (expected, predicted), amount in sorted(
                confusions.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "errors": count - valid_count,
        "attempt_count": len(attempts) if attempt_history_known else None,
        "failed_attempt_count": (
            sum(attempt["error"] is not None for attempt in attempts)
            if attempt_history_known
            else None
        ),
        "comparison_provenance": comparison_provenance(rows),
    }


def stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def comparison_provenance(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Legacy records remain evaluable, but cannot establish comparability."""
    if not rows or any(not row.get("inference_config") for row in rows):
        return None
    config = rows[0]["inference_config"]
    required = {
        "base_model",
        "prompt_sha256",
        "schema_sha256",
        "temperature",
        "reasoning_effort",
        "max_tokens",
    }
    if not required.issubset(config) or any(
        config[key] in (None, "") for key in required
    ):
        return None
    if any(row["inference_config"] != config for row in rows):
        raise ValueError("Mixed inference configurations in one report")
    if not rows[0].get("model") or any(
        row.get("model") != rows[0]["model"] for row in rows
    ):
        raise ValueError("Mixed or missing model identities in one report")
    inputs = [{key: row[key] for key in ("id", "text", "expected")} for row in rows]
    if len({row["id"] for row in inputs}) != len(inputs):
        raise ValueError("Duplicate evaluation IDs")
    return {
        "version": 1,
        "dataset_sha256": stable_digest(sorted(inputs, key=lambda row: row["id"])),
        "inference_config": config,
    }


def assess_banking_improvement(
    base: dict[str, Any],
    tuned: dict[str, Any],
    *,
    minimum_accuracy_delta: float = 0.10,
    minimum_tuned_accuracy: float = 0.70,
    minimum_valid_output_rate: float = 0.99,
) -> dict[str, Any]:
    """Apply the experiment's predeclared fine-tuning success gates."""

    accuracy_delta = float(tuned["intent_accuracy"]) - float(base["intent_accuracy"])
    failed_gates = []
    if accuracy_delta < minimum_accuracy_delta:
        failed_gates.append("accuracy_delta")
    if float(tuned["intent_accuracy"]) < minimum_tuned_accuracy:
        failed_gates.append("tuned_accuracy")
    if float(tuned["valid_output_rate"]) < minimum_valid_output_rate:
        failed_gates.append("valid_output_rate")

    return {
        "passed": not failed_gates,
        "accuracy_delta": round(accuracy_delta, 4),
        "minimum_accuracy_delta": minimum_accuracy_delta,
        "minimum_tuned_accuracy": minimum_tuned_accuracy,
        "minimum_valid_output_rate": minimum_valid_output_rate,
        "failed_gates": failed_gates,
    }
