import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from fireworks_lab.banking77 import evaluate_banking_rows
from fireworks_lab.banking77_client import BankingClassifier
from fireworks_lab.client import FireworksClassifier


@pytest.mark.parametrize("kind", ["banking", "support"])
@pytest.mark.parametrize("budget", [1, 2])
def test_http_attempt_budget_and_saved_retry_history(kind, budget, monkeypatch):
    calls = []
    payload = (
        {"intent": "activate_my_card"}
        if kind == "banking"
        else {
            "category": "general_question",
            "priority": "P3",
            "escalate": False,
            "reason_code": "general_information",
        }
    )

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "warming"}})
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 6,
                    "total_tokens": 16,
                },
            },
        )

    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    async def exercise():
        cls = BankingClassifier if kind == "banking" else FireworksClassifier
        classifier = cls(
            api_key="dummy", base_url="https://example.invalid/v1", model="mock"
        )
        # Retain the production SDK retry setting while replacing only transport.
        original = classifier.client
        classifier.client = original.with_options(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )
        await original.close()
        try:
            args = ("test",) if kind == "banking" else ("test", "baseline")
            return await classifier.classify(*args, max_attempts=budget)
        finally:
            await classifier.client.close()

    result = asyncio.run(exercise())
    assert len(calls) == budget
    assert len(result.attempts) == budget
    first = result.attempts[0]
    assert first["status_code"] == 503
    assert "warming" in first["raw_output"]
    assert first["error"] and first["latency_ms"] >= 0
    assert first["usage"] is None
    if budget == 1:
        assert result.prediction is None and result.error
    else:
        assert result.prediction == payload and result.error is None
        assert result.attempts[1]["usage"] == result.usage


@pytest.mark.parametrize(
    "prediction,error",
    [
        ({}, None),
        ({"intent": "made_up"}, None),
        ({"intent": "activate_my_card", "extra": "forbidden"}, None),
        ({"intent": "activate_my_card"}, "failed request"),
    ],
)
def test_evaluator_rejects_invalid_or_failed_predictions(prediction, error):
    report = evaluate_banking_rows(
        [
            {
                "expected": {"intent": "activate_my_card"},
                "prediction": prediction,
                "error": error,
            }
        ]
    )
    assert report["valid_output_rate"] == 0
    assert report["intent_accuracy"] == 0
    assert report["errors"] == 1


def comparison_rows():
    return [
        {
            "id": str(i),
            "text": f"query {i}",
            "expected": {"intent": "activate_my_card"},
            "prediction": {"intent": "activate_my_card"},
            "error": None,
            "model": "base-deployment",
            "inference_config": {
                "base_model": "accounts/fireworks/models/example-base",
                "prompt_sha256": "a" * 64,
                "schema_sha256": "b" * 64,
                "temperature": 0,
                "reasoning_effort": "none",
                "max_tokens": 40,
            },
        }
        for i in range(2)
    ]


@pytest.mark.parametrize(
    "change",
    ["text", "expected", "base_model", "prompt_sha256", "max_tokens", "missing"],
)
def test_comparison_rejects_unmatched_or_missing_provenance(change):
    from fireworks_lab.banking77_compare import validate_comparison

    original = comparison_rows()
    changed = copy.deepcopy(original)
    if change == "text":
        changed[0]["text"] = "different query"
    elif change == "expected":
        changed[0]["expected"] = {"intent": "card_arrival"}
    else:
        for row in changed:
            if change == "missing":
                del row["inference_config"]
            else:
                row["inference_config"][change] = (
                    80 if change == "max_tokens" else "different"
                )
    with pytest.raises(ValueError):
        validate_comparison(
            evaluate_banking_rows(original), evaluate_banking_rows(changed)
        )


def test_comparison_accepts_same_inputs_in_different_order_and_adapter():
    from fireworks_lab.banking77_compare import validate_comparison

    base = comparison_rows()
    tuned = copy.deepcopy(base[::-1])
    for row in tuned:
        row["model"] = "tuned-deployment"
    validate_comparison(evaluate_banking_rows(base), evaluate_banking_rows(tuned))


def test_comparison_cli_rejects_legacy_reports(tmp_path, monkeypatch):
    from fireworks_lab.banking77_compare import main

    paths = [tmp_path / "base.json", tmp_path / "tuned.json"]
    for path in paths:
        path.write_text(json.dumps({"intent_accuracy": 0.8, "valid_output_rate": 1.0}))
    monkeypatch.setattr("sys.argv", ["compare", *map(str, paths)])
    with pytest.raises(ValueError, match="provenance"):
        main()


def test_attempt_counts_distinguish_recovered_errors_from_unknown_history():
    rows = comparison_rows()
    assert evaluate_banking_rows(rows)["attempt_count"] is None
    for row in rows:
        row["attempts"] = [{"error": "503 warming"}, {"error": None}]
    report = evaluate_banking_rows(rows)
    assert report["attempt_count"] == 4
    assert report["failed_attempt_count"] == 2
    assert report["errors"] == 0


def test_inference_persists_attempts_and_comparable_configuration(monkeypatch):
    from fireworks_lab.banking77_run import classify_banking_dataset
    from fireworks_lab.client import ClassificationResult

    async def exercise():
        classifier = BankingClassifier(
            api_key="dummy",
            base_url="https://example.invalid/v1",
            model="base",
            base_model="accounts/fireworks/models/example-base",
        )
        result = ClassificationResult(
            prediction={"intent": "activate_my_card"},
            latency_ms=2,
            usage={"prompt_tokens": 10, "completion_tokens": 6},
            raw_output='{"intent":"activate_my_card"}',
            error=None,
            attempts=[{"error": "503"}, {"error": None}],
        )
        monkeypatch.setattr(classifier, "classify", AsyncMock(return_value=result))
        try:
            rows = await classify_banking_dataset(
                comparison_rows(), classifier=classifier, concurrency=1
            )
        finally:
            await classifier.client.close()
        assert rows[0]["attempts"] == result.attempts
        config = rows[0]["inference_config"]
        assert config["base_model"] == classifier.base_model
        assert config["prompt_sha256"] == rows[0]["prompt_sha256"]
        assert config["max_tokens"] == classifier.request("")["max_tokens"]
        assert evaluate_banking_rows(rows)["comparison_provenance"] is not None

    asyncio.run(exercise())
