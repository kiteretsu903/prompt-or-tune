import asyncio

from fireworks_lab.banking77_run import classify_banking_dataset
from fireworks_lab.client import ClassificationResult


class FakeBankingClassifier:
    model = "test-banking-model"

    async def classify(self, text: str) -> ClassificationResult:
        intent = "activate_my_card" if "activate" in text else "card_arrival"
        return ClassificationResult(
            prediction={"intent": intent},
            latency_ms=12.5,
            usage={"prompt_tokens": 10, "completion_tokens": 3},
            raw_output=f'{{"intent":"{intent}"}}',
            error=None,
        )


def test_classify_banking_dataset_preserves_input_and_diagnostics() -> None:
    rows = [
        {
            "id": "one",
            "text": "activate my card",
            "expected": {"intent": "activate_my_card"},
        },
        {
            "id": "two",
            "text": "where is my card",
            "expected": {"intent": "card_arrival"},
        },
    ]

    results = asyncio.run(
        classify_banking_dataset(
            rows,
            classifier=FakeBankingClassifier(),
            concurrency=2,
        )
    )

    assert [row["id"] for row in results] == ["one", "two"]
    assert results[0]["model"] == "test-banking-model"
    assert results[0]["prediction"] == {"intent": "activate_my_card"}
    assert results[0]["raw_output"] == '{"intent":"activate_my_card"}'
    assert results[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    assert results[0]["error"] is None
