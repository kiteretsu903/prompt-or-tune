import asyncio
from types import SimpleNamespace

from fireworks_lab.banking77_client import BankingClassifier


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=101, completion_tokens=7),
        )


def _classifier(content: str) -> tuple[BankingClassifier, FakeCompletions]:
    completions = FakeCompletions(content)
    classifier = BankingClassifier(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
    )
    classifier.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return classifier, completions


def test_banking_client_returns_validated_intent_and_uses_fixed_contract() -> None:
    classifier, completions = _classifier('{"intent":"activate_my_card"}')

    result = asyncio.run(classifier.classify("Please activate my card"))

    assert result.prediction == {"intent": "activate_my_card"}
    assert result.error is None
    assert result.raw_output == '{"intent":"activate_my_card"}'
    assert result.usage == {"prompt_tokens": 101, "completion_tokens": 7}
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["reasoning_effort"] == "none"
    assert completions.kwargs["response_format"]["type"] == "json_schema"
    assert "activate_my_card" in completions.kwargs["messages"][0]["content"]


def test_banking_client_preserves_invalid_output_and_usage() -> None:
    classifier, _ = _classifier('{"intent":"made_up_intent"}')

    result = asyncio.run(classifier.classify("A test query"))

    assert result.prediction is None
    assert result.error is not None
    assert result.raw_output == '{"intent":"made_up_intent"}'
    assert result.usage == {"prompt_tokens": 101, "completion_tokens": 7}
