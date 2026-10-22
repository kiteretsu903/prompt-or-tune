import asyncio
from types import SimpleNamespace

from fireworks_lab.client import FireworksClassifier


class FakeCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="this is not JSON"))
            ],
            usage=SimpleNamespace(
                prompt_tokens=17,
                completion_tokens=4,
            ),
        )


def test_invalid_json_preserves_raw_output_and_usage() -> None:
    classifier = FireworksClassifier(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
    )
    classifier.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    result = asyncio.run(
        classifier.classify(
            ticket="A test ticket",
            prompt_version="baseline",
        )
    )

    assert result.prediction is None
    assert result.error is not None
    assert result.raw_output == "this is not JSON"
    assert result.usage == {
        "prompt_tokens": 17,
        "completion_tokens": 4,
    }
