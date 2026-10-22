import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from fireworks_lab.banking77 import BANKING77_SYSTEM, build_banking_messages
from fireworks_lab.banking77_client import BankingClassifier
from fireworks_lab.banking77_run import classify_banking_dataset


def test_original_prompt_is_unchanged_and_unknown_versions_fail():
    assert (
        build_banking_messages("query", version="original")[0]["content"]
        == BANKING77_SYSTEM
    )
    with pytest.raises(ValueError, match="prompt version"):
        build_banking_messages("query", version="typo")


def test_refined_prompt_reaches_api_and_is_identified_in_saved_records():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"intent":"get_physical_card"}')
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2000, completion_tokens=8),
        )

    classifier = BankingClassifier(
        api_key="test",
        base_url="https://example.invalid",
        model="test-model",
        prompt_version="definitions-v1",
    )
    classifier.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    rows = [
        {
            "id": "one",
            "text": "Where is my PIN?",
            "expected": {"intent": "get_physical_card"},
        }
    ]
    result = asyncio.run(
        classify_banking_dataset(rows, classifier=classifier, concurrency=1)
    )[0]
    prompt = calls[0]["messages"][0]["content"]
    assert "get_physical_card: Finding or retrieving the existing card PIN" in prompt
    assert result["prompt_version"] == "definitions-v1"
    assert result["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert result["task_version"] != "banking77-zero-shot-v1"
    assert result["expected"] == rows[0]["expected"]
    assert result["raw_output"] == '{"intent":"get_physical_card"}'
    assert result["usage"]["prompt_tokens"] == 2000
    assert calls[0]["temperature"] == 0
    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["max_tokens"] == 40


def test_refined_definitions_cover_every_label_once():
    from fireworks_lab.banking77 import BANKING77_INTENTS

    prompt = build_banking_messages("query", version="definitions-v1")[0]["content"]
    for label in BANKING77_INTENTS:
        assert prompt.count(f"\n{label}: ") == 1
