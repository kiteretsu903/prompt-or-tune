import asyncio
import json
from types import SimpleNamespace

import pytest

from fireworks_lab.banking77_prompt_experiment import run_split, select_confirmation


def test_confirmation_is_balanced_reproducible_and_rejects_overlap():
    rows = [
        {
            "id": f"{label}-{i}",
            "text": f"{label} text {i}",
            "expected": {"intent": label},
        }
        for label in ("activate_my_card", "card_arrival")
        for i in range(8)
    ]
    chosen = select_confirmation(rows, excluded=[], per_intent=3, seed=20261012)
    assert chosen == select_confirmation(rows, excluded=[], per_intent=3, seed=20261012)
    assert len(chosen) == 6
    assert sum(r["expected"]["intent"] == "card_arrival" for r in chosen) == 3
    with pytest.raises(ValueError, match="overlap"):
        select_confirmation(rows, excluded=[rows[0]], per_intent=3, seed=20261012)


def test_both_prompts_receive_identical_rows_and_errors_are_saved(
    tmp_path, monkeypatch
):
    import fireworks_lab.banking77_prompt_experiment as runner

    calls = []

    async def classify(rows, *, classifier, concurrency):
        calls.append((classifier.prompt_version, rows))
        return [
            {
                **r,
                "prediction": None,
                "error": "preserved failure",
                "raw_output": "bad",
                "latency_ms": 2,
                "usage": {"prompt_tokens": 9},
            }
            for r in rows
        ]

    monkeypatch.setattr(runner, "classify_banking_dataset", classify)
    clients = {
        v: SimpleNamespace(prompt_version=v) for v in ("original", "definitions-v1")
    }
    rows = [{"id": "one", "text": "x", "expected": {"intent": "activate_my_card"}}]
    asyncio.run(
        run_split(
            rows,
            clients=clients,
            output_dir=tmp_path,
            model_name="base",
            split="dev",
            concurrency=1,
        )
    )
    assert [c[1] for c in calls] == [rows, rows]
    for version in clients:
        saved = json.loads((tmp_path / f"base_{version}_dev.jsonl").read_text())
        assert saved["error"] == "preserved failure"
        assert saved["raw_output"] == "bad"
        assert saved["usage"] == {"prompt_tokens": 9}
    with pytest.raises(FileExistsError):
        asyncio.run(
            run_split(
                rows,
                clients=clients,
                output_dir=tmp_path,
                model_name="base",
                split="dev",
                concurrency=1,
            )
        )
