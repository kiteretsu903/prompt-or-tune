import copy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sample():
    inputs = [
        {
            "id": "one",
            "text": "Activate my card",
            "expected": {"intent": "activate_my_card"},
        }
    ]
    rows = [
        {
            **inputs[0],
            "model": "recorded://base",
            "prediction": {"intent": "activate_my_card"},
            "raw_output": '{"intent":"activate_my_card"}',
            "error": None,
            "latency_ms": 1,
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
    ]
    config = {"task": "banking", "count": 1, "model": "recorded://base"}
    return rows, inputs, config


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "text", "label", "model", "raw", "schema", "diagnostics"],
)
def test_replay_rejects_corrupted_evidence(change):
    from fireworks_lab.portfolio import validate_rows

    rows, inputs, config = sample()
    if change == "missing":
        rows.clear()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "text":
        rows[0]["text"] = "different input"
    elif change == "label":
        rows[0]["expected"] = {"intent": "card_arrival"}
    elif change == "model":
        rows[0]["model"] = "different model"
    elif change == "raw":
        rows[0]["raw_output"] = '{"intent":"card_arrival"}'
    elif change == "schema":
        rows[0]["prediction"]["extra"] = "not allowed"
    elif change == "diagnostics":
        del rows[0]["usage"]
    with pytest.raises(ValueError):
        validate_rows(rows, inputs, config)


def test_replay_accepts_preserved_failure():
    from fireworks_lab.portfolio import validate_rows

    rows, inputs, config = sample()
    rows[0].update(prediction=None, raw_output=None, error="503 scaling up")
    assert validate_rows(rows, inputs, config) == rows


def test_integrity_check_detects_changed_missing_and_extra_files(tmp_path):
    from fireworks_lab.portfolio import verify_files

    path = tmp_path / "evidence.txt"
    path.write_text("original")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/manifest.json").write_text(
        json.dumps(
            {"files": {"evidence.txt": hashlib.sha256(path.read_bytes()).hexdigest()}}
        )
    )
    assert verify_files(tmp_path) == 1
    path.write_text("changed")
    with pytest.raises(ValueError):
        verify_files(tmp_path)
    path.unlink()
    with pytest.raises(ValueError):
        verify_files(tmp_path)
    path.write_text("original")
    (tmp_path / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError):
        verify_files(tmp_path)


def test_complete_recorded_comparison_reproduces_published_results():
    from fireworks_lab.portfolio import replay

    report = replay(ROOT)
    assert report["scored_records"] == 9456
    assert report["reports"]["banking77_qwen4b_base_test"]["correct"] == 2097
    assert report["reports"]["banking77_qwen4b_tuned_test"]["correct"] == 2680
    assert report["reports"]["base_definitions-v1_confirmation"]["correct"] == 294
    assert report["reports"]["tuned_original_confirmation"]["correct"] == 318
    assert report["reports"]["base_v2"]["correct"] == 54
    assert (
        report["paired_comparisons"]["confirmation"]["lora_effect_refined"]["corrected"]
        == 32
    )


def test_split_reconstruction_and_sft_separation():
    from fireworks_lab.portfolio import verify_data

    result = verify_data(ROOT)
    assert result["train_count"] == 1925
    assert result["confirmation_count"] == 385
    assert result["overlap_count"] == 0
