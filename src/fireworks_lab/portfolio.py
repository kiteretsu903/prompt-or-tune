"""Offline inspection and verification of the published classification records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from fireworks_lab.banking77 import (
    BankingPrediction,
    build_sft_records,
    evaluate_banking_rows,
    normalize_text,
    prepare_banking77_splits,
)
from fireworks_lab.banking77_prepare import (
    TEST_SHA256,
    TRAIN_SHA256,
    read_source_csv,
)
from fireworks_lab.banking77_prompt_experiment import select_confirmation
from fireworks_lab.io_utils import read_jsonl
from fireworks_lab.metrics import evaluate_rows
from fireworks_lab.schemas import Prediction
from fireworks_lab.training_export import to_sft_record

IGNORED = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "output",
    "dist",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows, inputs, config):
    """Align each complete run to its frozen inputs and inspect its raw outputs."""
    indexed = {r["id"]: r for r in inputs}
    require(len(indexed) == len(inputs), "Duplicate input IDs")
    require(len(rows) == len(inputs) == config["count"], "Incomplete run")
    by_id = {r["id"]: r for r in rows}
    require(
        len(by_id) == len(rows) and set(by_id) == set(indexed),
        "Duplicate or mismatched result IDs",
    )
    field = "text" if config["task"] == "banking" else "ticket"
    schema = BankingPrediction if config["task"] == "banking" else Prediction
    for row in rows:
        original = indexed[row["id"]]
        require(row[field] == original[field], "Changed input text")
        require(row["expected"] == original["expected"], "Changed expected label")
        schema.model_validate(row["expected"])
        require(row["model"] == config["model"], "Wrong recorded model")
        require(
            all(k in row for k in ("error", "raw_output", "usage", "latency_ms")),
            "Missing diagnostics",
        )
        require(
            isinstance(row["usage"], dict) and row["latency_ms"] >= 0,
            "Invalid diagnostics",
        )
        if config.get("prompt_version"):
            require(
                row.get("prompt_version") == config["prompt_version"],
                "Wrong prompt version",
            )
        if row.get("prediction") is None:
            require(bool(row["error"]), "Missing failure explanation")
        else:
            require(row["error"] is None, "Prediction carries a final error")
            predicted = schema.model_validate(row["prediction"]).model_dump(mode="json")
            raw = schema.model_validate_json(row["raw_output"]).model_dump(mode="json")
            require(predicted == raw, "Raw output disagrees with prediction")
    return [by_id[r["id"]] for r in inputs]


def paired(before, after, a, b):
    groups = defaultdict(list)
    corrected = regressed = 0
    for x, y in zip(before, after, strict=True):
        require(x["id"] == y["id"], "Paired inputs do not align")
        ca = x["prediction"] == x["expected"]
        cb = y["prediction"] == y["expected"]
        corrected += not ca and cb
        regressed += ca and not cb
        groups[x["expected"]["intent"]].append(int(cb) - int(ca))
    rng = random.Random(20261012)
    samples = sorted(
        sum(sum(rng.choices(ds, k=len(ds))) for ds in groups.values()) / len(before)
        for _ in range(5000)
    )
    return {
        "before": list(a),
        "after": list(b),
        "corrected": corrected,
        "regressed": regressed,
        "delta_pp": 100 * (corrected - regressed) / len(before),
        "paired_stratified_bootstrap_95_percentile_pp": [
            100 * samples[125],
            100 * samples[4874],
        ],
    }


def replay(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "evaluation/protocol.json").read_text())
    expected = json.loads((root / "evaluation/expected_results.json").read_text())
    reports = {}
    records = {}
    for name, config in protocol["runs"].items():
        rows = validate_rows(
            read_jsonl(root / config["path"]),
            read_jsonl(root / config["inputs"]),
            config,
        )
        if config.get("prompt_version") and config["task"] == "banking":
            prompt = root / f"evaluation/prompts/prompt_{config['prompt_version']}.txt"
            require(
                all(r["prompt_sha256"] == sha256(prompt) for r in rows),
                f"Prompt digest mismatch in {name}",
            )
        metric = evaluate_banking_rows if config["task"] == "banking" else evaluate_rows
        report = metric(rows)
        report.update(
            correct=sum(r["prediction"] == r["expected"] for r in rows),
            api_errors=sum(r["error"] is not None for r in rows),
        )
        report["accuracy_exact"] = report["correct"] / len(rows)
        for key, value in expected["reports"][name].items():
            require(report.get(key) == value, f"Result mismatch: {name}/{key}")
        reports[name] = report
        records[name] = rows
    comparisons = {}
    pairs = {
        "prompt_effect_base": (("base", "original"), ("base", "definitions-v1")),
        "prompt_effect_tuned": (("tuned", "original"), ("tuned", "definitions-v1")),
        "lora_effect_original": (("base", "original"), ("tuned", "original")),
        "lora_effect_refined": (
            ("base", "definitions-v1"),
            ("tuned", "definitions-v1"),
        ),
        "base_refined_vs_tuned_original": (
            ("tuned", "original"),
            ("base", "definitions-v1"),
        ),
    }
    for split in ("dev", "confirmation"):
        comparisons[split] = {}
        for name, (a, b) in pairs.items():
            before = records[f"{a[0]}_{a[1]}_{split}"]
            after = records[f"{b[0]}_{b[1]}_{split}"]
            result = paired(before, after, a, b)
            require(
                result == expected["paired_comparisons"][split][name],
                f"Paired result mismatch: {split}/{name}",
            )
            comparisons[split][name] = result
    return {
        "scored_records": sum(r["count"] for r in reports.values()),
        "reports": reports,
        "paired_comparisons": comparisons,
    }


def verify_data(root: Path) -> dict[str, int]:
    data = root / "data/banking77"
    require(
        sha256(data / "raw/train.csv") == TRAIN_SHA256, "Source train hash mismatch"
    )
    require(sha256(data / "raw/test.csv") == TEST_SHA256, "Source test hash mismatch")
    splits = prepare_banking77_splits(
        read_source_csv(data / "raw/train.csv"), read_source_csv(data / "raw/test.csv")
    )
    for split in ("train", "dev", "test", "reserve"):
        require(
            splits[split] == read_jsonl(data / f"{split}.jsonl"),
            f"Split does not reproduce: {split}",
        )
    require(
        build_sft_records(splits["train"]) == read_jsonl(data / "sft_train.jsonl"),
        "SFT export does not match selected training rows",
    )
    confirmation = select_confirmation(
        splits["reserve"], excluded=splits["train"] + splits["dev"] + splits["test"]
    )
    require(
        confirmation == read_jsonl(root / "evaluation/inputs/confirmation_input.jsonl"),
        "Confirmation sample does not reproduce",
    )
    require(
        splits["dev"] == read_jsonl(root / "evaluation/inputs/dev_input.jsonl"),
        "Development snapshot differs",
    )
    evidence = json.loads(
        (root / "experiments/followup/definition_training_evidence.json").read_text()
    )
    training = {r["id"]: r for r in splits["train"]}
    require(
        len(evidence) == 77 and all(len(v) == 4 for v in evidence.values()),
        "Definition evidence must contain four examples per label",
    )
    for label, examples in evidence.items():
        require(
            all(
                training.get(r["id"]) == r and r["expected"]["intent"] == label
                for r in examples
            ),
            "Definition evidence is not selected training data",
        )
    support_train = read_jsonl(root / "data/train.jsonl")
    support_eval = read_jsonl(root / "data/eval.jsonl")
    require(
        [to_sft_record(r) for r in support_train]
        == read_jsonl(root / "data/sft_train.jsonl"),
        "Support SFT export mismatch",
    )
    train_texts = {normalize_text(r["ticket"]) for r in support_train}
    eval_texts = {normalize_text(r["ticket"]) for r in support_eval}
    require(not train_texts.intersection(eval_texts), "Support text overlap")
    return {
        "train_count": len(splits["train"]),
        "confirmation_count": len(confirmation),
        "overlap_count": 0,
        "support_train_count": len(support_train),
    }


def public_files(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(
            part in IGNORED or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.is_file() and relative.as_posix() != "evaluation/manifest.json":
            require(not path.is_symlink(), f"Symlink is not a release file: {relative}")
            files.add(relative.as_posix())
    return files


def verify_files(root: Path) -> int:
    manifest = json.loads((root / "evaluation/manifest.json").read_text())
    wanted = manifest["files"]
    require(public_files(root) == set(wanted), "Release file inventory differs")
    for name, expected in wanted.items():
        require(sha256(root / name) == expected, f"Release file hash mismatch: {name}")
    return len(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "replay", "verify"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Portfolio checkout containing evaluation/ and data/",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        example = json.loads((args.root / "examples/label_convention.json").read_text())
        print("Recorded comparison. No model is called.\n")
        print("Query:", example["text"])
        print("Expected:", example["expected"]["intent"])
        for name, row in example["runs"].items():
            print(name + ":", row["prediction"]["intent"])
        return
    if args.command == "verify":
        count = verify_files(args.root)
        data = verify_data(args.root)
    result = replay(args.root)
    if args.command == "verify":
        result.update(verified_files=count, data_checks=data)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Verified {result['scored_records']:,} recorded classifications in "
        f"{len(result['reports'])} complete runs."
    )
    for name, report in result["reports"].items():
        print(f"{name}: {report['correct']}/{report['count']}")
    print("Paired changes and all ten bootstrap intervals match the saved results.")
    if args.command == "verify":
        print(f"{count} file hashes verified; data splits and SFT separation verified.")


if __name__ == "__main__":
    main()
