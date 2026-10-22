"""Frozen 2x2 BANKING77 prompt comparison; no training or deployment changes.

Prepare snapshots offline first, then explicitly --run the saved experiment.
Each batch is checkpointed; existing prediction files are never overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from fireworks_lab.banking77 import build_banking_messages, normalize_text
from fireworks_lab.banking77_client import BankingClassifier
from fireworks_lab.banking77_run import classify_banking_dataset
from fireworks_lab.io_utils import read_jsonl, write_jsonl

VERSIONS = ("original", "definitions-v1")
BASE_MODEL = "accounts/fireworks/models/qwen3-4b-instruct-2507"
MODELS = {
    "base": "recorded://qwen3-4b-base",
    "tuned": "recorded://qwen3-4b-banking77-lora",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_confirmation(reserve, *, excluded, per_intent=5, seed=20261012):
    excluded_texts = {normalize_text(r["text"]) for r in excluded}
    reserve_texts = [normalize_text(r["text"]) for r in reserve]
    if excluded_texts.intersection(reserve_texts):
        raise ValueError("Reserve text overlap with excluded train/dev/test data")
    if len(set(reserve_texts)) != len(reserve_texts):
        raise ValueError("Duplicate reserve texts")
    grouped = defaultdict(list)
    for row in reserve:
        grouped[row["expected"]["intent"]].append(row)
    rng = random.Random(seed)
    chosen = []
    for intent in sorted(grouped):
        chosen.extend(rng.sample(grouped[intent], per_intent))
    rng.shuffle(chosen)
    return chosen


def prepare(output_dir: Path):
    data = Path("data/banking77")
    sources = {
        s: data / f"{s}.jsonl" for s in ("train", "dev", "test", "reserve", "sft_train")
    }
    splits = {s: read_jsonl(p) for s, p in sources.items() if s != "sft_train"}
    confirmation = select_confirmation(
        splits["reserve"], excluded=splits["train"] + splits["dev"] + splits["test"]
    )
    exported_texts = {
        normalize_text(r["messages"][1]["content"])
        for r in read_jsonl(sources["sft_train"])
    }
    assert not exported_texts.intersection(
        normalize_text(r["text"]) for r in confirmation
    )
    assert len(confirmation) == 385 and len(splits["dev"]) == 385
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / ".gitignore").write_text("*\n!.gitignore\n")
    write_jsonl(output_dir / "dev_input.jsonl", splits["dev"])
    write_jsonl(output_dir / "confirmation_input.jsonl", confirmation)
    # Warm-up is from training, never from either measured split.
    write_jsonl(output_dir / "warmup_input.jsonl", splits["train"][:1])
    prompt_hashes = {}
    for version in VERSIONS:
        path = output_dir / f"prompt_{version}.txt"
        path.write_text(build_banking_messages("", version=version)[0]["content"])
        prompt_hashes[version] = digest(path)
    package = Path(__file__).parent
    code_paths = [
        package / name
        for name in (
            "banking77.py",
            "banking77_definitions.py",
            "banking77_client.py",
            "banking77_run.py",
            "banking77_prompt_experiment.py",
            "client.py",
        )
    ]
    manifest = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "hypothesis": "Definitions improve base accuracy by >=5 percentage points with >=99% valid output; separately measure whether the LoRA advantage shrinks.",
        "models": MODELS,
        "base_model": BASE_MODEL,
        "prompt_versions": list(VERSIONS),
        "prompt_sha256": prompt_hashes,
        "confirmation_seed": 20261012,
        "confirmation_per_intent": 5,
        "confirmation_provenance": "Previously unused original-training reserve, disjoint by normalized text from SFT/train/dev/official test. Not a new external-domain test.",
        "prior_exposure": "Official-test aggregate and some per-intent outcomes were discussed before this experiment. New prompt authored using label names and four selected-training examples per label, not confirmation examples. Development is reused. No new blind official-test claim.",
        "settings": {
            "temperature": 0,
            "reasoning_effort": "none",
            "max_tokens": 40,
            "concurrency_per_model": 8,
            "batch_size": 32,
        },
        "request_count": 3080,
        "request_count_unit": "logical classifications; HTTP retries are separate",
        "warmup_request_limit_per_model": 4,
        "source_sha256": {str(p): digest(p) for p in sources.values()},
        "code_sha256": {str(p): digest(p) for p in code_paths},
        "snapshot_sha256": {
            p.name: digest(p)
            for p in output_dir.iterdir()
            if p.suffix in (".txt", ".jsonl")
        },
        "decision_rule": "Run all four arms on both splits once; do not revise definitions after seeing development results. Report all arms, paired changes, errors, latency and tokens. Confirmation is the primary check; >=5 pp is a practical threshold, not a statistical significance claim.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Frozen {manifest['request_count']} measured requests in {output_dir}",
        flush=True,
    )


async def run_split(rows, *, clients, output_dir, model_name, split, concurrency):
    paths = {v: output_dir / f"{model_name}_{v}_{split}.jsonl" for v in VERSIONS}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    for path in paths.values():
        path.touch(exist_ok=False)
    for batch_index, start in enumerate(range(0, len(rows), 32)):
        # Alternate which prompt runs first to reduce systematic time-order bias.
        versions = VERSIONS if batch_index % 2 == 0 else VERSIONS[::-1]
        for version in versions:
            results = await classify_banking_dataset(
                rows[start : start + 32],
                classifier=clients[version],
                concurrency=concurrency,
            )
            now = datetime.now(UTC).isoformat()
            with paths[version].open("a") as stream:
                for row in results:
                    stream.write(
                        json.dumps({**row, "batch_finished_at_utc": now}) + "\n"
                    )
            errors = sum(r.get("error") is not None for r in results)
            print(
                f"{model_name} {split} {version}: {min(start + 32, len(rows))}/{len(rows)}, batch errors={errors}",
                flush=True,
            )
            if errors >= 4:
                raise RuntimeError(
                    "Stopping after >=4 errors in a batch; partial raw records preserved"
                )


async def run(output_dir: Path):
    manifest = json.loads((output_dir / "manifest.json").read_text())
    for name, expected in manifest["snapshot_sha256"].items():
        if digest(output_dir / name) != expected:
            raise ValueError(f"Changed frozen snapshot: {name}")
    for group in ("source_sha256", "code_sha256"):
        for name, expected in manifest[group].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Changed frozen source: {name}")
    for version in VERSIONS:
        prompt = build_banking_messages("", version=version)[0]["content"]
        if (
            hashlib.sha256(prompt.encode()).hexdigest()
            != manifest["prompt_sha256"][version]
        ):
            raise ValueError("Changed frozen prompt")
    load_dotenv()
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ.get(
        "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
    )
    if not base_url.startswith("https://api.fireworks.ai/"):
        raise ValueError("Expected the Fireworks API endpoint")
    # Claim the run before issuing any billable request.
    with (output_dir / "run_started.json").open("x") as stream:
        json.dump({"started_at_utc": datetime.now(UTC).isoformat()}, stream)

    async def run_model(name, model):
        clients = {
            v: BankingClassifier(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt_version=v,
                base_model=manifest["base_model"],
            )
            for v in VERSIONS
        }
        try:
            warm_text = read_jsonl(output_dir / "warmup_input.jsonl")[0]["text"]
            with (output_dir / f"{name}_warmup.jsonl").open("x") as stream:
                for attempt in range(4):
                    response = await clients["original"].classify(
                        warm_text, max_attempts=2
                    )
                    stream.write(
                        json.dumps(
                            {"model": model, "attempt": attempt + 1, **asdict(response)}
                        )
                        + "\n"
                    )
                    stream.flush()
                    print(
                        f"{name} warmup {attempt + 1}: {'ready' if response.error is None else response.error[:180]}",
                        flush=True,
                    )
                    if response.error is None:
                        break
                    await asyncio.sleep(10)
                else:
                    raise RuntimeError(f"{name} did not warm up; diagnostics preserved")
            for split in ("dev", "confirmation"):
                await run_split(
                    read_jsonl(output_dir / f"{split}_input.jsonl"),
                    clients=clients,
                    output_dir=output_dir,
                    model_name=name,
                    split=split,
                    concurrency=manifest["settings"]["concurrency_per_model"],
                )
        finally:
            for client in clients.values():
                await client.client.close()

    outcomes = await asyncio.gather(
        *(run_model(n, m) for n, m in manifest["models"].items()),
        return_exceptions=True,
    )
    failures = [str(o) for o in outcomes if isinstance(o, BaseException)]
    (output_dir / "run_finished.json").write_text(
        json.dumps(
            {"finished_at_utc": datetime.now(UTC).isoformat(), "failures": failures},
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run", action="store_true", help="Run an already prepared frozen experiment"
    )
    args = parser.parse_args()
    if args.run:
        asyncio.run(run(args.output_dir))
    else:
        prepare(args.output_dir)


if __name__ == "__main__":
    main()
