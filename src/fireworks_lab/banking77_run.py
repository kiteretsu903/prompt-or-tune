"""Run a Fireworks model over a prepared BANKING77 split."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fireworks_lab.banking77 import build_banking_messages, stable_digest
from fireworks_lab.banking77_client import BankingClassifier
from fireworks_lab.io_utils import read_jsonl, write_jsonl


async def classify_banking_dataset(
    rows: list[dict[str, Any]],
    *,
    classifier: BankingClassifier,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    prompt_version = getattr(classifier, "prompt_version", "original")
    system = build_banking_messages("", version=prompt_version)[0]["content"]
    prompt_sha256 = hashlib.sha256(system.encode()).hexdigest()
    inference_config = None
    if getattr(classifier, "base_model", None):
        request = classifier.request("")
        inference_config = {
            "base_model": classifier.base_model,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": stable_digest(request["response_format"]),
            **{
                key: request[key]
                for key in ("temperature", "reasoning_effort", "max_tokens")
            },
        }

    async def classify_one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await classifier.classify(row["text"])
        return {
            **row,
            "model": classifier.model,
            "task_version": (
                "banking77-zero-shot-v1"
                if prompt_version == "original"
                else f"banking77-{prompt_version}"
            ),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "inference_config": inference_config,
            "prediction": result.prediction,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
            "raw_output": result.raw_output,
            "error": result.error,
            "attempts": result.attempts,
        }

    return await asyncio.gather(*(classify_one(row) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Verified underlying untuned model ID, identical for both arms",
    )
    parser.add_argument(
        "--prompt-version",
        choices=("original", "definitions-v1"),
        default="original",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")

    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    classifier = BankingClassifier(
        api_key=api_key,
        base_url=os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        ),
        model=args.model,
        base_model=args.base_model,
        prompt_version=args.prompt_version,
    )
    results = asyncio.run(
        classify_banking_dataset(
            rows,
            classifier=classifier,
            concurrency=args.concurrency,
        )
    )
    write_jsonl(args.output, results)
    errors = sum(row["error"] is not None for row in results)
    print(f"Wrote {len(results)} results to {args.output}; errors={errors}")


if __name__ == "__main__":
    main()
