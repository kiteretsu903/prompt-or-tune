"""Run a prompt/model configuration over the held-out evaluation dataset."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fireworks_lab.client import FireworksClassifier, classifier_from_env
from fireworks_lab.io_utils import read_jsonl, write_jsonl
from fireworks_lab.prompts import PromptVersion


async def classify_dataset(
    rows: list[dict[str, Any]],
    *,
    classifier: FireworksClassifier,
    prompt_version: PromptVersion,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def classify_one(row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            result = await classifier.classify(row["ticket"], prompt_version)
        return {
            **row,
            "model": classifier.model,
            "prompt_version": prompt_version,
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
    parser.add_argument("--input", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-version",
        choices=["baseline", "optimized", "optimized_v2"],
        required=True,
    )
    parser.add_argument("--model", help="Override FIREWORKS_MODEL")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv()
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    classifier = classifier_from_env(model_override=args.model)
    results = asyncio.run(
        classify_dataset(
            rows,
            classifier=classifier,
            prompt_version=args.prompt_version,
            concurrency=args.concurrency,
        )
    )
    write_jsonl(args.output, results)
    errors = sum(row["error"] is not None for row in results)
    print(f"Wrote {len(results)} results to {args.output}; errors={errors}")


if __name__ == "__main__":
    main()
