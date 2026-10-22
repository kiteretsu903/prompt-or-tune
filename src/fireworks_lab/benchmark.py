"""Benchmark client concurrency against the same model and prompt."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fireworks_lab.io_utils import read_jsonl
from fireworks_lab.metrics import _percentile


def summarize_run(
    rows: list[dict[str, Any]], *, concurrency: int, elapsed_s: float
) -> dict[str, Any]:
    latencies = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    ]
    successes = sum(row.get("prediction") is not None for row in rows)
    count = len(rows)
    return {
        "concurrency": concurrency,
        "count": count,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_rps": round(count / elapsed_s, 3) if elapsed_s else None,
        "success_rate": round(successes / count, 4) if count else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }


async def run_benchmarks(
    *,
    rows: list[dict[str, Any]],
    concurrencies: list[int],
    model: str | None,
) -> list[dict[str, Any]]:
    from fireworks_lab.client import classifier_from_env
    from fireworks_lab.run_inference import classify_dataset

    classifier = classifier_from_env(model_override=model)
    summaries: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        started = time.perf_counter()
        results = await classify_dataset(
            rows,
            classifier=classifier,
            prompt_version="optimized",
            concurrency=concurrency,
        )
        elapsed_s = time.perf_counter() - started
        summaries.append(
            summarize_run(results, concurrency=concurrency, elapsed_s=elapsed_s)
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/concurrency.json"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--model")
    args = parser.parse_args()

    load_dotenv()
    rows = read_jsonl(args.input)[: args.limit]
    summaries = asyncio.run(
        run_benchmarks(rows=rows, concurrencies=args.concurrency, model=args.model)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    for summary in summaries:
        print(summary)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
