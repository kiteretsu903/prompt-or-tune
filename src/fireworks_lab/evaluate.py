"""Evaluate an inference result file and produce JSON plus Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fireworks_lab.io_utils import read_jsonl
from fireworks_lab.metrics import evaluate_rows


def markdown_report(report: dict[str, Any], title: str) -> str:
    return f"""# {title}

| Metric | Value |
|---|---:|
| Examples | {report["count"]} |
| Valid output rate | {report["valid_output_rate"]:.1%} |
| Category accuracy | {report["category_accuracy"]:.1%} |
| Priority accuracy | {report["priority_accuracy"]:.1%} |
| Escalation accuracy | {report["escalation_accuracy"]:.1%} |
| Reason-code accuracy | {report["reason_code_accuracy"]:.1%} |
| Exact-match accuracy | {report["exact_match_accuracy"]:.1%} |
| p50 latency | {report["p50_latency_ms"]} ms |
| p95 latency | {report["p95_latency_ms"]} ms |
| Total tokens | {report["total_tokens"]} |
| Errors | {report["errors"]} |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default="Evaluation Report")
    args = parser.parse_args()

    report = evaluate_rows(read_jsonl(args.input))
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report, args.title), encoding="utf-8")
    print(markdown_report(report, args.title))
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
