"""Evaluate BANKING77 inference output and save JSON plus Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fireworks_lab.banking77 import evaluate_banking_rows
from fireworks_lab.io_utils import read_jsonl


def markdown_report(report: dict[str, Any], title: str) -> str:
    attempts = report.get("attempt_count")
    failed_attempts = report.get("failed_attempt_count")
    confusion_rows = "\n".join(
        f"| `{row['expected']}` | `{row['predicted']}` | {row['count']} |"
        for row in report["top_confusions"][:10]
    )
    if not confusion_rows:
        confusion_rows = "| - | - | 0 |"
    return f"""# {title}

| Metric | Value |
|---|---:|
| Examples | {report["count"]} |
| Valid output rate | {report["valid_output_rate"]:.1%} |
| Intent accuracy | {report["intent_accuracy"]:.1%} |
| Macro intent accuracy | {report["macro_intent_accuracy"]:.1%} |
| p50 latency | {report["p50_latency_ms"]} ms |
| p95 latency | {report["p95_latency_ms"]} ms |
| Total tokens | {report["total_tokens"]} |
| Final classification failures | {report["errors"]} |
| HTTP attempts | {attempts if attempts is not None else "Unknown (not recorded)"} |
| Failed attempts (including recovered failures) | {failed_attempts if failed_attempts is not None else "Unknown (not recorded)"} |

## Top confusions

| Expected | Predicted | Count |
|---|---|---:|
{confusion_rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default="BANKING77 Evaluation")
    args = parser.parse_args()

    report = evaluate_banking_rows(read_jsonl(args.input))
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report, args.title), encoding="utf-8")
    print(markdown_report(report, args.title))
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
