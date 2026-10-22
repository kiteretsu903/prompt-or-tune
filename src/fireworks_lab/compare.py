"""Compare multiple evaluation JSON reports in one Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = [
    "valid_output_rate",
    "exact_match_accuracy",
    "category_accuracy",
    "priority_accuracy",
    "escalation_accuracy",
    "p50_latency_ms",
    "p95_latency_ms",
    "total_tokens",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args()

    loaded = [(path.stem, json.loads(path.read_text())) for path in args.reports]
    header = "| System | " + " | ".join(METRICS) + " |"
    separator = "|---|" + "---:|" * len(METRICS)
    lines = ["# System Comparison", "", header, separator]
    for name, report in loaded:
        values = []
        for metric in METRICS:
            value = report.get(metric)
            if metric.endswith(("_rate", "_accuracy")):
                values.append(f"{value:.1%}")
            else:
                values.append(str(value))
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
