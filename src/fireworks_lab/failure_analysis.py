"""Inspect individual model failures instead of trusting aggregate metrics alone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fireworks_lab.io_utils import read_jsonl

FIELDS = ("category", "priority", "escalate", "reason_code")


def find_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        prediction = row.get("prediction")
        if prediction is None:
            mismatched = list(FIELDS)
        else:
            mismatched = [
                field
                for field in FIELDS
                if prediction.get(field) != row["expected"].get(field)
            ]
        if mismatched:
            failures.append(
                {
                    "id": row.get("id"),
                    "ticket": row.get("ticket"),
                    "expected": row.get("expected"),
                    "prediction": prediction,
                    "mismatched_fields": mismatched,
                    "error": row.get("error"),
                }
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    failures = find_failures(read_jsonl(args.input))
    print(f"Failures: {len(failures)}")
    for failure in failures[: args.limit]:
        print("\n" + "=" * 80)
        print(f"ID: {failure['id']}")
        print(f"Ticket: {failure['ticket']}")
        print(f"Mismatched: {', '.join(failure['mismatched_fields'])}")
        print(f"Expected: {failure['expected']}")
        print(f"Predicted: {failure['prediction']}")
        if failure["error"]:
            print(f"Error: {failure['error']}")


if __name__ == "__main__":
    main()
