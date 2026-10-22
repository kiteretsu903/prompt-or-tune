"""Convert labeled tickets into Fireworks/OpenAI chat-format SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fireworks_lab.io_utils import read_jsonl, write_jsonl
from fireworks_lab.policy import POLICY_TEXT


def to_sft_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a support-routing engine.\n\n" + POLICY_TEXT,
            },
            {"role": "user", "content": example["ticket"]},
            {
                "role": "assistant",
                "content": json.dumps(example["expected"], separators=(",", ":")),
            },
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/sft_train.jsonl"))
    args = parser.parse_args()

    records = [to_sft_record(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} SFT records to {args.output}")


if __name__ == "__main__":
    main()
