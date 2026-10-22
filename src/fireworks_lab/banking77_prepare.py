"""Download, verify, split, and export the official BANKING77 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from fireworks_lab.banking77 import build_sft_records, prepare_banking77_splits
from fireworks_lab.io_utils import write_jsonl

TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
TRAIN_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        return [dict(row) for row in reader]


def download_verified(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ValueError(
                f"Existing {path} has SHA-256 {actual}; expected {expected_sha256}"
            )
        return

    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded {url} has SHA-256 {actual}; expected {expected_sha256}"
            )
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/banking77"))
    parser.add_argument("--train-per-intent", type=int, default=25)
    parser.add_argument("--dev-per-intent", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw"
    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"
    download_verified(TRAIN_URL, train_csv, TRAIN_SHA256)
    download_verified(TEST_URL, test_csv, TEST_SHA256)

    splits = prepare_banking77_splits(
        read_source_csv(train_csv),
        read_source_csv(test_csv),
        train_per_intent=args.train_per_intent,
        dev_per_intent=args.dev_per_intent,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "train.jsonl", splits["train"])
    write_jsonl(args.output_dir / "dev.jsonl", splits["dev"])
    write_jsonl(args.output_dir / "reserve.jsonl", splits["reserve"])
    write_jsonl(args.output_dir / "test.jsonl", splits["test"])
    write_jsonl(
        args.output_dir / "sft_train.jsonl",
        build_sft_records(splits["train"]),
    )

    manifest = {
        "dataset": "PolyAI BANKING77",
        "license": "CC BY 4.0",
        "sources": {
            "train": {"url": TRAIN_URL, "sha256": TRAIN_SHA256},
            "test": {"url": TEST_URL, "sha256": TEST_SHA256},
        },
        **splits["metadata"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Prepared BANKING77: "
        f"train={manifest['train_count']}, dev={manifest['dev_count']}, "
        f"reserve={manifest['reserve_count']}, test={manifest['test_count']}, "
        "SFT test overlaps=0"
    )


if __name__ == "__main__":
    main()
