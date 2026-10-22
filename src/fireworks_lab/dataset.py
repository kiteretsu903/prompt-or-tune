"""Generate a small, reproducible ticket dataset without private data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

Case = tuple[str, str, str, bool, str]

TRAIN_CASES: list[Case] = [
    (
        "Someone used my stolen password to sign in from another country.",
        "security_incident",
        "P0",
        True,
        "credential_theft",
    ),
    (
        "I received a login alert I did not cause and my password was changed.",
        "security_incident",
        "P0",
        True,
        "credential_theft",
    ),
    (
        "My saved project disappeared after the update and cannot be restored.",
        "bug_report",
        "P0",
        True,
        "data_loss",
    ),
    (
        "The app has deleted my work twice this week.",
        "bug_report",
        "P0",
        True,
        "data_loss",
    ),
    (
        "I see two separate $720 charges for the same annual plan.",
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        "You billed my card twice for $899 each.",
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        (
            "Hey, I got a $892 subscription fee billed on my credit card 3 times, "
            "so the total is $2676."
        ),
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        "Please refund last month's $29 subscription renewal.",
        "billing_refund",
        "P2",
        False,
        "ordinary_refund",
    ),
    (
        "I cancelled yesterday and need the $80 charge reversed.",
        "billing_refund",
        "P2",
        False,
        "ordinary_refund",
    ),
    (
        "Checkout fails every time, so our team cannot purchase anything.",
        "bug_report",
        "P1",
        True,
        "service_blocking_bug",
    ),
    (
        "The export button does nothing and blocks our weekly reporting.",
        "bug_report",
        "P1",
        True,
        "service_blocking_bug",
    ),
    (
        "The settings icon is slightly misaligned on the mobile page.",
        "bug_report",
        "P3",
        False,
        "cosmetic_bug",
    ),
    (
        "The dark-mode border is the wrong shade of gray.",
        "bug_report",
        "P3",
        False,
        "cosmetic_bug",
    ),
    (
        "Could you add bulk editing for account tags?",
        "feature_request",
        "P3",
        False,
        "product_enhancement",
    ),
    (
        "We would like a native integration with our CRM.",
        "feature_request",
        "P3",
        False,
        "product_enhancement",
    ),
    (
        "My account is locked after too many sign-in attempts.",
        "account_access",
        "P2",
        False,
        "locked_account",
    ),
    (
        "I know my password, but the account says it is temporarily locked.",
        "account_access",
        "P2",
        False,
        "locked_account",
    ),
    (
        "Where can I download invoices for the current year?",
        "general_question",
        "P3",
        False,
        "general_information",
    ),
    (
        "Does the business plan include audit logs?",
        "general_question",
        "P3",
        False,
        "general_information",
    ),
]

EVAL_CASES: list[Case] = [
    (
        "An unfamiliar device accessed my profile and I did not authorize it.",
        "security_incident",
        "P0",
        True,
        "credential_theft",
    ),
    (
        "My credentials appear compromised; there are sessions I do not recognize.",
        "security_incident",
        "P0",
        True,
        "credential_theft",
    ),
    (
        "Three documents vanished after syncing, including the recovery copies.",
        "bug_report",
        "P0",
        True,
        "data_loss",
    ),
    (
        "Edits keep being erased whenever the desktop client reconnects.",
        "bug_report",
        "P0",
        True,
        "data_loss",
    ),
    (
        "There are duplicate $650 transactions for one license purchase.",
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        "One order resulted in two charges of $1,200 apiece.",
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        (
            "I got charged $2676 incorrectly. There should be one single $892 "
            "charge. Please help."
        ),
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    ),
    (
        "Could I get back the $45 fee from my accidental renewal?",
        "billing_refund",
        "P2",
        False,
        "ordinary_refund",
    ),
    (
        "Please reverse a single $300 charge because we no longer need the plan.",
        "billing_refund",
        "P2",
        False,
        "ordinary_refund",
    ),
    (
        "No employee can submit time sheets because Save always crashes.",
        "bug_report",
        "P1",
        True,
        "service_blocking_bug",
    ),
    (
        "Search returns an error for every query, preventing all case work.",
        "bug_report",
        "P1",
        True,
        "service_blocking_bug",
    ),
    (
        "The logo has two pixels of extra padding in Safari.",
        "bug_report",
        "P3",
        False,
        "cosmetic_bug",
    ),
    (
        "A tooltip overlaps the corner of a button but the button still works.",
        "bug_report",
        "P3",
        False,
        "cosmetic_bug",
    ),
    (
        "Please support scheduled exports to cloud storage.",
        "feature_request",
        "P3",
        False,
        "product_enhancement",
    ),
    (
        "Can the product gain a dashboard for usage trends?",
        "feature_request",
        "P3",
        False,
        "product_enhancement",
    ),
    (
        "The system locked me out after several typos, but the account is mine.",
        "account_access",
        "P2",
        False,
        "locked_account",
    ),
    (
        "My valid account is inaccessible because the lockout timer never ends.",
        "account_access",
        "P2",
        False,
        "locked_account",
    ),
    (
        "How long are activity logs retained?",
        "general_question",
        "P3",
        False,
        "general_information",
    ),
    (
        "Which file formats can the import tool read?",
        "general_question",
        "P3",
        False,
        "general_information",
    ),
]


def generate_examples(
    *, split: str, count: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Generate deterministic examples while keeping train/eval wording separate."""

    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    if count <= 0:
        raise ValueError("count must be positive")

    cases = TRAIN_CASES if split == "train" else EVAL_CASES
    rng = random.Random(f"{seed}:{split}")
    rows: list[dict[str, Any]] = []

    for index in range(count):
        ticket, category, priority, escalate, reason_code = cases[index % len(cases)]
        variation = index // len(cases)
        if variation:
            prefixes = [
                "Customer report: ",
                "Support request: ",
                "Enterprise user says: ",
                "New ticket: ",
            ]
            ticket = f"{rng.choice(prefixes)}{ticket} Reference {split.upper()}-{variation:02d}."
        rows.append(
            {
                "id": f"{split}-{index:04d}",
                "ticket": ticket,
                "expected": {
                    "category": category,
                    "priority": priority,
                    "escalate": escalate,
                    "reason_code": reason_code,
                },
            }
        )
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=180)
    parser.add_argument("--eval-count", type=int, default=54)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    write_jsonl(
        Path("data/train.jsonl"),
        generate_examples(split="train", count=args.train_count, seed=args.seed),
    )
    write_jsonl(
        Path("data/eval.jsonl"),
        generate_examples(split="eval", count=args.eval_count, seed=args.seed),
    )
    print(
        f"Wrote {args.train_count} training and {args.eval_count} evaluation examples."
    )


if __name__ == "__main__":
    main()
