import json

from fireworks_lab.dataset import (
    EVAL_CASES,
    TRAIN_CASES,
    generate_examples,
    write_jsonl,
)


def test_large_duplicate_charge_wordings_stay_split() -> None:
    train_case = (
        (
            "Hey, I got a $892 subscription fee billed on my credit card 3 times, "
            "so the total is $2676."
        ),
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    )
    eval_case = (
        (
            "I got charged $2676 incorrectly. There should be one single $892 "
            "charge. Please help."
        ),
        "billing_refund",
        "P1",
        True,
        "duplicate_charge_over_500",
    )

    assert train_case in TRAIN_CASES
    assert eval_case in EVAL_CASES

    train_ticket = train_case[0]
    eval_ticket = eval_case[0]
    assert train_ticket not in {case[0] for case in EVAL_CASES}
    assert eval_ticket not in {case[0] for case in TRAIN_CASES}


def test_generate_examples_is_deterministic_and_disjoint() -> None:
    train = generate_examples(split="train", count=24, seed=7)
    evaluation = generate_examples(split="eval", count=12, seed=7)

    assert len(train) == 24
    assert len(evaluation) == 12
    assert train == generate_examples(split="train", count=24, seed=7)
    assert {row["ticket"] for row in train}.isdisjoint(
        {row["ticket"] for row in evaluation}
    )


def test_write_jsonl_writes_one_object_per_line(tmp_path) -> None:
    rows = generate_examples(split="train", count=3, seed=1)
    output = tmp_path / "rows.jsonl"

    write_jsonl(output, rows)

    loaded = [json.loads(line) for line in output.read_text().splitlines()]
    assert loaded == rows
