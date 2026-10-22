from fireworks_lab.failure_analysis import find_failures


def test_find_failures_returns_mismatched_fields() -> None:
    rows = [
        {
            "id": "eval-1",
            "ticket": "Duplicate $900 charge.",
            "expected": {
                "category": "billing_refund",
                "priority": "P1",
                "escalate": True,
                "reason_code": "duplicate_charge_over_500",
            },
            "prediction": {
                "category": "billing_refund",
                "priority": "P2",
                "escalate": False,
                "reason_code": "ordinary_refund",
            },
            "error": None,
        }
    ]

    failures = find_failures(rows)

    assert failures[0]["mismatched_fields"] == [
        "priority",
        "escalate",
        "reason_code",
    ]
