from fireworks_lab.metrics import evaluate_rows


def test_evaluate_rows_computes_field_and_exact_accuracy() -> None:
    rows = [
        {
            "expected": {
                "category": "billing_refund",
                "priority": "P1",
                "escalate": True,
                "reason_code": "duplicate_charge_over_500",
            },
            "prediction": {
                "category": "billing_refund",
                "priority": "P1",
                "escalate": True,
                "reason_code": "duplicate_charge_over_500",
            },
            "latency_ms": 100.0,
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            "error": None,
        },
        {
            "expected": {
                "category": "security_incident",
                "priority": "P0",
                "escalate": True,
                "reason_code": "credential_theft",
            },
            "prediction": {
                "category": "security_incident",
                "priority": "P1",
                "escalate": True,
                "reason_code": "credential_theft",
            },
            "latency_ms": 300.0,
            "usage": {"prompt_tokens": 60, "completion_tokens": 12},
            "error": None,
        },
    ]

    report = evaluate_rows(rows)

    assert report["count"] == 2
    assert report["valid_output_rate"] == 1.0
    assert report["category_accuracy"] == 1.0
    assert report["priority_accuracy"] == 0.5
    assert report["exact_match_accuracy"] == 0.5
    assert report["total_tokens"] == 132


def test_evaluate_rows_counts_invalid_outputs() -> None:
    rows = [
        {
            "expected": {
                "category": "feature_request",
                "priority": "P3",
                "escalate": False,
                "reason_code": "product_enhancement",
            },
            "prediction": None,
            "latency_ms": 50.0,
            "usage": {"prompt_tokens": 20, "completion_tokens": 0},
            "error": "invalid json",
        }
    ]

    report = evaluate_rows(rows)

    assert report["valid_output_rate"] == 0.0
    assert report["exact_match_accuracy"] == 0.0
