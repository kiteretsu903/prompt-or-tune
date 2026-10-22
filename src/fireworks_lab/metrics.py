"""Deterministic evaluation metrics for structured ticket decisions."""

from __future__ import annotations

import math
from typing import Any

FIELDS = ("category", "priority", "escalate", "reason_code")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 2)


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    valid_rows = [row for row in rows if row.get("prediction") is not None]
    correct = {field: 0 for field in FIELDS}
    exact = 0

    for row in valid_rows:
        expected = row["expected"]
        prediction = row["prediction"]
        matches = []
        for field in FIELDS:
            match = prediction.get(field) == expected.get(field)
            matches.append(match)
            correct[field] += int(match)
        exact += int(all(matches))

    latencies = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None
    ]
    prompt_tokens = sum(
        int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in rows
    )
    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows
    )
    denominator = count or 1

    return {
        "count": count,
        "valid_output_rate": round(len(valid_rows) / denominator, 4),
        "category_accuracy": round(correct["category"] / denominator, 4),
        "priority_accuracy": round(correct["priority"] / denominator, 4),
        "escalation_accuracy": round(correct["escalate"] / denominator, 4),
        "reason_code_accuracy": round(correct["reason_code"] / denominator, 4),
        "exact_match_accuracy": round(exact / denominator, 4),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "errors": count - len(valid_rows),
    }
