"""Compare base and tuned BANKING77 reports against frozen success gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireworks_lab.banking77 import assess_banking_improvement


def validate_comparison(base, tuned):
    for report in (base, tuned):
        provenance = report.get("comparison_provenance")
        if not isinstance(provenance, dict) or provenance.get("version") != 1:
            raise ValueError(
                "Missing comparison provenance; legacy scores alone cannot establish a controlled comparison"
            )
        config = provenance.get("inference_config")
        required = {
            "base_model",
            "prompt_sha256",
            "schema_sha256",
            "temperature",
            "reasoning_effort",
            "max_tokens",
        }
        if (
            not provenance.get("dataset_sha256")
            or not isinstance(config, dict)
            or not required.issubset(config)
            or any(config[key] in (None, "") for key in required)
            or not isinstance(report.get("count"), int)
            or report["count"] <= 0
        ):
            raise ValueError("Incomplete comparison provenance")
    if base["count"] != tuned["count"] or (
        base["comparison_provenance"] != tuned["comparison_provenance"]
    ):
        raise ValueError(
            "Mismatched comparison provenance: dataset, prompt, base model, schema, and generation settings must match"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("tuned", type=Path)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    tuned = json.loads(args.tuned.read_text(encoding="utf-8"))
    validate_comparison(base, tuned)
    assessment = assess_banking_improvement(base, tuned)

    print("# BANKING77 Fine-Tuning Comparison\n")
    print("| System | Valid output | Intent accuracy | Macro accuracy | p50 | p95 |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, report in ((args.base.stem, base), (args.tuned.stem, tuned)):
        print(
            f"| {name} | {report['valid_output_rate']:.1%} | "
            f"{report['intent_accuracy']:.1%} | "
            f"{report['macro_intent_accuracy']:.1%} | "
            f"{report['p50_latency_ms']} ms | {report['p95_latency_ms']} ms |"
        )
    print()
    print(f"Accuracy delta: {assessment['accuracy_delta']:+.1%}")
    print(f"Success gate: {'PASS' if assessment['passed'] else 'FAIL'}")
    if assessment["failed_gates"]:
        print("Failed gates: " + ", ".join(assessment["failed_gates"]))


if __name__ == "__main__":
    main()
