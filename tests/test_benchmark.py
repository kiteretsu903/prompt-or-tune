from fireworks_lab.benchmark import summarize_run


def test_summarize_run_computes_throughput_and_success_rate() -> None:
    rows = [
        {"prediction": {"category": "bug_report"}, "error": None, "latency_ms": 100},
        {"prediction": {"category": "bug_report"}, "error": None, "latency_ms": 300},
    ]

    summary = summarize_run(rows, concurrency=2, elapsed_s=0.5)

    assert summary["concurrency"] == 2
    assert summary["throughput_rps"] == 4.0
    assert summary["success_rate"] == 1.0
    assert summary["p50_latency_ms"] == 200.0
