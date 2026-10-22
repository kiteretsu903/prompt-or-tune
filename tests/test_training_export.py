import json

from fireworks_lab.training_export import to_sft_record


def test_to_sft_record_uses_openai_chat_jsonl_format() -> None:
    example = {
        "ticket": "I was charged $900 twice.",
        "expected": {
            "category": "billing_refund",
            "priority": "P1",
            "escalate": True,
            "reason_code": "duplicate_charge_over_500",
        },
    }

    record = to_sft_record(example)

    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assistant_payload = json.loads(record["messages"][2]["content"])
    assert assistant_payload == example["expected"]
