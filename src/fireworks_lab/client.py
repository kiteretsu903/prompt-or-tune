"""Thin asynchronous Fireworks client with bounded retries and structured output."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIStatusError, AsyncOpenAI, OpenAIError

from fireworks_lab.prompts import PromptVersion, build_messages
from fireworks_lab.schemas import Prediction

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ClassificationResult:
    prediction: dict[str, Any] | None
    latency_ms: float
    usage: dict[str, int]
    raw_output: str | None
    error: str | None
    attempts: list[dict[str, Any]] = field(default_factory=list)


async def classify_with_retries(client, *, request, schema, max_attempts):
    """Own the HTTP retry budget and retain diagnostics for every attempt.

    SDK retries must be disabled on the supplied client. Usage is unknown for
    failed HTTP requests; reported token totals include only observed usage.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    started = time.perf_counter()
    attempts = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    prediction = None
    for number in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        raw_output = None
        attempt_usage = None
        status_code = None
        error = None
        retryable = False
        try:
            response = await client.chat.completions.create(**request)
            status_code = 200
            raw_output = response.choices[0].message.content or ""
            if response.usage is not None:
                attempt_usage = {
                    "prompt_tokens": int(response.usage.prompt_tokens or 0),
                    "completion_tokens": int(response.usage.completion_tokens or 0),
                }
                for key in usage:
                    usage[key] += attempt_usage[key]
            prediction = schema.model_validate(json.loads(raw_output)).model_dump(
                mode="json"
            )
        except APIStatusError as exc:
            status_code = exc.status_code
            raw_output = exc.response.text
            error = f"{type(exc).__name__}: {exc}"
            retryable = status_code in RETRYABLE_STATUS_CODES
        except (json.JSONDecodeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except OpenAIError as exc:
            error = f"{type(exc).__name__}: {exc}"
            retryable = True
        attempts.append(
            {
                "attempt": number,
                "status_code": status_code,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000, 2),
                "raw_output": raw_output,
                "usage": attempt_usage,
                "error": error,
            }
        )
        if error is None or not retryable or number == max_attempts:
            break
        delay = (
            min(30.0, 1.5 * (2 ** (number - 1))) + random.random()
            if status_code is not None
            else min(10.0, 2 ** (number - 1))
        )
        await asyncio.sleep(delay)
    return ClassificationResult(
        prediction=prediction,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        usage=usage,
        raw_output=raw_output,
        error=error,
        attempts=attempts,
    )


class FireworksClassifier:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    async def classify(
        self,
        ticket: str,
        prompt_version: PromptVersion,
        *,
        max_attempts: int = 8,
    ) -> ClassificationResult:
        return await classify_with_retries(
            self.client,
            request={
                "model": self.model,
                "messages": build_messages(ticket, prompt_version),
                "temperature": 0,
                "max_tokens": 120,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "support_decision",
                        "schema": Prediction.model_json_schema(),
                    },
                },
            },
            schema=Prediction,
            max_attempts=max_attempts,
        )


def classifier_from_env(*, model_override: str | None = None) -> FireworksClassifier:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is not set")
    model = model_override or os.environ.get("FIREWORKS_MODEL")
    if not model:
        raise RuntimeError("FIREWORKS_MODEL is not set")
    return FireworksClassifier(
        api_key=api_key,
        base_url=os.environ.get(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        ),
        model=model,
    )
