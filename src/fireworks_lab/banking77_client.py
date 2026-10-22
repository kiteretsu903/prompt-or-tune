"""Asynchronous Fireworks client for the fixed BANKING77 intent contract."""

from __future__ import annotations

from openai import AsyncOpenAI

from fireworks_lab.banking77 import BankingPrediction, build_banking_messages
from fireworks_lab.client import ClassificationResult, classify_with_retries


class BankingClassifier:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        prompt_version: str = "original",
        base_model: str | None = None,
    ) -> None:
        self.model = model
        self.base_model = base_model
        build_banking_messages("", version=prompt_version)
        self.prompt_version = prompt_version
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def request(self, text):
        return {
            "model": self.model,
            "messages": build_banking_messages(text, version=self.prompt_version),
            "temperature": 0,
            "reasoning_effort": "none",
            "max_tokens": 40,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "banking77_intent",
                    "schema": BankingPrediction.model_json_schema(),
                },
            },
        }

    async def classify(
        self, text: str, *, max_attempts: int = 10
    ) -> ClassificationResult:
        return await classify_with_retries(
            self.client,
            request=self.request(text),
            schema=BankingPrediction,
            max_attempts=max_attempts,
        )
