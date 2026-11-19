"""Clients for PEASEC, TUDaGPT and offline mock responses."""

import json
import os
import time
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT, project_path

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

API_LOG = project_path("experiments/results/api_log.jsonl")


class ModelUnavailable(Exception):
    """The requested model is unavailable."""


class BadRequest(Exception):
    """The request was rejected; retrying the same payload is not useful."""


class TruncatedResponse(Exception):
    """The response reached its token limit."""


class BaseClient:
    """Shared requests, retry handling and call metadata."""

    def __init__(self, models, temperature=1.0, timeout=180,
                 allow_fallback=True, max_tokens=None,
                 url_env=None, token_env=None):
        load_dotenv(PROJECT_ROOT / ".env")
        self.models = models if isinstance(models, list) else [models]
        self.temperature = temperature
        self.timeout = timeout
        self.allow_fallback = allow_fallback
        self.max_tokens = max_tokens
        self.base_url = os.environ[url_env]
        self.token = os.environ[token_env]
        self.active_model = None

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _build_body(self, model: str, messages: list[dict]) -> dict:
        raise NotImplementedError

    def _extract_text(self, data: dict) -> str:
        raise NotImplementedError

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _post(self, body: dict) -> dict:
        """Retry network errors, rate limits and server errors."""
        response = requests.post(
            self._endpoint(),
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Auth error {response.status_code}: check token / API access")

        if (response.status_code >= 400
                and "not available" in response.text.lower()):
            try:
                msg = response.json().get("message", response.text[:200])
            except ValueError:
                msg = response.text[:200]
            raise ModelUnavailable(msg)

        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise BadRequest(f"{response.status_code}: {response.text[:300]}")

        response.raise_for_status()
        return response.json()

    def _log_call(self, model, prompt_chars, response_chars, latency):
        API_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(API_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "provider": self.provider,
                "model": model,
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "latency_s": round(latency, 2),
            }) + "\n")

    def complete(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "text": "..."}]"""
        errors = []
        for model in self.models:
            body = self._build_body(model, messages)
            try:
                start = time.time()
                data = self._post(body)
                latency = time.time() - start
            except (ModelUnavailable, BadRequest, TruncatedResponse,
                    requests.exceptions.RequestException) as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Model {model} failed and fallback is disabled: {e}"
                    ) from e
                print(f"  [{model}] failed, falling back to next model")
                continue

            self.active_model = model
            text = self._extract_text(data)
            self._log_call(model,
                           sum(len(m["text"]) for m in messages),
                           len(text), latency)
            return text

        raise RuntimeError("All models failed:\n" + "\n".join(errors))


class TUDaGPTClient(BaseClient):
    """Use the TUDaGPT payload and response format."""

    provider = "tudagpt"

    def __init__(self, models, url_env="TUDAGPT_URL",
                 token_env="TUDAGPT_TOKEN", **kwargs):
        super().__init__(models, url_env=url_env, token_env=token_env, **kwargs)

    def _endpoint(self) -> str:
        return self.base_url

    def _build_body(self, model, messages):

        return {"payload": {
            "model": model,
            "temperature": self.temperature,
            "messages": [{"role": m["role"], "content": {"text": m["text"]}}
                         for m in messages],
        }}

    def _extract_text(self, data):
        return data["content"]["text"]


class PEASECClient(BaseClient):
    """Use an OpenAI-style chat-completions endpoint."""

    provider = "peasec"

    def __init__(self, models, url_env="PEASEC_URL",
                 token_env="PEASEC_TOKEN", **kwargs):
        super().__init__(models, url_env=url_env, token_env=token_env, **kwargs)

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _build_body(self, model, messages):
        body = {
            "model": model,
            "temperature": self.temperature,
            "messages": [{"role": m["role"], "content": m["text"]}
                         for m in messages],
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return body

    def _extract_text(self, data):
        choice = data["choices"][0]

        if choice.get("finish_reason") == "length":
            raise TruncatedResponse(
                f"hit max_tokens={self.max_tokens}; raise it in config")
        return choice["message"]["content"]

    def list_models(self) -> list[str]:
        """List models available from the provider."""
        r = requests.get(self.base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {self.token}"},
                         timeout=30)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))


class MockClient:
    """Return fixed synthetic JSON without making model requests."""

    provider = "mock"

    def __init__(self, model="mock", temperature=1.0, broken=False, **kwargs):
        self.model = model
        self.active_model = model
        self.temperature = temperature

        self.broken = broken
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def complete(self, messages: list[dict]) -> str:
        self._calls += 1
        if self.broken and self._calls == 1:
            return "Sure! Here is your answer: {hate: true"
        return json.dumps({
            "reasoning": "The text attacks a protected group.",
            "hate": True,
            "target_group": ["gender"],
            "hate_type": ["explicit"],
            "severity": "medium",
        })

CLIENTS = {"tudagpt": TUDaGPTClient, "peasec": PEASECClient, "mock": MockClient}


def make_client(provider: str, models=None, **kwargs):
    """Construct the requested provider client."""
    if provider not in CLIENTS:
        raise ValueError(f"unknown provider '{provider}'; "
                         f"known: {sorted(CLIENTS)}")
    if provider == "mock":
        return MockClient(**kwargs)
    return CLIENTS[provider](models, **kwargs)
