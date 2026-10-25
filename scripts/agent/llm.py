#!/usr/bin/env python3
"""Small auditable client for the repository's LiteLLM-compatible proxy.

Use the cheap model for anything mechanical and at volume. Reserve the expensive
one for tasks where reasoning depth actually changes the answer. A thousand
title parses is the first kind. Deciding what an ambiguous clause asserts is the
second.

    from llm import ask, ask_many
    ask("Return only the regnal year as an integer.", "year 12 of king Alexander")
    ask_many(prompt, ["title one", "title two", ...], workers=8)

Reads LITELLM_API_KEY from ~/.env and defaults to OSU's proxy.

Nothing is cached, so a caller repeating work pays for it twice.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Model names as the OSU proxy lists them at /v1/models. Query that endpoint if
# these ever stop resolving; the aliases change as versions are added.
CHEAP = "claude-sonnet-5"
DEEP = "claude-opus-5"
# OSU's hosted LiteLLM proxy. Override with LITELLM_BASE_URL in the environment
# or in ~/.env. The proxy speaks the OpenAI dialect at /v1/chat/completions and
# the Anthropic one at /v1/messages; this client uses the former.
BASE = os.environ.get("LITELLM_BASE_URL", "https://litellm.cloud.osu.edu") + "/v1"


def _base() -> str:
    return BASE.rstrip("/")


def _key() -> str:
    """The API key, from the environment or from a .env file.

    A .env at the repository root is looked at before the one in the home
    directory, so a clone works on a machine that has never seen this project.
    Neither is committed.
    """
    if os.environ.get("LITELLM_API_KEY"):
        return os.environ["LITELLM_API_KEY"]
    for env in (Path(__file__).resolve().parents[2] / ".env",
                Path.home() / ".env"):
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            if line.startswith("LITELLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "LITELLM_API_KEY not found. Set it in the environment, or put "
        "LITELLM_API_KEY=... in a .env at the repository root or in your home "
        "directory. Only commands that explicitly request LiteLLM need it.")


@dataclass(frozen=True)
class Completion:
    """Assistant content plus the provider metadata needed for an audit."""

    content: str
    model: str
    finish_reason: str | None
    usage: dict[str, Any]
    response_cost_usd: float | None


class EmptyCompletionError(RuntimeError):
    """Provider consumed tokens but returned no usable assistant content."""

    def __init__(self, message: str, usage: dict[str, Any],
                 finish_reason: str | None):
        super().__init__(message)
        self.usage = usage
        self.finish_reason = finish_reason


def complete(system: str, user: str, model: str = CHEAP,
             max_tokens: int = 1024, *,
             response_format: dict[str, Any] | None = None,
             effort: str | None = None) -> Completion:
    """Make one request and retain token/cost/finish metadata."""
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        # This proxy's Claude route expects system at the top level and rejects
        # temperature, so neither is represented as an OpenAI-style message.
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_base()}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    choice = out["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if not content:
        fields = {key: (type(value).__name__,
                        len(value) if hasattr(value, "__len__") else None)
                  for key, value in message.items()}
        raise EmptyCompletionError(
            "LiteLLM returned empty content; "
            f"finish_reason={choice.get('finish_reason')!r}, "
            f"usage={out.get('usage', {})!r}, message_fields={fields}",
            out.get("usage") or {}, choice.get("finish_reason"))
    hidden = out.get("_hidden_params") or {}
    cost = out.get("response_cost")
    if cost is None:
        cost = hidden.get("response_cost")
    return Completion(
        content=content,
        model=str(out.get("model", model)),
        finish_reason=choice.get("finish_reason"),
        usage=out.get("usage") or {},
        response_cost_usd=float(cost) if cost is not None else None,
    )


def ask(system: str, user: str, model: str = CHEAP, max_tokens: int = 1024) -> str:
    """One call. Returns the assistant text, or raises."""
    return complete(system, user, model=model, max_tokens=max_tokens).content


def ask_many(system: str, users: list[str], model: str = CHEAP, workers: int = 8,
             **kw) -> list[str | None]:
    """Same prompt over many inputs, in parallel. Failures come back as None."""
    def one(u):
        try:
            return ask(system, u, model=model, **kw)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                RuntimeError, TimeoutError):
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, users))


if __name__ == "__main__":
    # Connectivity check. A 400 here usually means the model alias is wrong
    # rather than the URL; GET /v1/models lists what the proxy accepts.
    try:
        print(ask("Reply with the single word: ok", "ping"))
    except Exception as exc:                                # noqa: BLE001
        print(f"failed: {exc}\n\nSet LITELLM_BASE_URL to the proxy address. "
              f"Currently {BASE!r}.")
