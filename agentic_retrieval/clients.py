"""HTTP clients for local vLLM embedding, reranking, and chat services."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import requests

from retrieval.data import (
    DEFAULT_QUERY_INSTRUCTION,
    qwen3_query_text,
)


class ModelOutputError(ValueError):
    """The chat model returned a syntactically unusable agent action."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts returned by an OpenAI-compatible generation endpoint."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ChatCompletion:
    """One model response together with server-reported usage."""

    content: str
    finish_reason: str
    usage: TokenUsage


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"Model server request failed: HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return response.json()


@dataclass
class OpenAIEmbeddingClient:
    """OpenAI-compatible embedding client used for agent-generated queries."""

    base_url: str
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    instruction: str = DEFAULT_QUERY_INSTRUCTION
    timeout: int = 120

    def embed(self, text: str) -> np.ndarray:
        query = qwen3_query_text(text, self.instruction)
        body = _post_json(
            f"{self.base_url.rstrip('/')}/v1/embeddings",
            {"model": self.model, "input": [query]},
            self.timeout,
        )
        rows = sorted(body["data"], key=lambda row: int(row["index"]))
        if len(rows) != 1:
            raise ValueError(f"Embedding server returned {len(rows)} vectors for one query.")
        vector = np.asarray(rows[0]["embedding"], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Invalid query embedding returned with shape {vector.shape}.")
        return vector


@dataclass
class VllmQwenRerankerClient:
    """Qwen3-Reranker scoring through vLLM's pooling rerank API."""

    base_url: str
    model: str = "Qwen/Qwen3-Reranker-0.6B"
    timeout: int = 300

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        body = _post_json(
            f"{self.base_url.rstrip('/')}/rerank",
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            self.timeout,
        )
        scores: list[float | None] = [None] * len(documents)
        for result in body["results"]:
            scores[int(result["index"])] = float(result["relevance_score"])
        if any(score is None for score in scores):
            raise ValueError("vLLM reranker omitted one or more document scores.")
        return [float(score) for score in scores]


@dataclass
class OpenAIChatClient:
    """OpenAI-compatible chat client using LlamaIndex LlamaCPP defaults."""

    base_url: str
    model: str
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 256
    timeout: int = 180

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
    ) -> ChatCompletion:
        request_messages = messages
        if self.model == "nvidia/NVIDIA-Nemotron-Nano-9B-v2":
            request_messages = [message.copy() for message in messages]
            if request_messages and request_messages[0]["role"] == "system":
                request_messages[0]["content"] = (
                    "/no_think\n" + request_messages[0]["content"]
                )
            else:
                request_messages.insert(
                    0,
                    {"role": "system", "content": "/no_think"},
                )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        if self.model.startswith(("Qwen/", "google/gemma-4-")):
            request["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.model.startswith("ibm-granite/"):
            request["chat_template_kwargs"] = {"thinking": False}
        elif self.model == "openai/gpt-oss-20b":
            request["reasoning_effort"] = "low"
        body = _post_json(
            f"{self.base_url.rstrip('/')}/v1/chat/completions",
            request,
            self.timeout,
        )
        choice = body["choices"][0]
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelOutputError("Chat model returned an empty action.")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise ModelOutputError(
                f"Chat response ended with finish_reason={finish_reason!r}; "
                f"content_length={len(content)}."
            )
        raw_usage = body.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
        )
        if usage.total_tokens == 0:
            usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.prompt_tokens + usage.completion_tokens,
            )
        return ChatCompletion(
            content=content,
            finish_reason=str(finish_reason),
            usage=usage,
        )
