#!/usr/bin/env python3
"""Unified Qwen3 embedding pipeline: tool_registry.json → .npz.

Supports any benchmark whose tools are in the unified format:
    {name, category, function, signature, docstring}

The endpoint is a local vLLM pooling server; no API credential is used.

Usage:
    export EMBEDDING_BASE_URL=http://127.0.0.1:8101/v1
    export EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

    python -m embedding.embed_tools \\
        --tools "dataset and embedding/scitool_clean/tool_registry.json" \\
        --output "dataset and embedding/scitool_clean/tool_embeddings.npz"

    # Embed benchmark queries
    python -m embedding.embed_tools \\
        --queries "dataset and embedding/scitool_clean/benchmark.jsonl" \\
        --output "dataset and embedding/scitool_clean/query_embeddings.npz"

    # Embed public query segments for a multi-turn adapter
    python -m embedding.embed_tools \\
        --query-segments "dataset and embedding/bfcl/benchmark.jsonl" \\
        --output "dataset and embedding/bfcl/query_segment_embeddings.npz"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_VERSION = "qwen3_embedding_0.6b_v1"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a user task, retrieve tools that are necessary and appropriate "
    "to solve the task."
)


def qwen3_query_text(text: str) -> str:
    return f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery: {text}"


# ---------------------------------------------------------------------------
# Embedding call through a local vLLM /v1/embeddings endpoint
# ---------------------------------------------------------------------------

def _embed_one_batch(
    batch: list[str],
    url: str,
    headers: dict[str, str],
    model: str,
) -> list[list[float]]:
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"input": batch, "model": model},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return []


def embed_texts(
    texts: list[str],
    *,
    base_url: str | None = None,
    model: str = EMBEDDING_MODEL,
    batch_size: int = 16,
    max_concurrent: int = 1,
) -> np.ndarray:
    """Call the local Qwen3 vLLM endpoint and return an ``(N, D)`` array."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base_url = (
        base_url or os.getenv("EMBEDDING_BASE_URL")
        or "http://127.0.0.1:8101/v1"
    ).rstrip("/")
    model = os.getenv("EMBEDDING_MODEL") or model
    if model != EMBEDDING_MODEL:
        raise ValueError(f"embedding model must be {EMBEDDING_MODEL}")
    url = f"{base_url}/embeddings"
    headers = {"Content-Type": "application/json"}

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results: list[list[float] | None] = [None] * len(batches)
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_to_idx = {
            pool.submit(_embed_one_batch, batch, url, headers, model): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done_count += 1
            done_texts = min((idx + 1) * batch_size, len(texts))
            if done_count % 50 == 0 or done_count == len(batches):
                print(f"  embedded {done_count * batch_size}/{len(texts)} "
                      f"({done_count}/{len(batches)} batches)")

    all_embeddings: list[list[float]] = []
    for r in results:
        all_embeddings.extend(r)
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tool embedding text construction
# ---------------------------------------------------------------------------

EMBED_TOOLS_VERSION = EMBEDDING_VERSION


def _embedding_model(model: str | None) -> str:
    return os.getenv("EMBEDDING_MODEL") or model or EMBEDDING_MODEL


def tool_to_text(tool: dict, *, text_mode: str = "full-doc") -> str:
    """Convert a unified tool dict to text for embedding.

    ``full-doc`` uses the name, signature, and full docstring. ``first-line``
    uses only the first docstring line.
    """
    name = str(tool["name"])
    function = str(tool.get("function") or name)
    category = str(tool.get("category") or "")
    sig = str(tool.get("signature") or "")
    doc = str(tool.get("docstring") or "").strip()
    if text_mode == "first-line":
        first_line = doc.split("\n")[0].strip() if doc else ""
        return f"{name}{sig}: {first_line}" if first_line else f"{name}{sig}"
    if text_mode == "full-doc":
        parts = [
            f"Tool: {name}",
            f"Function: {function}",
            f"Category: {category}",
            f"Signature: {function}{sig}",
        ]
        if doc:
            parts.append(f"Docstring:\n{doc}")
        return "\n".join(parts)
    raise ValueError(f"Unknown tool text mode: {text_mode}")


def _cache_matches(
    data: np.lib.npyio.NpzFile,
    *,
    item_key: str,
    expected_items: list[str],
    text_mode: str,
) -> bool:
    if item_key not in data or "embeddings" not in data:
        return False
    cached_items = list(map(str, data[item_key]))
    embeddings = data["embeddings"]
    if cached_items != expected_items:
        return False
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_items):
        return False
    if not np.isfinite(embeddings).all():
        return False
    if "version" not in data:
        return False
    if str(data["version"]) != EMBED_TOOLS_VERSION:
        return False
    if "text_mode" in data and str(data["text_mode"]) != text_mode:
        return False
    return True


# ---------------------------------------------------------------------------
# Embed tools
# ---------------------------------------------------------------------------

def embed_tool_registry(
    tools_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
    text_mode: str = "full-doc",
    **embed_kwargs,
) -> tuple[list[str], np.ndarray]:
    """Embed a tool_registry.json and save to .npz."""
    tools_path = Path(tools_path)
    output_path = Path(output_path)

    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    names = [t["name"] for t in tools]
    texts = [tool_to_text(t, text_mode=text_mode) for t in tools]
    model = _embedding_model(embed_kwargs.get("model"))

    # Check cache
    if not force and output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        if _cache_matches(
            data,
            item_key="names",
            expected_items=names,
            text_mode=text_mode,
        ):
            embeddings = data["embeddings"]
            print(f"Using cached embeddings: {output_path} ({len(names)} tools)")
            return names, embeddings

    # Embed
    print(f"Embedding {len(texts)} tools from {tools_path} ({text_mode}, {model})...")
    embeddings = embed_texts(texts, **embed_kwargs)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        names=np.array(names),
        embeddings=embeddings,
        version=np.array(EMBED_TOOLS_VERSION),
        model=np.array(model),
        text_mode=np.array(text_mode),
        source_path=np.array(str(tools_path)),
    )
    print(f"Saved embeddings: {output_path} shape={embeddings.shape}")
    return names, embeddings


# ---------------------------------------------------------------------------
# Embed queries
# ---------------------------------------------------------------------------

def embed_queries(
    queries_path: str | Path,
    output_path: str | Path,
    *,
    question_key: str = "question",
    force: bool = False,
    **embed_kwargs,
) -> tuple[list[str], np.ndarray]:
    """Embed benchmark queries from a .jsonl file and save to .npz."""
    queries_path = Path(queries_path)
    output_path = Path(output_path)

    questions = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item[question_key])
    model = _embedding_model(embed_kwargs.get("model"))

    # Check cache
    if not force and output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        if _cache_matches(
            data,
            item_key="questions",
            expected_items=questions,
            text_mode=f"qwen3-instruction-query:{question_key}",
        ):
            embeddings = data["embeddings"]
            print(f"Using cached query embeddings: {output_path} ({len(questions)} queries)")
            return questions, embeddings

    print(f"Embedding {len(questions)} queries from {queries_path} ({model})...")
    embeddings = embed_texts(
        [qwen3_query_text(question) for question in questions], **embed_kwargs
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        questions=np.array(questions),
        embeddings=embeddings,
        version=np.array(EMBED_TOOLS_VERSION),
        model=np.array(model),
        text_mode=np.array(f"qwen3-instruction-query:{question_key}"),
        source_path=np.array(str(queries_path)),
    )
    print(f"Saved query embeddings: {output_path} shape={embeddings.shape}")
    return questions, embeddings


def embed_query_segments(
    queries_path: str | Path,
    output_path: str | Path,
    *,
    question_key: str = "question",
    force: bool = False,
    **embed_kwargs,
) -> tuple[list[str], np.ndarray]:
    """Embed each nonempty public line while retaining its parent query."""

    queries_path = Path(queries_path)
    output_path = Path(output_path)
    questions: list[str] = []
    segments: list[str] = []
    query_indices: list[int] = []
    with queries_path.open(encoding="utf-8") as source:
        for query_index, line in enumerate(source):
            question = str(json.loads(line)[question_key])
            questions.append(question)
            parts = [part.strip() for part in question.splitlines() if part.strip()]
            if not parts:
                raise ValueError(f"query {query_index} has no nonempty segment")
            segments.extend(parts)
            query_indices.extend([query_index] * len(parts))

    if not force and output_path.exists():
        data = np.load(output_path, allow_pickle=True)
        if (
            list(map(str, data.get("questions", ()))) == questions
            and list(map(str, data.get("segments", ()))) == segments
            and list(map(int, data.get("query_indices", ()))) == query_indices
            and "embeddings" in data
            and data["embeddings"].shape[0] == len(segments)
        ):
            print(f"Using cached segment embeddings: {output_path}")
            return segments, data["embeddings"]

    model = _embedding_model(embed_kwargs.get("model"))
    print(f"Embedding {len(segments)} query segments from {queries_path} ({model})...")
    embeddings = embed_texts(
        [qwen3_query_text(segment) for segment in segments], **embed_kwargs
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        questions=np.asarray(questions),
        segments=np.asarray(segments),
        query_indices=np.asarray(query_indices, dtype=np.int32),
        embeddings=embeddings,
        version=np.asarray(EMBED_TOOLS_VERSION),
        model=np.asarray(model),
        text_mode=np.asarray("qwen3-instruction-query:nonempty-public-lines"),
        source_path=np.asarray(str(queries_path)),
    )
    print(f"Saved segment embeddings: {output_path} shape={embeddings.shape}")
    return segments, embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed tools or queries to .npz")
    parser.add_argument("--tools", help="Path to tool_registry.json to embed")
    parser.add_argument("--queries", help="Path to benchmark.jsonl to embed queries")
    parser.add_argument(
        "--query-segments",
        help="Path to benchmark.jsonl whose nonempty public lines are segments",
    )
    parser.add_argument("-o", "--output", required=True, help="Output .npz path")
    parser.add_argument("--force", action="store_true", help="Force re-embed even if cache exists")
    parser.add_argument("--model", default=None, help="Override embedding model name")
    parser.add_argument("--batch-size", type=int, default=16, help="Texts per embedding request.")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Concurrent embedding requests.")
    parser.add_argument("--tool-text-mode", choices=["full-doc", "first-line"], default="full-doc",
                        help="Text template for tool embedding.")
    args = parser.parse_args()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    kwargs["batch_size"] = args.batch_size
    kwargs["max_concurrent"] = args.max_concurrent

    if args.tools:
        embed_tool_registry(
            args.tools, args.output, force=args.force,
            text_mode=args.tool_text_mode, **kwargs,
        )
    elif args.queries:
        embed_queries(args.queries, args.output, force=args.force, **kwargs)
    elif args.query_segments:
        embed_query_segments(
            args.query_segments, args.output, force=args.force, **kwargs
        )
    else:
        parser.error("Must specify --tools, --queries, or --query-segments")


if __name__ == "__main__":
    main()
