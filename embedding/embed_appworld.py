#!/usr/bin/env python3
"""Embed AppWorld tools with their short public description."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from embedding.embed_tools import (
    EMBEDDING_MODEL,
    EMBEDDING_VERSION,
    embed_texts,
    qwen3_query_text,
)


def tool_text(tool: dict) -> str:
    """Adapt AppWorld's public ``API:`` line to the shared short document."""

    descriptions = [
        line.removeprefix("API:").strip()
        for line in str(tool.get("docstring") or "").splitlines()
        if line.startswith("API:")
    ]
    if len(descriptions) != 1:
        raise ValueError(
            f"expected one public API description for {tool.get('name')!r}"
        )
    return f"Tool: {tool['name']}\nDescription: {descriptions[0]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    benchmark = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = [str(tool["name"]) for tool in registry]
    questions = [str(row["question"]) for row in benchmark]
    model = os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL
    tool_vectors = embed_texts(
        [tool_text(tool) for tool in registry], batch_size=16, max_concurrent=1
    )
    query_vectors = embed_texts(
        [qwen3_query_text(question) for question in questions],
        batch_size=16,
        max_concurrent=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "tool_embeddings.npz",
        names=np.array(names),
        embeddings=tool_vectors,
        version=np.array(EMBEDDING_VERSION),
        model=np.array(model),
        text_mode=np.array("identity_plus_public_api_description"),
        source_path=np.array(str(args.registry)),
    )
    np.savez_compressed(
        args.output_dir / "query_embeddings.npz",
        questions=np.array(questions),
        embeddings=query_vectors,
        version=np.array(EMBEDDING_VERSION),
        model=np.array(model),
        text_mode=np.array("qwen3-instruction-query:question"),
        source_path=np.array(str(args.benchmark)),
    )
    print(json.dumps({
        "tools": len(names),
        "queries": len(questions),
        "dimensions": int(tool_vectors.shape[1]),
        "tool_text": "identity_plus_public_api_description",
        "query_text": "qwen3_instruction",
        "model": model,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
