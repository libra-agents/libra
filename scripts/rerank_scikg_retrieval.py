#!/usr/bin/env python3
"""Rerank SciToolKG's fixed Top-20 with Qwen3-Reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_retrieval.clients import VllmQwenRerankerClient
from retrieval.data import load_tool_registry, qts_document
from retrieval.metrics import evaluate_retrieval
from scripts.rerank_retrieval_baselines import rerank_rows


KS = (1, 3, 5, 10, 15, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-stage", type=Path, required=True)
    parser.add_argument("--tool-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    source = json.loads(args.first_stage.read_text(encoding="utf-8"))
    rows = source["per_item"]
    registry = load_tool_registry(args.tool_registry)
    documents = {name: qts_document(tool) for name, tool in registry.items()}
    reranked = rerank_rows(
        rows,
        documents=documents,
        client=VllmQwenRerankerClient(base_url=args.base_url),
        output_path=args.output,
        concurrency=16,
        expected_depth=None,
    )
    metrics = evaluate_retrieval(
        [([str(item["name"]) for item in row["predicted"]], row["gold"])
         for row in reranked],
        ks=KS,
    )
    payload = {
        "dataset": "scitool",
        "method": "SciToolKG (Qwen-adapted)",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
        "candidate_depth": 20,
        "num_queries": len(rows),
        "metrics": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
