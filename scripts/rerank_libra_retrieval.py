#!/usr/bin/env python3
"""Rerank frozen LIBRA Top-20 predictions with Qwen3-Reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.data import load_benchmark, load_tool_registry, qts_document
from retrieval.metrics import evaluate_retrieval
from scripts.rerank_retrieval_baselines import (
    load_jsonl,
    predicted_name,
    rerank_rows,
)
from scripts.run_retrieval_baselines import DATASETS, dataset_directory


KS = (1, 3, 5, 10, 15, 20)
CANDIDATE_DEPTH = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8102")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")

    from agentic_retrieval.clients import VllmQwenRerankerClient

    client = VllmQwenRerankerClient(base_url=args.base_url)
    data_root = args.data_root or args.repo_root / "data"
    for dataset in DATASETS:
        data_dir = dataset_directory(args.repo_root, data_root, dataset)
        benchmark = load_benchmark(data_dir / "benchmark.jsonl")
        registry = load_tool_registry(data_dir / "tool_registry.json")
        rows = load_jsonl(args.prediction_root / f"{dataset}-predictions.jsonl")
        expected_ids = [str(row["id"]) for row in benchmark]
        if [str(row["id"]) for row in rows] != expected_ids:
            raise ValueError(f"{dataset}: predictions are not benchmark-aligned")
        if any(len(row["predicted"]) != CANDIDATE_DEPTH for row in rows):
            raise ValueError(f"{dataset}: expected fixed Top-{CANDIDATE_DEPTH} input")

        documents = {
            name: qts_document(tool)
            for name, tool in registry.items()
        }
        output_dir = args.output_root / dataset
        reranked = rerank_rows(
            rows,
            documents=documents,
            client=client,
            output_path=output_dir / "predictions.jsonl",
            concurrency=args.concurrency,
        )
        metrics = evaluate_retrieval(
            [
                (
                    [predicted_name(item) for item in row["predicted"]],
                    [str(name) for name in row["gold"]],
                )
                for row in reranked
            ],
            ks=KS,
        )
        payload = {
            "dataset": dataset,
            "method": "LIBRA",
            "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
            "candidate_depth": CANDIDATE_DEPTH,
            "ks": list(KS),
            "num_queries": len(reranked),
            "metrics": metrics,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
