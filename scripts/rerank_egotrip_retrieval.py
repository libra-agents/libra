#!/usr/bin/env python3
"""Rerank Ego-Trip candidates and evaluate the ranked tool list."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agentic_retrieval.clients import VllmQwenRerankerClient
from retrieval.data import load_benchmark, load_tool_registry, qts_document
from retrieval.metrics import evaluate_retrieval


KS = (1, 3, 5, 10, 15, 20)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rerank_row(
    row: dict[str, Any],
    *,
    documents: dict[str, str],
    client: VllmQwenRerankerClient,
    top_k: int,
) -> dict[str, Any]:
    candidates = [str(item["name"]) for item in row["predicted"]]
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"{row['id']}: duplicate Ego-Trip candidates")
    missing = sorted(set(candidates) - set(documents))
    if missing:
        raise ValueError(f"{row['id']}: graph candidates absent from registry: {missing[:5]}")
    scores = client.score(str(row["question"]), [documents[name] for name in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0]))
    return {
        **row,
        "predicted": [
            {"name": name, "rerank_score": float(score)}
            for name, score in ranked[:top_k]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8102")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.top_k < 1 or args.concurrency < 1:
        raise SystemExit("--top-k and --concurrency must be positive")

    benchmark = load_benchmark(args.data_dir / "benchmark.jsonl")
    registry = load_tool_registry(args.data_dir / "tool_registry.json")
    documents = {name: qts_document(tool) for name, tool in registry.items()}
    candidates = load_jsonl(args.candidates)
    expected_ids = [str(row["id"]) for row in benchmark]
    if [str(row["id"]) for row in candidates] != expected_ids:
        raise ValueError("Candidate rows are not aligned with the canonical benchmark")

    completed = (
        {str(row["id"]): row for row in load_jsonl(args.output)}
        if args.output.exists()
        else {}
    )
    pending = [row for row in candidates if str(row["id"]) not in completed]
    client = VllmQwenRerankerClient(base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    rerank_row,
                    row,
                    documents=documents,
                    client=client,
                    top_k=args.top_k,
                ): str(row["id"])
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                completed[str(row["id"])] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"reranked {index}/{len(pending)} pending queries", flush=True)

    rows = [completed[row_id] for row_id in expected_ids]
    metrics = evaluate_retrieval(
        [
            (
                [str(item["name"]) for item in row["predicted"]],
                [str(name) for name in row["gold"]],
            )
            for row in rows
        ],
        ks=KS,
    )
    candidate_complete = sum(
        set(str(name) for name in row["gold"]).issubset(
            set(str(item["name"]) for item in row["predicted"])
        )
        for row in rows
    ) / len(rows)
    payload = {
        "method": "Ego-Trip (adapted)",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
        "semantic_entry_k": 10,
        "expansion_hops": 1,
        "top_k": args.top_k,
        "num_queries": len(rows),
        "candidate_depth": args.top_k,
        "candidate_generation": "Ego-Trip one-hop expansion",
        "candidate_pruning": "QTS Top-20",
        "candidate_set_complete_recall": candidate_complete,
        "metrics": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
