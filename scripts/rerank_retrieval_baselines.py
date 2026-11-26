#!/usr/bin/env python3
"""Rerank the five retrieval baselines with local Qwen3-Reranker."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agentic_retrieval.clients import VllmQwenRerankerClient
from retrieval.data import (
    load_benchmark,
    load_tool_registry,
    load_toolde_profiles,
    qts_document,
    toolde_document,
)
from retrieval.metrics import evaluate_retrieval
from scripts.run_retrieval_baselines import (
    DATASETS,
    KS,
    METHODS,
    TOP_K,
    dataset_directory,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def predicted_name(item: str | dict[str, Any]) -> str:
    return str(item["name"]) if isinstance(item, dict) else str(item)


def predicted_score(item: str | dict[str, Any]) -> float | None:
    if not isinstance(item, dict):
        return None
    value = item.get("score", item.get("first_stage_score"))
    return None if value is None else float(value)


def rerank_row(
    row: dict[str, Any],
    *,
    documents: dict[str, str],
    client: VllmQwenRerankerClient,
    expected_depth: int | None = TOP_K,
) -> dict[str, Any]:
    candidates = list(row["predicted"])
    names = [predicted_name(item) for item in candidates]
    valid_size = (
        len(names) == expected_depth
        if expected_depth is not None
        else 0 < len(names) <= TOP_K
    )
    if not valid_size or len(set(names)) != len(names):
        raise ValueError(
            f"{row['id']}: invalid candidate depth {len(names)}"
        )
    scores = client.score(
        str(row["question"]),
        [documents[name] for name in names],
    )
    ranked = sorted(
        zip(names, scores, map(predicted_score, candidates)),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        **row,
        "predicted": [
            {
                "name": name,
                "rerank_score": float(score),
                **(
                    {"first_stage_score": float(first_stage_score)}
                    if first_stage_score is not None
                    else {}
                ),
            }
            for name, score, first_stage_score in ranked
        ],
    }


def rerank_rows(
    rows: list[dict[str, Any]],
    *,
    documents: dict[str, str],
    client: VllmQwenRerankerClient,
    output_path: Path,
    concurrency: int,
    expected_depth: int | None = TOP_K,
) -> list[dict[str, Any]]:
    completed = (
        {str(row["id"]): row for row in load_jsonl(output_path)}
        if output_path.exists()
        else {}
    )
    pending = [row for row in rows if str(row["id"]) not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    rerank_row,
                    row,
                    documents=documents,
                    client=client,
                    expected_depth=expected_depth,
                ): str(row["id"])
                for row in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row_id = str(row["id"])
                completed[row_id] = row
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"reranked {index}/{len(pending)}", flush=True)
    return [completed[str(row["id"])] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8102")
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")

    client = VllmQwenRerankerClient(base_url=args.base_url)
    data_root = args.data_root or args.repo_root / "data"
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    methods = [args.method] if args.method else list(METHODS)
    for dataset in datasets:
        source_dir = dataset_directory(args.repo_root, data_root, dataset)
        registry = load_tool_registry(source_dir / "tool_registry.json")
        benchmark = load_benchmark(source_dir / "benchmark.jsonl")
        benchmark_ids = [str(row["id"]) for row in benchmark]
        profiles = load_toolde_profiles(source_dir / "toolde_profiles.jsonl")
        for method in methods:
            input_path = (
                args.result_root
                / "first_stage"
                / method
                / f"{dataset}_top20.jsonl"
            )
            rows = load_jsonl(input_path)
            if [str(row["id"]) for row in rows] != benchmark_ids:
                raise ValueError(
                    f"{input_path} is not aligned with the canonical benchmark."
                )
            documents = {
                name: (
                    toolde_document(tool, profiles[name])
                    if method == "qts_de"
                    else qts_document(tool)
                )
                for name, tool in registry.items()
            }
            output_dir = args.result_root / "reranked" / method
            reranked = rerank_rows(
                rows,
                documents=documents,
                client=client,
                output_path=output_dir / f"{dataset}_top20.jsonl",
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
            metrics_path = output_dir / f"{dataset}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "method": method,
                        "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                        "candidate_depth": TOP_K,
                        "ks": list(KS),
                        "num_queries": len(reranked),
                        "metrics": metrics,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved reranked {dataset}/{method}", flush=True)


if __name__ == "__main__":
    main()
