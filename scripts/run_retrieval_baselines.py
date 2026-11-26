#!/usr/bin/env python3
"""Run BM25, TF-IDF, QTS, QTS-TGR, and QTS-DE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.bm25_retrieve_tool_text import run_bm25_retrieval
from retrieval.dense_retrieve_tool_embedding import run_dense_retrieval
from retrieval.tfidf_retrieve_tool_text import run_tfidf_retrieval
from retrieval.tgr_retrieve_tool_embedding import (
    run_tgr_retrieval,
    write_tgr_embeddings,
)


DATASETS = ("appworld", "scitool", "bfcl")
EMBEDDING_DIRS = {
    "appworld": "appworld",
    "scitool": "scitool_clean",
    "bfcl": "bfcl",
}
TGR_GRAPH_FILES = {
    "appworld": "tgr_graph.json",
    "scitool": "tgr_llm_graph.json",
    "bfcl": "tgr_graph.json",
}
METHODS = ("bm25", "tfidf", "qts", "qts_tgr", "qts_de")
KS = (1, 3, 5, 10, 15, 20)
TOP_K = 20


def dataset_directory(
    repo_root: Path,
    data_root: Path,
    dataset: str,
) -> Path:
    return (
        repo_root / "dataset and embedding" / "scitool_clean"
        if dataset == "scitool"
        else data_root / dataset
    )


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_method(
    *,
    method: str,
    source_dir: Path,
    embedding_dir: Path,
    tgr_embedding_path: Path,
) -> tuple[dict[str, float], list[dict]]:
    benchmark = source_dir / "benchmark.jsonl"
    registry = source_dir / "tool_registry.json"
    query_embeddings = embedding_dir / "query_embeddings.npz"
    if method == "bm25":
        return run_bm25_retrieval(
            benchmark_path=benchmark,
            tools_path=registry,
            top_k=TOP_K,
            ks=KS,
        )
    if method == "tfidf":
        return run_tfidf_retrieval(
            benchmark_path=benchmark,
            tools_path=registry,
            top_k=TOP_K,
            ks=KS,
        )
    if method == "qts":
        tool_embeddings = embedding_dir / "tool_embeddings.npz"
    elif method == "qts_de":
        tool_embeddings = embedding_dir / "toolde_embeddings.npz"
    elif method == "qts_tgr":
        return run_tgr_retrieval(
            benchmark_path=benchmark,
            tgr_tool_embeddings_path=tgr_embedding_path,
            query_embeddings_path=query_embeddings,
            top_k=TOP_K,
            ks=KS,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return run_dense_retrieval(
        benchmark_path=benchmark,
        tool_embeddings_path=tool_embeddings,
        query_embeddings_path=query_embeddings,
        top_k=TOP_K,
        ks=KS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS)
    methods = [args.method] if args.method else list(METHODS)
    data_root = args.data_root or args.repo_root / "data"
    for dataset in datasets:
        source_dir = dataset_directory(args.repo_root, data_root, dataset)
        embedding_dir = args.embedding_root / EMBEDDING_DIRS[dataset]
        tgr_embedding_path = embedding_dir / "tgr_tool_embeddings.npz"
        if "qts_tgr" in methods and not tgr_embedding_path.exists():
            write_tgr_embeddings(
                tool_embeddings_path=embedding_dir / "tool_embeddings.npz",
                graph_path=source_dir / TGR_GRAPH_FILES[dataset],
                output_path=tgr_embedding_path,
            )
        for method in methods:
            metrics, rows = run_method(
                method=method,
                source_dir=source_dir,
                embedding_dir=embedding_dir,
                tgr_embedding_path=tgr_embedding_path,
            )
            method_dir = args.output_root / "first_stage" / method
            prediction_path = method_dir / f"{dataset}_top20.jsonl"
            metrics_path = method_dir / f"{dataset}_metrics.json"
            write_predictions(prediction_path, rows)
            metrics_path.write_text(
                json.dumps({
                    "dataset": dataset,
                    "method": method,
                    "embedding_model": (
                        None
                        if method in {"bm25", "tfidf"}
                        else "Qwen/Qwen3-Embedding-0.6B"
                    ),
                    "top_k": TOP_K,
                    "ks": list(KS),
                    "num_queries": len(rows),
                    "metrics": metrics,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved {dataset}/{method}: {prediction_path}", flush=True)


if __name__ == "__main__":
    main()
