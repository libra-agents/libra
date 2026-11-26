"""Pure dense tool retrieval over precomputed embeddings.

Cosine similarity between query and tool embeddings, return top-k. This is
benchmark-agnostic as long as the caller provides aligned query/tool embeddings
and tool names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retrieval.metrics import evaluate_retrieval
from retrieval.data import load_benchmark


def normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize embedding rows once for repeated cosine searches."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms).astype(np.float32)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and matrix rows."""
    query_normalized = query / (np.linalg.norm(query) + 1e-10)
    return normalize_embedding_matrix(matrix) @ query_normalized


def retrieve_normalized_embedding(
    query_emb: np.ndarray,
    tool_names: list[str],
    normalized_tool_embeddings: np.ndarray,
    *,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k cosine results from an already normalized tool matrix."""
    query = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    similarities = normalized_tool_embeddings @ query
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(tool_names[i], float(similarities[i])) for i in top_indices]


def retrieve_embedding(
    query_emb: np.ndarray,
    tool_names: list[str],
    tool_embeddings: np.ndarray,
    *,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return top-k cosine results from an unnormalized tool matrix."""
    return retrieve_normalized_embedding(
        query_emb,
        tool_names,
        normalize_embedding_matrix(tool_embeddings),
        top_k=top_k,
    )


def load_npz_embeddings(path: str | Path, *, item_key: str) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if item_key not in data or "embeddings" not in data:
        raise KeyError(f"{path} must contain {item_key!r} and 'embeddings'")
    items = [str(x) for x in data[item_key]]
    embeddings = data["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
        raise ValueError(f"Bad embedding shape in {path}: {embeddings.shape}, items={len(items)}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite embeddings in {path}")
    return items, embeddings


def run_dense_retrieval(
    *,
    benchmark_path: str | Path,
    tool_embeddings_path: str | Path,
    query_embeddings_path: str | Path,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    question_key: str = "question",
    gold_key: str = "tool_path",
) -> tuple[dict[str, float], list[dict]]:
    benchmark = load_benchmark(benchmark_path, question_key=question_key, gold_key=gold_key)
    tool_names, tool_embeddings = load_npz_embeddings(tool_embeddings_path, item_key="names")
    questions, query_embeddings = load_npz_embeddings(query_embeddings_path, item_key="questions")

    benchmark_questions = [str(item[question_key]) for item in benchmark]
    if questions != benchmark_questions:
        raise ValueError(
            "Query embeddings are not aligned with benchmark questions. "
            "Regenerate query embeddings from the same benchmark.jsonl."
        )
    if query_embeddings.shape[1] != tool_embeddings.shape[1]:
        raise ValueError(
            f"Query/tool embedding dimension mismatch: "
            f"{query_embeddings.shape[1]} vs {tool_embeddings.shape[1]}"
        )

    normalized_tool_embeddings = normalize_embedding_matrix(tool_embeddings)
    max_k = max(max(ks), top_k)
    rows: list[dict] = []
    metric_pairs: list[tuple[list[str], list[str]]] = []
    for item, query_emb in zip(benchmark, query_embeddings):
        ranked = retrieve_normalized_embedding(
            query_emb,
            tool_names,
            normalized_tool_embeddings,
            top_k=max_k,
        )
        predicted = [name for name, _ in ranked]
        gold = [str(x) for x in item[gold_key]]
        metric_pairs.append((predicted, gold))
        rows.append({
            "id": item.get("id"),
            "question": item[question_key],
            "gold": gold,
            "predicted": [{"name": name, "score": score} for name, score in ranked[:top_k]],
        })

    metrics = evaluate_retrieval(metric_pairs, ks=ks)
    return metrics, rows


def _parse_ks(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense tool retrieval baseline over precomputed embeddings.")
    parser.add_argument("--benchmark", required=True, help="benchmark.jsonl with question and tool_path.")
    parser.add_argument("--tool-embeddings", required=True, help="tool_embeddings.npz with names + embeddings.")
    parser.add_argument("--query-embeddings", required=True, help="query_embeddings.npz with questions + embeddings.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of predictions to save per query.")
    parser.add_argument("--ks", default="1,3,5,10", help="Comma-separated metric cutoffs.")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="tool_path")
    parser.add_argument("--output", help="Optional output predictions JSONL.")
    parser.add_argument("--metrics-output", help="Optional metrics JSON output.")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    metrics, rows = run_dense_retrieval(
        benchmark_path=args.benchmark,
        tool_embeddings_path=args.tool_embeddings,
        query_embeddings_path=args.query_embeddings,
        top_k=args.top_k,
        ks=ks,
        question_key=args.question_key,
        gold_key=args.gold_key,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions: {output_path}")

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
