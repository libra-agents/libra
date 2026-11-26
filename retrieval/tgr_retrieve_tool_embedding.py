"""TGR-style graph-enhanced tool embedding and retrieval.

Main encoding follows the Tool Graph Retriever graph convolution without
trainable parameters:

    M = A_sym + I
    D = diag(rowsum(M))
    X_tgr = D^-1/2 M D^-1/2 X_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.dense_retrieve_tool_embedding import (
    load_benchmark,
    load_npz_embeddings,
    retrieve_embedding,
)
from retrieval.metrics import evaluate_retrieval


TGR_EMBEDDING_VERSION = "tgr_tool_embeddings_v1"


def load_graph(path: str | Path) -> dict[str, Any]:
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if graph.get("version") != "tgr_llm_graph_v1":
        raise ValueError(f"Unsupported graph version in {path}: {graph.get('version')}")
    return graph


def build_directed_adjacency(
    *,
    graph: dict[str, Any],
    embedding_names: list[str],
    allow_name_mismatch: bool = False,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Build directed A[i, j] = 1 iff tool i depends on tool j.

    Uses directed_edges (source depends on target). Rows index consumers,
    columns index prerequisites, so propagation lets a consumer absorb its
    prerequisites without polluting prerequisite hubs.
    """
    name_to_idx = {name: idx for idx, name in enumerate(embedding_names)}
    n = len(embedding_names)
    adjacency = np.zeros((n, n), dtype=np.float32)
    dropped_edges: list[dict[str, str]] = []
    for edge in graph.get("directed_edges", []):
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src not in name_to_idx or tgt not in name_to_idx:
            dropped_edges.append({"source": src, "target": tgt})
            continue
        i, j = name_to_idx[src], name_to_idx[tgt]
        if i != j:
            adjacency[i, j] = 1.0
    if dropped_edges and not allow_name_mismatch:
        raise ValueError(
            f"{len(dropped_edges)} directed edges reference tools missing from embeddings. "
            "Pass --allow-name-mismatch to drop them."
        )
    return adjacency, dropped_edges


def build_symmetric_adjacency(
    *,
    graph: dict[str, Any],
    embedding_names: list[str],
    allow_name_mismatch: bool = False,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Build A_sym in the exact order of tool_embeddings.npz names."""
    graph_num_tools = graph.get("num_tools")
    if graph_num_tools is not None and int(graph_num_tools) != len(embedding_names) and not allow_name_mismatch:
        raise ValueError(
            f"Graph num_tools ({graph_num_tools}) != embedding names ({len(embedding_names)}). "
            "Pass --allow-name-mismatch to skip unmatched edges."
        )

    name_to_idx = {name: idx for idx, name in enumerate(embedding_names)}
    if len(name_to_idx) != len(embedding_names):
        raise ValueError("Duplicate names in tool embeddings.")

    adjacency = np.zeros((len(embedding_names), len(embedding_names)), dtype=np.float32)
    dropped_edges: list[dict[str, str]] = []

    for edge in graph.get("undirected_edges", []):
        a = str(edge.get("tool_a", ""))
        b = str(edge.get("tool_b", ""))
        if a not in name_to_idx or b not in name_to_idx:
            dropped_edges.append({"tool_a": a, "tool_b": b})
            continue
        i = name_to_idx[a]
        j = name_to_idx[b]
        if i == j:
            continue
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    if dropped_edges and not allow_name_mismatch:
        raise ValueError(
            f"{len(dropped_edges)} graph edges reference tools missing from embeddings. "
            "Pass --allow-name-mismatch to drop them."
        )
    return adjacency, dropped_edges


def graph_encode_embeddings(
    base_embeddings: np.ndarray,
    adjacency: np.ndarray,
    normalization: str = "sym",
) -> np.ndarray:
    """Single parameter-free graph convolution over (A + I).

    normalization:
      - "sym": D^-1/2 (A+I) D^-1/2 X   (TGR paper formula; D = rowsum(A+I))
      - "rw" : D_out^-1 (A+I) X        (random-walk / row-normalized)

    Works for both symmetric (undirected) and asymmetric (directed) A.
    """
    if adjacency.shape != (base_embeddings.shape[0], base_embeddings.shape[0]):
        raise ValueError(
            f"Adjacency shape {adjacency.shape} does not match embeddings {base_embeddings.shape}"
        )
    matrix = adjacency.astype(np.float32, copy=True)
    matrix += np.eye(matrix.shape[0], dtype=np.float32)
    degrees = matrix.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError("A+I produced a non-positive degree; this should be impossible.")
    if normalization == "sym":
        inv_sqrt = (1.0 / np.sqrt(degrees)).astype(np.float32)
        encoded = inv_sqrt[:, None] * (matrix @ (inv_sqrt[:, None] * base_embeddings))
    elif normalization == "rw":
        encoded = (matrix / degrees[:, None]) @ base_embeddings
    else:
        raise ValueError(f"Unknown normalization: {normalization!r} (use 'sym' or 'rw')")
    if not np.isfinite(encoded).all():
        raise ValueError("Non-finite values in TGR embeddings.")
    return encoded.astype(np.float32)


def write_tgr_embeddings(
    *,
    tool_embeddings_path: str | Path,
    graph_path: str | Path,
    output_path: str | Path,
    adjacency: str = "undirected",
    normalization: str = "sym",
    allow_name_mismatch: bool = False,
) -> tuple[list[str], np.ndarray, list[dict[str, str]]]:
    names, base_embeddings = load_npz_embeddings(tool_embeddings_path, item_key="names")
    graph = load_graph(graph_path)
    if adjacency == "undirected":
        adj, dropped_edges = build_symmetric_adjacency(
            graph=graph, embedding_names=names, allow_name_mismatch=allow_name_mismatch,
        )
    elif adjacency == "directed":
        adj, dropped_edges = build_directed_adjacency(
            graph=graph, embedding_names=names, allow_name_mismatch=allow_name_mismatch,
        )
    else:
        raise ValueError(f"Unknown adjacency: {adjacency!r} (use 'undirected' or 'directed')")
    embeddings = graph_encode_embeddings(base_embeddings, adj, normalization=normalization)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        names=np.array(names, dtype=object),
        embeddings=embeddings,
        version=TGR_EMBEDDING_VERSION,
    )
    return names, embeddings, dropped_edges


def load_tgr_embeddings(path: str | Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if str(data.get("version", "")) != TGR_EMBEDDING_VERSION:
        raise ValueError(f"{path} is not a {TGR_EMBEDDING_VERSION} file.")
    if "names" not in data or "embeddings" not in data:
        raise KeyError(f"{path} must contain names and embeddings.")
    names = [str(x) for x in data["names"]]
    embeddings = data["embeddings"].astype(np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(names):
        raise ValueError(f"Bad TGR embedding shape: {embeddings.shape}, names={len(names)}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite TGR embeddings in {path}")
    return names, embeddings


def run_tgr_retrieval(
    *,
    benchmark_path: str | Path,
    tgr_tool_embeddings_path: str | Path,
    query_embeddings_path: str | Path,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    question_key: str = "question",
    gold_key: str = "tool_path",
) -> tuple[dict[str, float], list[dict]]:
    benchmark = load_benchmark(benchmark_path, question_key=question_key, gold_key=gold_key)
    tool_names, tool_embeddings = load_tgr_embeddings(tgr_tool_embeddings_path)
    questions, query_embeddings = load_npz_embeddings(query_embeddings_path, item_key="questions")

    benchmark_questions = [str(item[question_key]) for item in benchmark]
    if questions != benchmark_questions:
        raise ValueError(
            "Query embeddings are not aligned with benchmark questions. "
            "Regenerate query embeddings from the same benchmark.jsonl."
        )
    if query_embeddings.shape[1] != tool_embeddings.shape[1]:
        raise ValueError(
            f"Query/tool embedding dimension mismatch: {query_embeddings.shape[1]} vs {tool_embeddings.shape[1]}"
        )

    tool_set = set(tool_names)
    missing_gold = sorted({
        str(tool)
        for item in benchmark
        for tool in item[gold_key]
        if str(tool) not in tool_set
    })
    if missing_gold:
        raise ValueError(f"Gold tools missing from TGR embeddings: {missing_gold[:20]}")

    max_k = max(max(ks), top_k)
    metric_pairs: list[tuple[list[str], list[str]]] = []
    rows: list[dict] = []
    for item, query_emb in zip(benchmark, query_embeddings):
        ranked = retrieve_embedding(query_emb, tool_names, tool_embeddings, top_k=max_k)
        predicted = [name for name, _ in ranked]
        gold = [str(x) for x in item[gold_key]]
        metric_pairs.append((predicted, gold))
        rows.append({
            "id": item.get("id"),
            "question": item[question_key],
            "gold": gold,
            "predicted": [{"name": name, "score": score} for name, score in ranked[:top_k]],
        })

    return evaluate_retrieval(metric_pairs, ks=ks), rows


def _parse_ks(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="TGR-style graph-enhanced tool retrieval.")
    parser.add_argument("--build-only", action="store_true", help="Only build tgr_tool_embeddings.npz.")
    parser.add_argument("--tool-embeddings", help="Base tool_embeddings.npz for building TGR embeddings.")
    parser.add_argument("--graph", help="tgr_llm_graph.json.")
    parser.add_argument("--write-tgr-embeddings", help="Output tgr_tool_embeddings.npz.")
    parser.add_argument("--adjacency", choices=["undirected", "directed"], default="undirected",
                        help="undirected (TGR symmetric edges) or directed (consumer->prerequisite).")
    parser.add_argument("--normalization", choices=["sym", "rw"], default="sym",
                        help="sym: D^-1/2(A+I)D^-1/2 (TGR formula); rw: D_out^-1(A+I) (random-walk).")
    parser.add_argument("--allow-name-mismatch", action="store_true")
    parser.add_argument("--benchmark", help="benchmark.jsonl with question and tool_path.")
    parser.add_argument("--query-embeddings", help="query_embeddings.npz.")
    parser.add_argument("--tgr-tool-embeddings", help="Prebuilt tgr_tool_embeddings.npz.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--gold-key", default="tool_path")
    parser.add_argument("--output", help="Optional predictions JSONL.")
    parser.add_argument("--metrics-output", help="Optional metrics JSON.")
    args = parser.parse_args()

    if args.write_tgr_embeddings:
        if not args.tool_embeddings or not args.graph:
            raise SystemExit("--tool-embeddings and --graph are required with --write-tgr-embeddings")
        names, embeddings, dropped = write_tgr_embeddings(
            tool_embeddings_path=args.tool_embeddings,
            graph_path=args.graph,
            output_path=args.write_tgr_embeddings,
            adjacency=args.adjacency,
            normalization=args.normalization,
            allow_name_mismatch=args.allow_name_mismatch,
        )
        print(json.dumps({
            "saved": args.write_tgr_embeddings,
            "adjacency": args.adjacency,
            "normalization": args.normalization,
            "num_tools": len(names),
            "shape": list(embeddings.shape),
            "num_dropped_edges": len(dropped),
        }, ensure_ascii=False, indent=2))

    if args.build_only:
        return

    if not args.benchmark or not args.query_embeddings or not args.tgr_tool_embeddings:
        raise SystemExit("--benchmark, --query-embeddings, and --tgr-tool-embeddings are required for retrieval")

    metrics, rows = run_tgr_retrieval(
        benchmark_path=args.benchmark,
        tgr_tool_embeddings_path=args.tgr_tool_embeddings,
        query_embeddings_path=args.query_embeddings,
        top_k=args.top_k,
        ks=_parse_ks(args.ks),
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
