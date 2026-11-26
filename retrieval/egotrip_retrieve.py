"""Ego-Trip entry matching, one-hop expansion, and evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.build_egotrip_graph import GRAPH_VERSION
from retrieval.data import load_benchmark
from retrieval.dense_retrieve_tool_embedding import (
    load_npz_embeddings,
    normalize_embedding_matrix,
)


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def load_graph(path: str | Path) -> dict[str, Any]:
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if graph.get("version") != GRAPH_VERSION:
        raise ValueError(f"{path}: expected {GRAPH_VERSION}, found {graph.get('version')!r}")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"{path}: graph must contain node and edge arrays")
    return graph


def query_ngrams(text: str, max_n: int = 3) -> set[str]:
    tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)]
    return {
        " ".join(tokens[start : start + n])
        for n in range(1, max_n + 1)
        for start in range(0, len(tokens) - n + 1)
    }


def lexical_entry_nodes(question: str, nodes: list[dict[str, Any]]) -> set[str]:
    """Match graph nodes by unigram, bigram, and trigram overlap."""
    ngrams = query_ngrams(question)
    return {
        str(node["id"])
        for node in nodes
        if str(node.get("label", "")).casefold() in ngrams
    }


def adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
    neighbors = {str(node["id"]): set() for node in graph["nodes"]}
    for edge in graph["edges"]:
        source, target = str(edge["source"]), str(edge["target"])
        if source not in neighbors or target not in neighbors:
            raise ValueError(f"Graph edge references an unknown node: {source!r}, {target!r}")
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def retrieve_candidates(
    *,
    question: str,
    query_embedding: np.ndarray,
    graph: dict[str, Any],
    normalized_node_embeddings: np.ndarray,
    semantic_entry_k: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    nodes = graph["nodes"]
    if normalized_node_embeddings.shape[0] != len(nodes):
        raise ValueError("Node embeddings are not aligned with the graph")
    query = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    scores = normalized_node_embeddings @ query
    semantic_indices = np.argsort(scores)[::-1][:semantic_entry_k]
    semantic_entries = [str(nodes[index]["id"]) for index in semantic_indices]
    lexical_entries = sorted(lexical_entry_nodes(question, nodes))
    entries = set(semantic_entries) | set(lexical_entries)

    neighbors = adjacency(graph)
    ego_nodes = set(entries)
    for node_id in entries:
        ego_nodes.update(neighbors[node_id])
    node_by_id = {str(node["id"]): node for node in nodes}
    candidates = sorted(
        {
            str(node_by_id[node_id]["label"])
            for node_id in ego_nodes
            if str(node_by_id[node_id].get("type")) == "tool"
        }
    )
    return candidates, semantic_entries, lexical_entries


def run_candidate_retrieval(
    *,
    benchmark_path: str | Path,
    graph_path: str | Path,
    node_embeddings_path: str | Path,
    query_embeddings_path: str | Path,
    semantic_entry_k: int = 10,
) -> list[dict[str, Any]]:
    benchmark = load_benchmark(benchmark_path)
    graph = load_graph(graph_path)
    node_ids, node_embeddings = load_npz_embeddings(node_embeddings_path, item_key="node_ids")
    expected_node_ids = [str(node["id"]) for node in graph["nodes"]]
    if node_ids != expected_node_ids:
        raise ValueError("Node embedding IDs are not aligned with graph node order")
    questions, query_embeddings = load_npz_embeddings(query_embeddings_path, item_key="questions")
    expected_questions = [str(row["question"]) for row in benchmark]
    if questions != expected_questions:
        raise ValueError("Query embeddings are not aligned with benchmark questions")
    if node_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("Node and query embedding dimensions differ")
    normalized_nodes = normalize_embedding_matrix(node_embeddings)

    rows: list[dict[str, Any]] = []
    for item, query_embedding in zip(benchmark, query_embeddings):
        candidates, semantic_entries, lexical_entries = retrieve_candidates(
            question=str(item["question"]),
            query_embedding=query_embedding,
            graph=graph,
            normalized_node_embeddings=normalized_nodes,
            semantic_entry_k=semantic_entry_k,
        )
        rows.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "gold": [str(name) for name in item["tool_path"]],
                "candidate_tools": candidates,
                "semantic_entry_nodes": semantic_entries,
                "lexical_entry_nodes": lexical_entries,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--node-embeddings", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-entry-k", type=int, default=10)
    args = parser.parse_args()
    rows = run_candidate_retrieval(
        benchmark_path=args.benchmark,
        graph_path=args.graph,
        node_embeddings_path=args.node_embeddings,
        query_embeddings_path=args.query_embeddings,
        semantic_entry_k=args.semantic_entry_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = [len(row["candidate_tools"]) for row in rows]
    print(
        json.dumps(
            {
                "queries": len(rows),
                "candidate_min": min(counts, default=0),
                "candidate_mean": sum(counts) / len(counts) if counts else 0,
                "candidate_max": max(counts, default=0),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
