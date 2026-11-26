"""Directed one-hop dependency retrieval over a frozen compact graph."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from retrieval.dense_retrieve_tool_embedding import (
    load_benchmark,
    load_npz_embeddings,
)
from retrieval.metrics import evaluate_retrieval


SEED_K = 15
TOP_K = 20
KS = (1, 3, 5, 10, 15, 20)


@dataclass(frozen=True)
class DependencyEdge:
    edge_id: str
    provider: str
    consumer: str
    consumer_slot: str
    kind: str
    field_score: float | None = None


def load_graph(path: str | Path) -> list[DependencyEdge]:
    """Load the small graph artifact consumed by every benchmark adapter."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        DependencyEdge(
            edge_id=str(row["id"]),
            provider=str(row["source"]),
            consumer=str(row["target"]),
            consumer_slot=str(row["input"]),
            kind=str(row["kind"]),
            field_score=(
                float(row["score"]) if row.get("score") is not None else None
            ),
        )
        for row in payload["edges"]
    ]


def write_graph(edges: Sequence[DependencyEdge], path: str | Path) -> None:
    """Publish retrieval evidence without build/runtime provenance."""

    rows = []
    for edge in sorted(edges, key=lambda row: row.edge_id):
        row: dict[str, Any] = {
            "id": edge.edge_id,
            "source": edge.provider,
            "target": edge.consumer,
            "input": edge.consumer_slot,
            "kind": edge.kind,
        }
        if edge.field_score is not None:
            row["score"] = edge.field_score
        rows.append(row)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"edges": rows},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _best_edges(
    edges: Sequence[DependencyEdge],
    *,
    seed_tools: set[str],
    relevance: Mapping[str, float],
) -> list[DependencyEdge]:
    """Keep field-best and query-best providers for each consumer input."""

    field_best: dict[
        tuple[str, str], tuple[tuple[Any, ...], DependencyEdge]
    ] = {}
    query_best: dict[
        tuple[str, str], tuple[tuple[Any, ...], DependencyEdge]
    ] = {}
    for edge in edges:
        if edge.consumer not in seed_tools:
            continue
        field_score = edge.field_score if edge.field_score is not None else -2.0
        key = (edge.consumer, edge.consumer_slot)
        field_priority = (
            edge.field_score is not None,
            field_score,
            relevance[edge.provider],
            edge.provider,
            edge.edge_id,
        )
        query_priority = (
            relevance[edge.provider],
            edge.field_score is not None,
            field_score,
            edge.provider,
            edge.edge_id,
        )
        if key not in field_best or field_priority > field_best[key][0]:
            field_best[key] = (field_priority, edge)
        if key not in query_best or query_priority > query_best[key][0]:
            query_best[key] = (query_priority, edge)

    selected = [row[1] for row in field_best.values()]
    selected_ids = {edge.edge_id for edge in selected}
    selected.extend(
        row[1]
        for row in query_best.values()
        if row[1].edge_id not in selected_ids
    )
    return selected


def rank_query(
    *,
    tool_names: Sequence[str],
    dense_scores: np.ndarray,
    edges: Sequence[DependencyEdge],
    consumer_fusion: str,
    seed_k: int = SEED_K,
    top_k: int = TOP_K,
) -> tuple[list[str], dict[str, float], dict[str, dict[str, Any]]]:
    """Promote upstream providers of Dense Top-15 consumers."""

    if consumer_fusion not in {"sum", "mean"}:
        raise ValueError(f"unknown consumer fusion: {consumer_fusion}")
    dense_indices = np.argsort(dense_scores)[::-1]
    dense_order = [tool_names[int(index)] for index in dense_indices]
    dense_rank = {tool: rank for rank, tool in enumerate(dense_order)}
    relevance = {
        tool: float(np.clip((dense_scores[index] + 1.0) / 2.0, 0.0, 1.0))
        for index, tool in enumerate(tool_names)
    }
    selected_edges = _best_edges(
        edges,
        seed_tools=set(dense_order[:seed_k]),
        relevance=relevance,
    )

    messages: dict[str, dict[str, float]] = defaultdict(dict)
    message_edges: dict[str, dict[str, DependencyEdge]] = defaultdict(dict)
    for edge in selected_edges:
        previous = message_edges[edge.provider].get(edge.consumer)
        if previous is None or (
            edge.field_score if edge.field_score is not None else -2.0,
            edge.edge_id,
        ) > (
            previous.field_score if previous.field_score is not None else -2.0,
            previous.edge_id,
        ):
            messages[edge.provider][edge.consumer] = relevance[edge.consumer]
            message_edges[edge.provider][edge.consumer] = edge

    scores = dict(relevance)
    propagated: dict[str, dict[str, Any]] = {}
    for provider, consumer_scores in messages.items():
        score = sum(consumer_scores.values())
        if consumer_fusion == "mean":
            score = (relevance[provider] + score) / (len(consumer_scores) + 1)
        if score <= scores[provider] + 1e-12:
            continue
        scores[provider] = score
        representative = max(
            message_edges[provider].values(),
            key=lambda edge: (
                relevance[edge.consumer],
                edge.field_score if edge.field_score is not None else -2.0,
                edge.edge_id,
            ),
        )
        propagated[provider] = {
            "edge": representative.edge_id,
            "source": provider,
            "target": representative.consumer,
            "input": representative.consumer_slot,
            "kind": representative.kind,
            "score": score,
            "consumers": sorted(consumer_scores),
        }

    order = sorted(
        tool_names,
        key=lambda tool: (
            -scores[tool],
            tool in propagated,
            dense_rank[tool],
        ),
    )[:top_k]
    return order, scores, {
        tool: propagated[tool] for tool in order if tool in propagated
    }


def _place_additions_after_dense(
    order: Sequence[str], dense_seed: Sequence[str], cutoff: int
) -> list[str]:
    return place_graph_additions_after_seed(
        order,
        dense_seed=dense_seed,
        cutoff=cutoff,
    )[:cutoff]


def place_graph_additions_after_seed(
    order: Sequence[str],
    *,
    dense_seed: Sequence[str],
    cutoff: int,
) -> list[str]:
    """Keep Top-K membership fixed, with dense seeds before graph additions."""

    selected = list(order[:cutoff])
    selected_set = set(selected)
    seed_set = set(dense_seed)
    return [
        *(tool for tool in dense_seed if tool in selected_set),
        *(tool for tool in selected if tool not in seed_set),
        *order[cutoff:],
    ]


def _names(items: Sequence[str | Mapping[str, Any]]) -> list[str]:
    return [str(item if isinstance(item, str) else item["name"]) for item in items]


def _dense_metrics(
    rows: Sequence[Mapping[str, Any]], ks: tuple[int, ...]
) -> dict[str, dict[str, float]]:
    return _metrics_by_scope(
        rows,
        lambda row, _: _names(row["dense_predicted"]),
        ks,
    )


def _graph_metrics(
    rows: Sequence[Mapping[str, Any]], ks: tuple[int, ...]
) -> dict[str, dict[str, float]]:
    return _metrics_by_scope(
        rows,
        lambda row, cutoff: _names(row["predicted_by_k"][str(cutoff)]),
        ks,
    )


def _metrics_by_scope(
    rows: Sequence[Mapping[str, Any]],
    predictions: Callable[[Mapping[str, Any], int], list[str]],
    ks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    scopes = ["all", *sorted({str(row["split"]) for row in rows if row["split"]})]
    result: dict[str, dict[str, float]] = {}
    for scope in scopes:
        selected = [
            row for row in rows if scope == "all" or row["split"] == scope
        ]
        values: dict[str, float] = {}
        for cutoff in ks:
            current = evaluate_retrieval(
                [(predictions(row, cutoff), row["gold"]) for row in selected],
                ks=(cutoff,),
            )
            values.update({key: value for key, value in current.items() if key != "MRR"})
        values["MRR"] = evaluate_retrieval(
            [(predictions(row, max(ks)), row["gold"]) for row in selected],
            ks=(max(ks),),
        )["MRR"]
        result[scope] = values
    return result


def run_dependency_retrieval(
    *,
    benchmark_path: str | Path,
    tool_embeddings_path: str | Path,
    query_embeddings_path: str | Path,
    graph_path: str | Path,
    consumer_fusion: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    edges = load_graph(graph_path)
    benchmark = load_benchmark(
        benchmark_path,
        question_key="question",
        gold_key="tool_path",
    )
    tool_names, tool_embeddings = load_npz_embeddings(
        tool_embeddings_path, item_key="names"
    )
    questions, query_embeddings = load_npz_embeddings(
        query_embeddings_path, item_key="questions"
    )
    if questions != [str(row["question"]) for row in benchmark]:
        raise ValueError("query embeddings are not aligned with benchmark")

    known_tools = set(tool_names)
    unknown_edges = [
        edge.edge_id
        for edge in edges
        if edge.provider not in known_tools or edge.consumer not in known_tools
    ]
    if unknown_edges:
        raise ValueError(f"{len(unknown_edges)} edges reference unknown tools")

    tools = tool_embeddings / (
        np.linalg.norm(tool_embeddings, axis=1, keepdims=True) + 1e-10
    )
    queries = query_embeddings / (
        np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10
    )
    dense_matrix = queries @ tools.T

    rows: list[dict[str, Any]] = []
    for item, dense_scores in zip(benchmark, dense_matrix, strict=True):
        dense_indices = np.argsort(dense_scores)[::-1][:TOP_K]
        dense_predicted = [
            {"name": tool_names[int(index)], "score": float(dense_scores[int(index)])}
            for index in dense_indices
        ]
        graph_order, graph_scores, propagated = rank_query(
            tool_names=tool_names,
            dense_scores=dense_scores,
            edges=edges,
            consumer_fusion=consumer_fusion,
        )
        dense_seed = [tool_names[int(index)] for index in dense_indices[:SEED_K]]
        predicted_by_k = {
            str(cutoff): [
                {"name": tool, "score": graph_scores[tool]}
                for tool in _place_additions_after_dense(
                    graph_order, dense_seed, cutoff
                )
            ]
            for cutoff in KS
        }
        rows.append({
            "id": item.get("id"),
            "split": str(item.get("split") or item.get("dataset") or ""),
            "question": str(item["question"]),
            "gold": list(map(str, item["tool_path"])),
            "predicted_by_k": predicted_by_k,
            "dense_predicted": dense_predicted,
            "propagated": propagated,
        })

    dense_metrics = _dense_metrics(rows, KS)
    graph_metrics = _graph_metrics(rows, KS)
    return {
        "policy": f"directed_{consumer_fusion}_one_hop",
        "seed_k": SEED_K,
        "edges": len(edges),
        "dense": dense_metrics,
        "graph": graph_metrics,
        "delta": {
            scope: {
                key: graph_metrics[scope][key] - dense_metrics[scope][key]
                for key in graph_metrics[scope]
            }
            for scope in graph_metrics
        },
    }, rows
