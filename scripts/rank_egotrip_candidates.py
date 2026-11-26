#!/usr/bin/env python3
"""Budget-match Ego-Trip candidates with frozen QTS ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.dense_retrieve_tool_embedding import (
    load_npz_embeddings,
    normalize_embedding_matrix,
)
from retrieval.metrics import evaluate_retrieval
from scripts.rerank_egotrip_retrieval import KS, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--tool-embeddings", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--candidate-depth", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(args.candidates)
    tool_names, tool_embeddings = load_npz_embeddings(
        args.tool_embeddings, item_key="names"
    )
    questions, query_embeddings = load_npz_embeddings(
        args.query_embeddings, item_key="questions"
    )
    if questions != [str(row["question"]) for row in rows]:
        raise ValueError("Query embeddings are not aligned with candidate rows")
    tool_index = {name: index for index, name in enumerate(tool_names)}
    normalized_tools = normalize_embedding_matrix(tool_embeddings)
    normalized_queries = normalize_embedding_matrix(query_embeddings)

    ranked_rows = []
    for row, query in zip(rows, normalized_queries):
        candidates = [str(name) for name in row["candidate_tools"]]
        missing = sorted(set(candidates) - set(tool_index))
        if missing:
            raise ValueError(f"{row['id']}: candidates absent from embeddings: {missing[:5]}")
        scored = [
            (name, float(normalized_tools[tool_index[name]] @ query))
            for name in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked_rows.append({
            **row,
            "predicted": [
                {"name": name, "first_stage_score": score}
                for name, score in scored[: args.candidate_depth]
            ],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in ranked_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = evaluate_retrieval(
        [
            ([item["name"] for item in row["predicted"]], row["gold"])
            for row in ranked_rows
        ],
        ks=KS,
    )
    payload = {
        "method": "Ego-Trip (adapted, budget-matched)",
        "candidate_generation": "Ego-Trip one-hop expansion",
        "candidate_ranking": "QTS with Qwen/Qwen3-Embedding-0.6B",
        "candidate_depth": args.candidate_depth,
        "reranker_model": None,
        "num_queries": len(ranked_rows),
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
