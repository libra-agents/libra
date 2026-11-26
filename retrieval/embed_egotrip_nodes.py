"""Embed Ego-Trip graph nodes with Qwen3-Embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.build_embeddings import embed_batches, save_embeddings
from retrieval.data import load_benchmark, qwen3_query_text
from retrieval.egotrip_retrieve import load_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8101")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--query-output", type=Path)
    args = parser.parse_args()
    if (args.benchmark is None) != (args.query_output is None):
        raise SystemExit("--benchmark and --query-output must be provided together")
    graph = load_graph(args.graph)
    node_ids = [str(node["id"]) for node in graph["nodes"]]
    texts = [str(node["text"]) for node in graph["nodes"]]
    embeddings = embed_batches(
        texts,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
    )
    save_embeddings(
        args.output,
        item_key="node_ids",
        items=node_ids,
        embeddings=embeddings,
        model=args.model,
        representation="egotrip-graph-node-text",
    )
    result = {"nodes": len(node_ids), "output": str(args.output)}
    if args.benchmark is not None and args.query_output is not None:
        benchmark = load_benchmark(args.benchmark)
        questions = [str(row["question"]) for row in benchmark]
        query_embeddings = embed_batches(
            [qwen3_query_text(question) for question in questions],
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
        )
        save_embeddings(
            args.query_output,
            item_key="questions",
            items=questions,
            embeddings=query_embeddings,
            model=args.model,
            representation="qwen3-instruction-query",
        )
        result.update({"queries": len(questions), "query_output": str(args.query_output)})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
