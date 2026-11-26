#!/usr/bin/env python3
"""Precompute KG contextual reranking embeddings.

These embeddings cover candidate+seed pair texts used by ``retrieve_tools_kg``.
Run this before eval to avoid API calls during evaluation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from embedding.embed_tools import embed_texts
from retrieval.dense_retrieve_knowledge_graph import (
    CONTEXT_EMBEDDING_PATH,
    DEFAULT_KG_PATH,
    build_contextual_rerank_texts,
    load_kg_edges,
    load_kg_tools,
    restrict_kg_to_catalog,
    save_context_embedding_cache,
)


def _load_existing(
    cache_path: Path,
    *,
    texts: list[str],
    kg_path: Path,
    expand_hops: int,
) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    try:
        data = np.load(cache_path, allow_pickle=True)
    except Exception:
        return {}
    if "texts" not in data or "embeddings" not in data:
        return {}
    if "expand_hops" in data and int(data["expand_hops"]) != expand_hops:
        return {}
    cached_texts = list(map(str, data["texts"]))
    embeddings = data["embeddings"]
    if cached_texts != texts:
        return {}
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cached_texts):
        return {}
    return {text: embeddings[i] for i, text in enumerate(cached_texts)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-cache KG contextual reranking embeddings.")
    parser.add_argument("--kg-path", default=None, help="Path to graph_store.json.")
    parser.add_argument("--tool-registry", default=None)
    parser.add_argument("--expand-hops", type=int, default=3)
    parser.add_argument("--cache-path", default=str(CONTEXT_EMBEDDING_PATH))
    parser.add_argument("--force", action="store_true", help="Rebuild cache from scratch.")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override embedding model name.")
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    kg_path = Path(args.kg_path) if args.kg_path else DEFAULT_KG_PATH
    tools = load_kg_tools(kg_path)
    graph_dict = load_kg_edges(kg_path)
    if args.tool_registry:
        import json
        registry = json.loads(Path(args.tool_registry).read_text(encoding="utf-8"))
        tools, graph_dict = restrict_kg_to_catalog(
            tools,
            graph_dict,
            {str(tool["name"]) for tool in registry},
        )
    texts = build_contextual_rerank_texts(tools, graph_dict, expand_hops=args.expand_hops)
    cache_path = Path(args.cache_path)

    existing = {} if args.force else _load_existing(
        cache_path,
        texts=texts,
        kg_path=kg_path,
        expand_hops=args.expand_hops,
    )
    missing = [text for text in texts if text not in existing]

    print(f"KG tools: {len(tools)}")
    print(f"Context texts: {len(texts)}")
    print(f"Already cached: {len(texts) - len(missing)}")
    print(f"Missing: {len(missing)}")
    print(f"Cache path: {cache_path}")
    if args.dry_run:
        return 0
    if not missing and existing:
        print("Cache already complete.")
        return 0

    if missing:
        print(f"Embedding {len(missing)} missing context texts...")
        started = time.time()
        chunk_size = args.save_every
        for chunk_start in range(0, len(missing), chunk_size):
            chunk = missing[chunk_start:chunk_start + chunk_size]
            chunk_embs = embed_texts(
                chunk,
                model=args.model,
                batch_size=args.batch_size,
                max_concurrent=args.max_concurrent,
            )
            for text, emb in zip(chunk, chunk_embs):
                existing[text] = emb

            done = chunk_start + len(chunk)
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (len(missing) - done) / rate if rate > 0 else 0.0
            print(
                f"  {done}/{len(missing)} ({done / len(missing):.0%}), "
                f"{rate:.1f} texts/s, eta {remaining / 60:.1f}min",
                flush=True,
            )
            ordered = [t for t in texts if t in existing]
            ordered_embs = np.array([existing[t] for t in ordered], dtype=np.float32)
            save_context_embedding_cache(
                ordered, ordered_embs, cache_path=cache_path,
                kg_path=kg_path, expand_hops=args.expand_hops, model=args.model,
            )
            print(f"  checkpoint: {len(ordered)}/{len(texts)} texts", flush=True)

    ordered_embeddings = np.array([existing[text] for text in texts], dtype=np.float32)
    save_context_embedding_cache(
        texts, ordered_embeddings, cache_path=cache_path,
        kg_path=kg_path, expand_hops=args.expand_hops, model=args.model,
    )
    print(f"Saved {len(texts)} context embeddings, shape {ordered_embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
