"""Dense KG retrieval baseline for SciToolAgent experiments.

Reimplements the core retrieval logic from SciToolAgent's KGTableRetriever.
Loads the knowledge graph from ``graph_store.json``, builds per-tool
functionality embeddings, and retrieves the top-k relevant tools via cosine
similarity + KG expansion + contextual reranking.

Two retrieval modes:
  - Full (retrieve_tools_kg): seed → KG expand → path filter → contextual
    reranking.  Requires pre-cached context embeddings.
  - Simple (retrieve_tools_simple): seed → BFS expand with decay.  No
    contextual reranking, no path compatibility checks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embedding.embed_tools import embed_texts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_KG_PATH = (
    Path(__file__).resolve().parent.parent
    / "SciToolAgent-main"
    / "KG"
    / "storage_graph_large"
    / "graph_store.json"
)

_SCITOOL_DIR = Path(__file__).resolve().parent.parent / "dataset and embedding" / "scitool_clean"

TOOL_EMBEDDING_PATH = _SCITOOL_DIR / "kg_tool_embeddings.npz"

QUERY_CACHE_PATH = _SCITOOL_DIR / "query_embeddings.npz"

CONTEXT_EMBEDDING_PATH = _SCITOOL_DIR / "kg_context_embeddings.npz"

KG_EDGE_TYPES = {
    "inputs",
    "outputs",
    "is the input of",
    "is the output of",
    "has the functionality that",
}

TOOL_EMBEDDING_VERSION = "functionality_triplet_v2"
CONTEXT_EMBEDDING_VERSION = "kg_context_v1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolNode:
    name: str
    functionality: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    category: str = ""


# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

def embedding_model_name(model: str | None = None) -> str:
    return (
        os.getenv("EMBEDDING_MODEL")
        or model
        or "Qwen/Qwen3-Embedding-0.6B"
    )

def load_kg_tools(kg_path: Path = DEFAULT_KG_PATH) -> dict[str, ToolNode]:
    """Parse ``graph_store.json`` into a dict of tool nodes."""
    graph_dict = json.loads(kg_path.read_text(encoding="utf-8"))["graph_dict"]
    tools: dict[str, ToolNode] = {}
    for node, rels in graph_dict.items():
        for rel in rels:
            if rel[0] == "has the functionality that":
                tools[node] = ToolNode(name=node, functionality=rel[1])
    for name, tool in tools.items():
        if name not in graph_dict:
            continue
        for rel in graph_dict[name]:
            if rel[0] == "inputs":
                tool.inputs.append(rel[1])
            elif rel[0] == "outputs":
                tool.outputs.append(rel[1])
            elif rel[0] == "is a":
                tool.category = rel[1]
    return tools


def load_kg_edges(kg_path: Path = DEFAULT_KG_PATH) -> dict[str, list[list[str]]]:
    """Return the raw graph_dict for KG hop expansion."""
    return json.loads(kg_path.read_text(encoding="utf-8"))["graph_dict"]


def restrict_kg_to_catalog(
    tools: dict[str, ToolNode],
    graph_dict: dict[str, list[list[str]]],
    allowed_tools: set[str],
) -> tuple[dict[str, ToolNode], dict[str, list[list[str]]]]:
    """Project the released KG onto the benchmark tool catalog."""
    excluded_tools = set(tools) - allowed_tools
    return (
        {name: tool for name, tool in tools.items() if name in allowed_tools},
        {
            source: [rel for rel in relations if rel[1] not in excluded_tools]
            for source, relations in graph_dict.items()
            if source not in excluded_tools
        },
    )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_query_cache: dict[str, np.ndarray] | None = None
_text_embedding_cache: dict[str, np.ndarray] = {}
_context_disk_cache_loaded = False


def _load_query_cache(cache_path: Path = QUERY_CACHE_PATH) -> dict[str, np.ndarray]:
    global _query_cache
    if _query_cache is not None:
        return _query_cache
    if not cache_path.exists():
        _query_cache = {}
        return _query_cache
    data = np.load(cache_path, allow_pickle=True)
    questions = list(data["questions"])
    embeddings = data["embeddings"]
    _query_cache = {q: embeddings[i] for i, q in enumerate(questions)}
    return _query_cache


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return (matrix / norms) @ query_norm


def _load_context_disk_cache(cache_path: Path = CONTEXT_EMBEDDING_PATH) -> None:
    global _context_disk_cache_loaded
    if _context_disk_cache_loaded:
        return
    _context_disk_cache_loaded = True
    if not cache_path.exists():
        return
    data = np.load(cache_path, allow_pickle=True)
    version = str(data["version"]) if "version" in data else ""
    if version and version != CONTEXT_EMBEDDING_VERSION:
        return
    if "texts" not in data or "embeddings" not in data:
        return
    texts = list(map(str, data["texts"]))
    embeddings = data["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
        return
    for text, embedding in zip(texts, embeddings):
        _text_embedding_cache[text] = embedding


def load_context_embedding_cache(
    cache_path: Path = CONTEXT_EMBEDDING_PATH,
) -> None:
    """Load the frozen contextual embeddings before concurrent retrieval."""

    _load_context_disk_cache(cache_path)


def _embed_texts_cached(
    texts: list[str],
    *,
    cache_path: Path = CONTEXT_EMBEDDING_PATH,
) -> np.ndarray:
    """Load context embeddings from pre-computed cache; never call the API."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    _load_context_disk_cache(cache_path)
    missing = [t for t in dict.fromkeys(texts) if t not in _text_embedding_cache]
    if missing:
        raise RuntimeError(
            "Missing KG context embeddings. Run "
            "`python -m embedding.embed_scitool_kg_context --expand-hops 3` first. "
            f"Missing {len(missing)} of {len(set(texts))} unique texts; "
            f"first: {missing[0][:200]}"
        )
    return np.array([_text_embedding_cache[t] for t in texts], dtype=np.float32)


def save_context_embedding_cache(
    texts: list[str],
    embeddings: np.ndarray,
    *,
    cache_path: Path = CONTEXT_EMBEDDING_PATH,
    kg_path: Path = DEFAULT_KG_PATH,
    expand_hops: int = 3,
    model: str | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        texts=np.array(texts),
        embeddings=embeddings.astype(np.float32),
        version=np.array(CONTEXT_EMBEDDING_VERSION),
        model=np.array(embedding_model_name(model)),
        kg_path=np.array(str(kg_path)),
        expand_hops=np.array(expand_hops),
    )


# ---------------------------------------------------------------------------
# Tool embedding text construction
# ---------------------------------------------------------------------------

def _functionality_triplet_text(tool: ToolNode) -> str:
    return f"{tool.name} has the functionality that {tool.functionality}"


def _contextual_functionality_text(*, candidate: ToolNode, seed: ToolNode) -> str:
    return (
        f"{_functionality_triplet_text(candidate)} "
        f"Existing retrieved tool: {_functionality_triplet_text(seed)}"
    )


# ---------------------------------------------------------------------------
# Tool embeddings
# ---------------------------------------------------------------------------

def build_tool_embeddings(
    tools: dict[str, ToolNode],
    *,
    cache_path: Path = TOOL_EMBEDDING_PATH,
    force: bool = False,
    **embed_kwargs: Any,
) -> tuple[list[str], np.ndarray]:
    """Return (tool_names, embeddings) with optional disk cache."""
    names = sorted(tools)
    if not force and cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        version = str(data["version"]) if "version" in data else ""
        cached_names = list(data["names"])
        embeddings = data["embeddings"]
        cache_ok = (
            version in {"", TOOL_EMBEDDING_VERSION}
            and cached_names == names
            and embeddings.ndim == 2
            and embeddings.shape[0] == len(names)
            and np.isfinite(embeddings).all()
        )
        if cache_ok:
            return cached_names, embeddings

        if (
            "version" not in data
            and cached_names == names
            and embeddings.ndim == 2
            and embeddings.shape[0] == len(names)
            and np.isfinite(embeddings).all()
        ):
            return cached_names, embeddings

    texts = [_functionality_triplet_text(tools[n]) for n in names]
    embeddings = embed_texts(texts, **embed_kwargs)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        names=np.array(names),
        embeddings=embeddings,
        version=np.array(TOOL_EMBEDDING_VERSION),
        model=np.array(embedding_model_name()),
    )
    return names, embeddings


# ---------------------------------------------------------------------------
# Contextual reranking text enumeration (for precaching)
# ---------------------------------------------------------------------------

def build_contextual_rerank_texts(
    tools: dict[str, ToolNode],
    graph_dict: dict[str, list[list[str]]],
    *,
    expand_hops: int = 3,
) -> list[str]:
    """Enumerate unique candidate+seed texts used by KG contextual reranking."""
    texts: list[str] = []
    seen: set[str] = set()
    for seed_name in sorted(tools):
        paths_by_tool = _collect_tool_paths(
            graph_dict, tools, start=seed_name,
            max_depth=expand_hops, edge_types=KG_EDGE_TYPES,
        )
        for candidate_name, paths in paths_by_tool.items():
            if candidate_name == seed_name or not _has_compatible_path(paths):
                continue
            text = _contextual_functionality_text(
                candidate=tools[candidate_name], seed=tools[seed_name],
            )
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Full KG retrieval (SciToolAgent with contextual reranking)
# ---------------------------------------------------------------------------

def retrieve_tools_kg(
    question: str,
    tools: dict[str, ToolNode],
    tool_names: list[str],
    tool_embeddings: np.ndarray,
    graph_dict: dict[str, list[list[str]]],
    *,
    top_k: int = 5,
    expand_hops: int = 3,
    max_tools: int = 10,
    query_cache_path: Path = QUERY_CACHE_PATH,
    context_embedding_path: Path = CONTEXT_EMBEDDING_PATH,
    **embed_kwargs: Any,
) -> list[str]:
    """Retrieve tools via SciToolKG-style retrieval with contextual reranking.

    1. Embed question → retrieve seed tools by functionality-triplet similarity.
    2. Expand each seed through KG edges.
    3. Drop paths with incompatible I/O direction patterns.
    4. Re-embed expanded candidates with seed context, multiply by seed
       similarity, sum repeated candidate scores, return top tools.
    """
    cache = _load_query_cache(query_cache_path)
    if question in cache:
        q_emb = cache[question]
    else:
        raise RuntimeError(
            "Missing benchmark query embedding. Run precache first; "
            f"question: {question[:200]}"
        )
    return retrieve_tools_kg_from_embedding(
        q_emb,
        tools,
        tool_names,
        tool_embeddings,
        graph_dict,
        top_k=top_k,
        expand_hops=expand_hops,
        max_tools=max_tools,
        context_embedding_path=context_embedding_path,
    )


def retrieve_tools_kg_from_embedding(
    query_embedding: np.ndarray,
    tools: dict[str, ToolNode],
    tool_names: list[str],
    tool_embeddings: np.ndarray,
    graph_dict: dict[str, list[list[str]]],
    *,
    top_k: int = 5,
    expand_hops: int = 3,
    max_tools: int = 10,
    context_embedding_path: Path = CONTEXT_EMBEDDING_PATH,
) -> list[str]:
    """Run SciToolKG retrieval from an already encoded query."""

    q_emb = np.asarray(query_embedding, dtype=np.float32)
    if q_emb.shape[-1] != tool_embeddings.shape[1]:
        raise RuntimeError(
            "Query embedding dim != tool embedding dim. "
            "Rebuild both caches with the same model."
        )
    sims = _cosine_similarity(q_emb, tool_embeddings)

    top_indices = np.argsort(sims)[::-1][:top_k]
    seed_tools = [(tool_names[i], float(sims[i])) for i in top_indices]

    scores: dict[str, float] = {}
    for seed_name, seed_score in seed_tools:
        scores[seed_name] = scores.get(seed_name, 0.0) + seed_score
        if expand_hops < 1:
            continue

        paths_by_tool = _collect_tool_paths(
            graph_dict, tools, start=seed_name,
            max_depth=expand_hops, edge_types=KG_EDGE_TYPES,
        )
        contextual_texts: list[str] = []
        contextual_targets: list[str] = []
        for candidate_name, paths in paths_by_tool.items():
            if candidate_name == seed_name or not _has_compatible_path(paths):
                continue
            contextual_texts.append(
                _contextual_functionality_text(
                    candidate=tools[candidate_name], seed=tools[seed_name],
                )
            )
            contextual_targets.append(candidate_name)

        if contextual_texts:
            contextual_embeddings = _embed_texts_cached(
                contextual_texts,
                cache_path=context_embedding_path,
            )
            contextual_sims = _cosine_similarity(q_emb, contextual_embeddings)
            rerank_indices = np.argsort(contextual_sims)[::-1][:top_k]
            for idx in rerank_indices:
                candidate_name = contextual_targets[int(idx)]
                candidate_score = contextual_sims[int(idx)]
                scores[candidate_name] = (
                    scores.get(candidate_name, 0.0)
                    + float(candidate_score) * seed_score
                )

    ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
    return ranked[:max_tools]


# ---------------------------------------------------------------------------
# Simple KG retrieval (no contextual reranking)
# ---------------------------------------------------------------------------

def retrieve_tools_simple(
    question: str,
    tools: dict[str, ToolNode],
    tool_names: list[str],
    tool_embeddings: np.ndarray,
    graph_dict: dict[str, list[list[str]]],
    *,
    top_k: int = 5,
    expand_hops: int = 1,
    max_tools: int = 10,
    query_cache_path: Path = QUERY_CACHE_PATH,
) -> list[str]:
    """Retrieve tools via embedding similarity + BFS expansion (no reranking)."""
    cache = _load_query_cache(query_cache_path)
    if question in cache:
        q_emb = cache[question]
    else:
        raise RuntimeError(
            "Missing benchmark query embedding. Run precache first; "
            f"question: {question[:200]}"
        )
    sims = _cosine_similarity(q_emb, tool_embeddings)

    name_to_idx = {n: i for i, n in enumerate(tool_names)}
    top_indices = np.argsort(sims)[::-1][:top_k]
    seed_tools = [tool_names[i] for i in top_indices]

    candidates: dict[str, float] = {}
    for name in seed_tools:
        candidates[name] = max(candidates.get(name, -1.0), float(sims[name_to_idx[name]]))

    if expand_hops >= 1:
        io_edges = {"inputs", "outputs", "is the input of", "is the output of"}
        frontier: set[str] = set(seed_tools)
        visited: set[str] = set(frontier)
        decay = 1.0
        for _ in range(expand_hops):
            decay *= 0.8
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in graph_dict:
                    continue
                for rel in graph_dict[node]:
                    if rel[0] not in io_edges:
                        continue
                    neighbor = rel[1]
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if neighbor in tools and neighbor in name_to_idx:
                        score = float(sims[name_to_idx[neighbor]]) * decay
                        candidates[neighbor] = max(candidates.get(neighbor, -1.0), score)
                    next_frontier.add(neighbor)
            frontier = next_frontier

    ranked = sorted(candidates, key=lambda n: candidates[n], reverse=True)
    return ranked[:max_tools]


# ---------------------------------------------------------------------------
# KG path utilities
# ---------------------------------------------------------------------------

def _collect_tool_paths(
    graph_dict: dict[str, list[list[str]]],
    tools: dict[str, ToolNode],
    *,
    start: str,
    max_depth: int,
    edge_types: set[str],
) -> dict[str, list[list[tuple[str, str, str]]]]:
    """Collect tool-to-tool paths in the KG neighborhood of ``start``."""
    paths_by_tool: dict[str, list[list[tuple[str, str, str]]]] = {}
    frontier: list[tuple[str, list[tuple[str, str, str]]]] = [(start, [])]

    for _ in range(max_depth):
        next_frontier: list[tuple[str, list[tuple[str, str, str]]]] = []
        for node, path in frontier:
            for pred, obj in graph_dict.get(node, []):
                if pred not in edge_types:
                    continue
                new_path = path + [(node, pred, obj)]
                if obj in tools and obj != start:
                    paths_by_tool.setdefault(obj, []).append(new_path)
                next_frontier.append((obj, new_path))
        frontier = next_frontier
        if not frontier:
            break
    return paths_by_tool


def _has_compatible_path(paths: list[list[tuple[str, str, str]]]) -> bool:
    return any(not _contains_io_mismatch_pattern(path) for path in paths)


def _contains_io_mismatch_pattern(path: list[tuple[str, str, str]]) -> bool:
    predicates = [pred for _, pred, _ in path]
    for left, right in zip(predicates, predicates[1:]):
        if (left == "is the input of" and right == "inputs") or (
            left == "outputs" and right == "is the output of"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Tool description formatting
# ---------------------------------------------------------------------------

def format_tool_descriptions(
    retrieved_names: list[str],
    tools: dict[str, ToolNode],
    tool_metadata: dict[str, str] | None = None,
) -> str:
    lines: list[str] = []
    for name in retrieved_names:
        if tool_metadata and name in tool_metadata:
            lines.append(f"- {name}: {tool_metadata[name]}")
        elif name in tools:
            t = tools[name]
            desc = (
                f"[Category]: {t.category}\n"
                f"  [Functionality]: {t.functionality}\n"
                f"  [Inputs]: {', '.join(t.inputs)}\n"
                f"  [Outputs]: {', '.join(t.outputs)}"
            )
            lines.append(f"- {name}:\n  {desc}")
    return "\n".join(lines)


def format_tool_descriptions_oneline(tools: dict[str, ToolNode]) -> str:
    lines: list[str] = []
    for name in sorted(tools):
        t = tools[name]
        inputs = ", ".join(t.inputs) or "?"
        outputs = ", ".join(t.outputs) or "?"
        lines.append(f"- {name}: {t.functionality} ({inputs} → {outputs})")
    return "\n".join(lines)
