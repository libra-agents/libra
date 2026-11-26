"""Retrieval backends exposed to the ToolReAGt meta-tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from agentic_retrieval.clients import OpenAIEmbeddingClient
from retrieval.dense_retrieve_tool_embedding import (
    load_npz_embeddings,
    normalize_embedding_matrix,
    retrieve_normalized_embedding,
)
from retrieval.data import (
    load_tool_registry,
    load_toolde_profiles,
    qts_document,
    toolde_document,
)
from retrieval.bm25_retrieve_tool_text import BM25Index
from retrieval.core.retrieval import (
    DependencyEdge,
    place_graph_additions_after_seed,
    rank_query,
)
from retrieval.egotrip_retrieve import (
    load_graph as load_egotrip_graph,
    retrieve_candidates as retrieve_egotrip_candidates,
)
from retrieval.tfidf_retrieve_tool_text import TfidfIndex
from retrieval.tgr_retrieve_tool_embedding import (
    build_symmetric_adjacency,
    graph_encode_embeddings,
    load_graph,
)


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class SearchBackend(Protocol):
    tool_description: str
    embedding_requests_per_search: int
    reranker_requests_per_search: int

    def search(self, query: str) -> list["SearchHit"]: ...

    def describe(self) -> dict: ...


@dataclass(frozen=True)
class SearchHit:
    name: str
    document: str
    retrieval_score: float | None
    rerank_score: float | None
    dependencies: tuple[str, ...] = ()

    def observation(self) -> dict:
        row = {
            "name": self.name,
            "document": self.document,
        }
        if self.retrieval_score is not None:
            row["retrieval_score"] = self.retrieval_score
        if self.rerank_score is not None:
            row["rerank_score"] = self.rerank_score
        if self.dependencies:
            row["dependencies"] = list(self.dependencies)
        return row


def _load_documents(
    registry_path: str | Path,
    profile_path: str | Path | None,
) -> tuple[dict[str, dict], dict[str, str]]:
    registry = load_tool_registry(registry_path)
    profiles = load_toolde_profiles(profile_path) if profile_path else {}
    documents = {
        name: (
            toolde_document(tool, profiles[name])
            if profiles
            else qts_document(tool)
        )
        for name, tool in registry.items()
    }
    return registry, documents


def _graph_dependencies(graph_path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if graph_path is None:
        return {}
    graph = load_graph(graph_path)
    dependencies: dict[str, list[str]] = {}
    for edge in graph.get("directed_edges", []):
        consumer = str(edge["source"])
        prerequisite = str(edge["target"])
        dependencies.setdefault(consumer, []).append(prerequisite)
    return {
        name: tuple(sorted(set(values)))
        for name, values in dependencies.items()
    }


class ToolSearchBackend:
    """Dense tool search with an optional fixed Qwen3 reranker."""

    tool_description = (
        "Search the complete tool registry with a focused natural-language "
        "query and return the top ranked tool descriptions."
    )
    embedding_requests_per_search = 1

    def __init__(
        self,
        *,
        representation: str,
        registry_path: str | Path,
        embeddings_path: str | Path,
        embedding_client: OpenAIEmbeddingClient,
        reranker: Reranker | None = None,
        profile_path: str | Path | None = None,
        graph_path: str | Path | None = None,
        recall_k: int = 100,
        result_k: int = 15,
    ) -> None:
        if representation not in {"raw", "toolde", "tgr"}:
            raise ValueError(f"Unknown representation: {representation}")
        if reranker is not None and recall_k < result_k:
            raise ValueError(f"recall_k ({recall_k}) must be >= result_k ({result_k}).")
        if representation == "toolde" and profile_path is None:
            raise ValueError("Tool-DE requires profile_path.")
        if representation == "tgr" and graph_path is None:
            raise ValueError("TGR requires graph_path.")

        self.representation = representation
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.recall_k = recall_k
        self.result_k = result_k
        self.reranker_requests_per_search = int(reranker is not None)
        self.registry, self.documents = _load_documents(registry_path, profile_path)
        self.names, embeddings = load_npz_embeddings(embeddings_path, item_key="names")
        if set(self.names) != set(self.registry):
            missing_docs = sorted(set(self.names) - set(self.registry))
            missing_embeddings = sorted(set(self.registry) - set(self.names))
            raise ValueError(
                "Registry/embedding names differ: "
                f"missing_docs={missing_docs[:5]}, "
                f"missing_embeddings={missing_embeddings[:5]}"
            )

        if representation == "tgr":
            graph = load_graph(graph_path)
            adjacency, _ = build_symmetric_adjacency(
                graph=graph,
                embedding_names=self.names,
            )
            embeddings = graph_encode_embeddings(embeddings, adjacency)
        self.embeddings = normalize_embedding_matrix(embeddings)
        self.dependencies = _graph_dependencies(
            graph_path if representation == "tgr" else None
        )

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def search(self, query: str) -> list[SearchHit]:
        query_embedding = self.embedding_client.embed(query)
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                "Query/document embedding dimension mismatch: "
                f"query={query_embedding.shape[0]}, documents={self.dimension}. "
                "The configured local query encoder cannot be mixed with this NPZ."
            )
        retrieved = retrieve_normalized_embedding(
            query_embedding,
            self.names,
            self.embeddings,
            top_k=min(
                self.recall_k if self.reranker is not None else self.result_k,
                len(self.names),
            ),
        )
        if self.reranker is None:
            return [
                SearchHit(
                    name=name,
                    document=self.documents[name],
                    retrieval_score=float(score),
                    rerank_score=None,
                    dependencies=self.dependencies.get(name, ()),
                )
                for name, score in retrieved[: self.result_k]
            ]

        names = [name for name, _ in retrieved]
        documents = [self.documents[name] for name in names]
        rerank_scores = self.reranker.score(query, documents)
        ranked = sorted(
            zip(names, documents, retrieved, rerank_scores),
            key=lambda row: (-row[3], -row[2][1], row[0]),
        )
        return [
            SearchHit(
                name=name,
                document=document,
                retrieval_score=float(retrieved[1]),
                rerank_score=float(rerank_score),
                dependencies=self.dependencies.get(name, ()),
            )
            for name, document, retrieved, rerank_score in ranked[: self.result_k]
        ]

    def describe(self) -> dict:
        description = {
            "backend": (
                "dense_recall_qwen_rerank"
                if self.reranker is not None
                else "dense_cosine"
            ),
            "representation": self.representation,
            "num_tools": len(self.names),
            "embedding_dimension": self.dimension,
            "result_k": self.result_k,
        }
        if self.reranker is not None:
            description["recall_k"] = self.recall_k
        return description


class SparseSearchBackend:
    """BM25 or TF-IDF over the same QTS documents used by static retrieval."""

    tool_description = (
        "Search the complete tool registry with a focused natural-language "
        "query and return the top ranked tool descriptions."
    )
    embedding_requests_per_search = 0
    reranker_requests_per_search = 0

    def __init__(
        self,
        *,
        method: str,
        registry_path: str | Path,
        result_k: int = 20,
    ) -> None:
        if method not in {"bm25", "tfidf"}:
            raise ValueError(f"Unknown sparse method: {method}")
        self.method = method
        self.result_k = result_k
        self.registry, self.documents = _load_documents(registry_path, None)
        self.names = list(self.registry)
        documents = [self.documents[name] for name in self.names]
        self.index = BM25Index(documents) if method == "bm25" else TfidfIndex(documents)

    def search(self, query: str) -> list[SearchHit]:
        scores = self.index.score(query)
        order = sorted(
            range(len(self.names)),
            key=lambda index: (-float(scores[index]), self.names[index]),
        )[: self.result_k]
        return [
            SearchHit(
                name=self.names[index],
                document=self.documents[self.names[index]],
                retrieval_score=float(scores[index]),
                rerank_score=None,
            )
            for index in order
        ]

    def describe(self) -> dict:
        return {
            "backend": self.method,
            "representation": "raw",
            "num_tools": len(self.names),
            "result_k": self.result_k,
        }


class EgoTripSearchBackend:
    """Published Ego-Trip entry matching and one-hop expansion, QTS-ranked."""

    tool_description = (
        "Search the tool knowledge graph with a focused natural-language query "
        "and return tools from the matched ego graphs."
    )
    embedding_requests_per_search = 1
    reranker_requests_per_search = 0

    def __init__(
        self,
        *,
        registry_path: str | Path,
        tool_embeddings_path: str | Path,
        graph_path: str | Path,
        node_embeddings_path: str | Path,
        embedding_client: OpenAIEmbeddingClient,
        semantic_entry_k: int = 10,
        result_k: int = 20,
    ) -> None:
        self.registry, self.documents = _load_documents(registry_path, None)
        self.tool_names, tool_embeddings = load_npz_embeddings(
            tool_embeddings_path, item_key="names"
        )
        if set(self.tool_names) != set(self.registry):
            raise ValueError("Ego-Trip tool embeddings and registry differ")
        self.normalized_tools = normalize_embedding_matrix(tool_embeddings)
        self.tool_index = {name: index for index, name in enumerate(self.tool_names)}
        self.graph = load_egotrip_graph(graph_path)
        node_ids, node_embeddings = load_npz_embeddings(
            node_embeddings_path, item_key="node_ids"
        )
        expected_node_ids = [str(node["id"]) for node in self.graph["nodes"]]
        if node_ids != expected_node_ids:
            raise ValueError("Ego-Trip node embeddings are not graph-aligned")
        self.normalized_nodes = normalize_embedding_matrix(node_embeddings)
        self.embedding_client = embedding_client
        self.semantic_entry_k = semantic_entry_k
        self.result_k = result_k

    def search(self, query: str) -> list[SearchHit]:
        query_embedding = self.embedding_client.embed(query)
        if query_embedding.shape[0] != self.normalized_tools.shape[1]:
            raise ValueError("Ego-Trip query and tool embedding dimensions differ")
        candidates, _, _ = retrieve_egotrip_candidates(
            question=query,
            query_embedding=query_embedding,
            graph=self.graph,
            normalized_node_embeddings=self.normalized_nodes,
            semantic_entry_k=self.semantic_entry_k,
        )
        normalized_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        scored = [
            (name, float(self.normalized_tools[self.tool_index[name]] @ normalized_query))
            for name in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchHit(
                name=name,
                document=self.documents[name],
                retrieval_score=score,
                rerank_score=None,
            )
            for name, score in scored[: self.result_k]
        ]

    def describe(self) -> dict:
        return {
            "backend": "egotrip_one_hop_qts_ranked",
            "representation": "raw",
            "num_tools": len(self.tool_names),
            "semantic_entry_k": self.semantic_entry_k,
            "result_k": self.result_k,
        }


class LibraSearchBackend:
    """Frozen LIBRA graph retrieval for arbitrary agent-generated queries."""

    tool_description = (
        "Search an execution-validated dependency library with a focused "
        "natural-language query and return a ranked tool list."
    )
    embedding_requests_per_search = 1
    reranker_requests_per_search = 0

    def __init__(
        self,
        *,
        registry_path: str | Path,
        tool_embeddings_path: str | Path,
        edges: Sequence[DependencyEdge],
        embedding_client: OpenAIEmbeddingClient,
        consumer_fusion: str,
        seed_k: int = 15,
        result_k: int = 20,
    ) -> None:
        self.registry, self.documents = _load_documents(registry_path, None)
        self.names, embeddings = load_npz_embeddings(
            tool_embeddings_path, item_key="names"
        )
        if set(self.names) != set(self.registry):
            raise ValueError("LIBRA tool embeddings and registry differ")
        self.normalized_tools = normalize_embedding_matrix(embeddings)
        self.edges = list(edges)
        self.embedding_client = embedding_client
        self.consumer_fusion = consumer_fusion
        self.seed_k = seed_k
        self.result_k = result_k

    def search(self, query: str) -> list[SearchHit]:
        query_embedding = self.embedding_client.embed(query)
        if query_embedding.shape[0] != self.normalized_tools.shape[1]:
            raise ValueError("LIBRA query and tool embedding dimensions differ")
        normalized_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        dense_scores = self.normalized_tools @ normalized_query
        order, scores, propagated = rank_query(
            tool_names=self.names,
            dense_scores=dense_scores,
            edges=self.edges,
            seed_k=self.seed_k,
            top_k=self.result_k,
            consumer_fusion=self.consumer_fusion,
        )
        dense_indices = np.argsort(dense_scores)[::-1][: self.seed_k]
        order = place_graph_additions_after_seed(
            order,
            dense_seed=[self.names[int(index)] for index in dense_indices],
            cutoff=min(15, self.result_k),
        )
        dependencies = {
            name: tuple(map(str, details.get("consumers", ())))
            for name, details in propagated.items()
        }
        return [
            SearchHit(
                name=name,
                document=self.documents[name],
                retrieval_score=float(scores[name]),
                rerank_score=None,
                dependencies=dependencies.get(name, ()),
            )
            for name in order
        ]

    def describe(self) -> dict:
        return {
            "backend": "libra_execution_validated_graph",
            "representation": "raw",
            "num_tools": len(self.names),
            "dependency_edges": len(self.edges),
            "consumer_fusion": self.consumer_fusion,
            "seed_k": self.seed_k,
            "result_k": self.result_k,
        }


class SciToolKGSearchBackend:
    """SciToolAgent KG retrieval with Qwen3 embeddings."""

    tool_description = (
        "Search the SciTool knowledge graph with a focused natural-language "
        "query and return the top ranked scientific tools."
    )
    embedding_requests_per_search = 1
    reranker_requests_per_search = 0

    def __init__(
        self,
        *,
        registry_path: str | Path,
        kg_path: str | Path,
        tool_embeddings_path: str | Path,
        context_embeddings_path: str | Path,
        embedding_client: OpenAIEmbeddingClient,
        result_k: int = 20,
    ) -> None:
        from retrieval.dense_retrieve_knowledge_graph import (
            load_context_embedding_cache,
            load_kg_edges,
            load_kg_tools,
            restrict_kg_to_catalog,
        )

        self.registry, self.documents = _load_documents(registry_path, None)
        tools = load_kg_tools(Path(kg_path))
        graph = load_kg_edges(Path(kg_path))
        self.tools, self.graph = restrict_kg_to_catalog(
            tools, graph, set(self.registry)
        )
        self.names, self.embeddings = load_npz_embeddings(
            tool_embeddings_path, item_key="names"
        )
        if self.names != sorted(self.tools):
            raise ValueError("SciToolKG tool embeddings are not catalog-aligned")
        self.context_embeddings_path = Path(context_embeddings_path)
        load_context_embedding_cache(self.context_embeddings_path)
        self.embedding_client = embedding_client
        self.result_k = result_k

    def search(self, query: str) -> list[SearchHit]:
        from retrieval.dense_retrieve_knowledge_graph import (
            retrieve_tools_kg_from_embedding,
        )

        query_embedding = self.embedding_client.embed(query)
        ranked = retrieve_tools_kg_from_embedding(
            query_embedding,
            self.tools,
            self.names,
            self.embeddings,
            self.graph,
            top_k=5,
            expand_hops=3,
            max_tools=self.result_k,
            context_embedding_path=self.context_embeddings_path,
        )
        return [
            SearchHit(
                name=name,
                document=self.documents[name],
                retrieval_score=None,
                rerank_score=None,
            )
            for name in ranked
        ]

    def describe(self) -> dict:
        return {
            "backend": "scitoolkg_official_qwen_adaptation",
            "representation": "raw",
            "num_tools": len(self.names),
            "seed_k": 5,
            "expand_hops": 3,
            "result_k": self.result_k,
        }
