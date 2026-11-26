"""Run ToolReAGt agentic retrieval over a benchmark."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any

from agentic_retrieval.backends import (
    EgoTripSearchBackend,
    LibraSearchBackend,
    SciToolKGSearchBackend,
    SearchBackend,
    SparseSearchBackend,
    ToolSearchBackend,
)
from agentic_retrieval.clients import (
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    VllmQwenRerankerClient,
)
from agentic_retrieval.runner import (
    CHECKPOINTS,
    PROMPT_CONDITIONS,
    RETRIEVER_TOOL_NAME,
    ToolReAGtRetriever,
)
from retrieval.data import load_benchmark
from retrieval.metrics import evaluate_retrieval


DATASET_DIRS = {
    "appworld": "appworld",
    "bfcl": "bfcl",
    "scitool": "scitool_clean",
}
DATA_PATHS = {
    "appworld": Path("data/appworld"),
    "bfcl": Path("data/bfcl"),
    "scitool": Path("dataset and embedding/scitool_clean"),
}
PRIMARY_K = {"appworld": 15, "bfcl": 15, "scitool": 15}
METHODS = (
    "bm25",
    "tfidf",
    "qts",
    "qts_de",
    "qts_tgr",
    "egotrip",
    "scitoolkg",
    "libra",
)


def _paths(
    root: Path,
    dataset: str,
    representation: str,
    data_root: Path | None = None,
    embedding_root: Path | None = None,
    toolde_embedding_root: Path | None = None,
) -> dict[str, Path | None]:
    data = (
        root / DATA_PATHS[dataset]
        if dataset == "scitool"
        else (data_root or root / "data") / dataset
    )
    embedding_dir = (
        embedding_root / DATASET_DIRS[dataset]
        if embedding_root is not None
        else data
    )
    toolde_embedding_dir = (
        toolde_embedding_root / DATASET_DIRS[dataset]
        if toolde_embedding_root is not None
        else embedding_dir
    )
    return {
        "registry": data / "tool_registry.json",
        "benchmark": data / "benchmark.jsonl",
        "embeddings": (
            toolde_embedding_dir / "toolde_embeddings.npz"
            if representation == "qts_de"
            else embedding_dir / "tool_embeddings.npz"
        ),
        "profiles": (
            data / "toolde_profiles.jsonl"
            if representation == "qts_de"
            else None
        ),
        "graph": (
            data / ("tgr_llm_graph.json" if dataset == "scitool" else "tgr_graph.json")
            if representation == "qts_tgr"
            else None
        ),
    }


def _libra_spec(root: Path, dataset: str):
    from retrieval.core.retrieval import load_graph

    method_data = root / "dataset and embedding" / DATASET_DIRS[dataset]
    fusion = "mean" if dataset == "bfcl" else "sum"
    return load_graph(method_data / "dependency_graph.json"), fusion


def _backend(
    args: argparse.Namespace,
    *,
    method: str,
    paths: dict[str, Path | None],
    retrieval_k: int,
) -> SearchBackend:
    embedding_client = OpenAIEmbeddingClient(
        base_url=args.embedding_url,
        model=args.embedding_model,
    )
    if method in {"bm25", "tfidf"}:
        return SparseSearchBackend(
            method=method,
            registry_path=paths["registry"],
            result_k=retrieval_k,
        )
    if method in {"qts", "qts_de", "qts_tgr"}:
        if method == "qts_de" and not Path(paths["profiles"]).is_file():
            raise FileNotFoundError(
                f"QTS-DE profiles are unavailable for {args.dataset}: {paths['profiles']}"
            )
        return ToolSearchBackend(
            representation={"qts": "raw", "qts_de": "toolde", "qts_tgr": "tgr"}[method],
            registry_path=paths["registry"],
            embeddings_path=paths["embeddings"],
            profile_path=paths["profiles"],
            graph_path=paths["graph"],
            embedding_client=embedding_client,
            reranker=_reranker(args),
            recall_k=args.recall_k,
            result_k=retrieval_k,
        )
    if args.retrieval_mode != "no_rerank":
        raise ValueError(f"{method} currently defines a frozen no-rerank protocol only")
    if method == "egotrip":
        return EgoTripSearchBackend(
            registry_path=paths["registry"],
            tool_embeddings_path=paths["embeddings"],
            graph_path=args.egotrip_root / f"{args.dataset}_graph.json",
            node_embeddings_path=args.egotrip_root / f"{args.dataset}_node_embeddings.npz",
            embedding_client=embedding_client,
            result_k=retrieval_k,
        )
    if method == "libra":
        edges, consumer_fusion = _libra_spec(args.repo_root, args.dataset)
        return LibraSearchBackend(
            registry_path=paths["registry"],
            tool_embeddings_path=paths["embeddings"],
            edges=edges,
            embedding_client=embedding_client,
            consumer_fusion=consumer_fusion,
            result_k=retrieval_k,
        )
    if method == "scitoolkg":
        if args.dataset != "scitool":
            raise ValueError("SciToolKG is defined only for SciToolEval")
        return SciToolKGSearchBackend(
            registry_path=paths["registry"],
            kg_path=args.repo_root / "SciToolAgent-main/KG/storage_graph_large/graph_store.json",
            tool_embeddings_path=args.scikg_root / "tool_embeddings.npz",
            context_embeddings_path=args.scikg_root / "context_embeddings.npz",
            embedding_client=embedding_client,
            result_k=retrieval_k,
        )
    raise ValueError(f"Unknown method: {method}")


def _completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def _reranker(args: argparse.Namespace):
    if args.retrieval_mode == "no_rerank":
        return None
    return VllmQwenRerankerClient(
        base_url=args.reranker_url,
        model=args.reranker_model,
    )


def _run_task(
    row: dict[str, Any],
    *,
    backend: SearchBackend,
    chat_url: str,
    model: str,
    seed: int,
    prompt_condition: str,
) -> dict[str, Any]:
    agent = ToolReAGtRetriever(
        backend=backend,
        chat_client=OpenAIChatClient(base_url=chat_url, model=model),
        prompt_condition=prompt_condition,
    )
    trajectory = agent.run(
        str(row["question"]),
        seed=seed,
        subtask=row.get("subtask"),
        decomposition=row.get("decomposition"),
    )
    return {
        "id": row["id"],
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "question": row["question"],
        "gold": list(map(str, row["tool_path"])),
        "model": model,
        **trajectory.to_dict(),
    }


def _curve_metrics(rows: list[dict[str, Any]], result_k: int) -> dict[str, Any]:
    curves = {}
    for checkpoint in CHECKPOINTS:
        pairs = [
            (
                list(row["checkpoints"][str(checkpoint)]["ranking"]),
                list(row["gold"]),
            )
            for row in rows
        ]
        curves[str(checkpoint)] = evaluate_retrieval(pairs, ks=(result_k,))
    return curves


def _usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    total = {
        field: sum(int(row["token_usage"][field]) for row in rows)
        for field in token_fields
    }
    mean_per_task = {
        field: fmean(float(row["token_usage"][field]) for row in rows)
        for field in token_fields
    }
    request_fields = (
        "generation_requests",
        "embedding_requests",
        "reranker_requests",
    )
    return {
        "generation_tokens_total": total,
        "generation_tokens_mean_per_task": mean_per_task,
        "requests_total": {
            field: sum(int(row[field]) for row in rows)
            for field in request_fields
        },
        "requests_mean_per_task": {
            field: fmean(float(row[field]) for row in rows)
            for field in request_fields
        },
    }


def _checkpoint_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for checkpoint in CHECKPOINTS:
        states = [row["checkpoints"][str(checkpoint)] for row in rows]
        output[str(checkpoint)] = {
            "generation_tokens_total": {
                field: sum(
                    int(state["cumulative_token_usage"][field])
                    for state in states
                )
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "embedding_requests_total": sum(
                int(state["embedding_requests_cumulative"]) for state in states
            ),
            "reranker_requests_total": sum(
                int(state["reranker_requests_cumulative"]) for state in states
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_DIRS), required=True)
    parser.add_argument("--representation", choices=METHODS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument(
        "--prompt-condition",
        choices=PROMPT_CONDITIONS,
        default="plain_fulltask",
    )
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--metric-k", type=int)
    parser.add_argument(
        "--retrieval-mode",
        choices=["no_rerank", "rerank"],
        required=True,
    )
    parser.add_argument("--recall-k", type=int, default=100)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8101")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--reranker-url", default="http://127.0.0.1:8102")
    parser.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8100")
    parser.add_argument("--task-concurrency", type=int, default=16)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--toolde-embedding-root", type=Path)
    parser.add_argument("--egotrip-root", type=Path)
    parser.add_argument("--scikg-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.task_concurrency < 1:
        parser.error("--task-concurrency must be positive.")
    if args.prompt_condition != "plain_fulltask":
        parser.error(
            "The three converted benchmark files do not expose a shared official "
            "subtask/decomposition field; the main cross-benchmark protocol must "
            "use Appendix A plain_fulltask without deriving decomposition from gold."
        )

    metric_k = args.metric_k or PRIMARY_K[args.dataset]
    retrieval_k = args.retrieval_k
    if retrieval_k < metric_k:
        parser.error("--retrieval-k must be at least --metric-k")
    paths = _paths(
        args.repo_root,
        args.dataset,
        args.representation,
        args.data_root,
        args.embedding_root,
        args.toolde_embedding_root,
    )
    backend = _backend(
        args,
        method=args.representation,
        paths=paths,
        retrieval_k=retrieval_k,
    )
    benchmark = load_benchmark(paths["benchmark"])
    if args.limit:
        benchmark = benchmark[: args.limit]
    completed = _completed(args.output)
    pending = [row for row in benchmark if str(row["id"]) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.output.open("a", encoding="utf-8") as output,
        ThreadPoolExecutor(max_workers=args.task_concurrency) as executor,
    ):
        futures = {
            executor.submit(
                _run_task,
                row,
                backend=backend,
                chat_url=args.chat_url,
                model=args.model,
                seed=args.seed,
                prompt_condition=args.prompt_condition,
            ): row
            for row in pending
        }
        for done, future in enumerate(as_completed(futures), 1):
            saved = future.result()
            saved["backend"] = backend.describe()
            output.write(json.dumps(saved, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"completed {done}/{len(pending)} id={saved['id']} "
                f"calls={len(saved['retrieval_calls'])} "
                f"tokens={saved['token_usage']['total_tokens']}",
                flush=True,
            )

    rows_by_id = _completed(args.output)
    rows = [rows_by_id[str(row["id"])] for row in benchmark if str(row["id"]) in rows_by_id]
    metrics_path = args.output.with_suffix(".metrics.json")
    payload = {
        "dataset": args.dataset,
        "representation": args.representation,
        "method": "ToolReAGt adaptation",
        "retrieval_mode": args.retrieval_mode,
        "appendix_a_prompt_condition": args.prompt_condition,
        "source_framework": "llama-index-core==0.10.17",
        "paper_protocol": {
            "source": "Braunschweiler et al. (2025), ToolReAGt",
            "original_benchmark": "UltraTool",
            "react_prompt": "Appendix B",
            "tool_interface": "textual ReAct (no native function calling)",
            "retriever_tool_name": RETRIEVER_TOOL_NAME,
            "retriever_policy": "mandatory",
            "max_retrieval_calls": 10,
            "training": False,
            "domain_tool_execution": False,
            "reranker": (
                "none"
                if args.retrieval_mode == "no_rerank"
                else args.reranker_model
            ),
        },
        "declared_adaptations": {
            "benchmark": args.dataset,
            "tool_representation": args.representation,
            "generator": args.model,
            "retrieval_backend": backend.describe(),
            "embedding_model": args.embedding_model,
            "reranker_model": (
                None
                if args.retrieval_mode == "no_rerank"
                else args.reranker_model
            ),
            "retrieval_k": retrieval_k,
            "metric_k": metric_k,
        },
        "generation_config": {
            "source": (
                "llama-index-core==0.10.17 LlamaCPP defaults and "
                "llama-cpp-python sampling defaults"
            ),
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 256,
            "seed": args.seed,
            "qwen_enable_thinking": False,
            "stop": None,
            "parsed_finish_reasons": ["stop", "length"],
        },
        "model": args.model,
        "seed": args.seed,
        "max_retrieval_calls": 10,
        "checkpoints": list(CHECKPOINTS),
        "retrieval_k": retrieval_k,
        "metric_k": metric_k,
        "num_tasks": len(rows),
        "metrics_by_retrieval_call": _curve_metrics(rows, metric_k),
        "usage": _usage_summary(rows),
        "usage_by_retrieval_call": _checkpoint_usage(rows),
        "native_end_to_end_metrics": "N/A (retrieval-only baseline)",
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
