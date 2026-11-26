"""Build QTS, QTS-DE, and query embeddings through a local vLLM endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests

from retrieval.data import (
    load_benchmark,
    load_tool_registry,
    load_toolde_profiles,
    qwen3_query_text,
    qts_document,
    toolde_document,
)


EMBEDDING_VERSION = "qwen3_embedding_0.6b_v1"


def embed_batches(
    texts: list[str],
    *,
    base_url: str,
    model: str,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/embeddings",
            json={"model": model, "input": texts[start : start + batch_size]},
            timeout=300,
        )
        if not response.ok:
            raise RuntimeError(
                f"Embedding request failed: HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = sorted(response.json()["data"], key=lambda row: int(row["index"]))
        vectors.extend(
            np.asarray(row["embedding"], dtype=np.float32) for row in data
        )
    matrix = np.stack(vectors)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"Embedding endpoint returned invalid matrix {matrix.shape}")
    return matrix


def save_embeddings(
    path: Path,
    *,
    item_key: str,
    items: list[str],
    embeddings: np.ndarray,
    model: str,
    representation: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            item_key: np.asarray(items),
            "embeddings": embeddings,
            "version": np.asarray(EMBEDDING_VERSION),
            "model": np.asarray(model),
            "representation": np.asarray(representation),
        },
    )


def build_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    base_url: str,
    model: str,
    batch_size: int,
    representations: set[str],
) -> dict[str, object]:
    benchmark = load_benchmark(data_dir / "benchmark.jsonl")
    registry = load_tool_registry(data_dir / "tool_registry.json")
    missing_gold = sorted(
        {
            str(tool)
            for row in benchmark
            for tool in row["tool_path"]
            if str(tool) not in registry
        }
    )
    if missing_gold:
        raise ValueError(f"{data_dir}: gold tools missing from registry: {missing_gold[:5]}")
    statistics = {"queries": len(benchmark), "tools": len(registry)}

    names = list(registry)
    embedding_dimension = None
    if "qts" in representations:
        qts_embeddings = embed_batches(
            [qts_document(registry[name]) for name in names],
            base_url=base_url,
            model=model,
            batch_size=batch_size,
        )
        save_embeddings(
            output_dir / "tool_embeddings.npz",
            item_key="names",
            items=names,
            embeddings=qts_embeddings,
            model=model,
            representation="qts-full-document",
        )
        embedding_dimension = int(qts_embeddings.shape[1])

    if "toolde" in representations:
        profiles = load_toolde_profiles(data_dir / "toolde_profiles.jsonl")
        if set(profiles) != set(registry):
            raise ValueError(f"{data_dir}: Tool-DE profiles and registry names differ")
        toolde_embeddings = embed_batches(
            [toolde_document(registry[name], profiles[name]) for name in names],
            base_url=base_url,
            model=model,
            batch_size=batch_size,
        )
        save_embeddings(
            output_dir / "toolde_embeddings.npz",
            item_key="names",
            items=names,
            embeddings=toolde_embeddings,
            model=model,
            representation="qts-toolde-profile",
        )
        embedding_dimension = int(toolde_embeddings.shape[1])

    questions = [str(row["question"]) for row in benchmark]
    query_embeddings = embed_batches(
        [qwen3_query_text(question) for question in questions],
        base_url=base_url,
        model=model,
        batch_size=batch_size,
    )
    save_embeddings(
        output_dir / "query_embeddings.npz",
        item_key="questions",
        items=questions,
        embeddings=query_embeddings,
        model=model,
        representation="qwen3-instruction-query",
    )
    return {
        **statistics,
        "embedding_dimension": embedding_dimension,
        "representations": sorted(representations),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("appworld", "bfcl"),
        action="append",
    )
    parser.add_argument(
        "--representation",
        choices=("qts", "toolde"),
        action="append",
        help="Tool representation to embed; defaults to both.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8101")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    datasets = args.dataset or ["appworld", "bfcl"]
    representations = set(args.representation or ["qts", "toolde"])
    output = {}
    for dataset in datasets:
        output[dataset] = build_dataset(
            args.data_root / dataset,
            args.output_root / dataset,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
            representations=representations,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
