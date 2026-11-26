"""Run LIBRA dependency retrieval for one benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from retrieval.core.retrieval import run_dependency_retrieval


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset and embedding"
BENCHMARKS = {
    "appworld": ("appworld", "sum", "appworld_required_input"),
    "scitool": ("scitool_clean", "sum", "scitool_optional_dataflow"),
    "bfcl": ("bfcl", "mean", "bfcl_state_and_contract"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=tuple(BENCHMARKS))
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(os.getenv(
            "SKILL_MINING_EMBEDDING_ROOT",
            ROOT / "embeddings" / "qwen3_embedding_0.6b_v1",
        )),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory, fusion, adapter = BENCHMARKS[args.benchmark]
    data = DATA / directory
    embeddings = args.embedding_root / directory
    metrics, rows = run_dependency_retrieval(
        benchmark_path=data / "benchmark.jsonl",
        tool_embeddings_path=embeddings / "tool_embeddings.npz",
        query_embeddings_path=embeddings / "query_embeddings.npz",
        graph_path=data / "dependency_graph.json",
        consumer_fusion=fusion,
    )
    metrics["adapter"] = adapter

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
