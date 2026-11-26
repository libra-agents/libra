#!/usr/bin/env python3
"""Aggregate agentic retrieval quality and token consumption across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from math import sqrt
from statistics import fmean, stdev
from typing import Any


METRICS = ("Precision", "Recall", "NDCG", "Completeness")


def mean_std_error(values: list[float]) -> tuple[float, float, float]:
    """Return the mean, sample standard deviation, and standard error."""
    mean = fmean(values)
    sample_std = stdev(values) if len(values) > 1 else 0.0
    standard_error = sample_std / sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sample_std, standard_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.input_root.glob("*.metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_k = int(payload["metric_k"])
        num_tasks = int(payload["num_tasks"])
        for checkpoint, metrics in payload["metrics_by_retrieval_call"].items():
            usage = payload["usage_by_retrieval_call"][checkpoint]
            groups[(
                str(payload["dataset"]),
                str(payload["representation"]),
                str(payload["model"]),
                int(checkpoint),
            )].append({
                "seed": int(payload["seed"]),
                "num_tasks": num_tasks,
                "metric_k": metric_k,
                "metrics": metrics,
                "prompt_tokens_per_query": (
                    usage["generation_tokens_total"]["prompt_tokens"] / num_tasks
                ),
                "completion_tokens_per_query": (
                    usage["generation_tokens_total"]["completion_tokens"] / num_tasks
                ),
                "total_tokens_per_query": (
                    usage["generation_tokens_total"]["total_tokens"] / num_tasks
                ),
                "embedding_requests_per_query": (
                    usage["embedding_requests_total"] / num_tasks
                ),
            })

    rows: list[dict[str, Any]] = []
    for (dataset, method, model, checkpoint), runs in sorted(groups.items()):
        metric_k = runs[0]["metric_k"]
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "model": model,
            "checkpoint": checkpoint,
            "metric_k": metric_k,
            "num_seeds": len(runs),
            "seeds": sorted(run["seed"] for run in runs),
            "num_tasks_per_seed": runs[0]["num_tasks"],
        }
        for metric in METRICS:
            values = [float(run["metrics"][f"{metric}@{metric_k}"]) for run in runs]
            mean, std, standard_error = mean_std_error(values)
            row[f"{metric.lower()}_mean"] = mean
            row[f"{metric.lower()}_std"] = std
            row[f"{metric.lower()}_se"] = standard_error
            row[f"{metric.lower()}_percent_mean"] = 100.0 * mean
            row[f"{metric.lower()}_percent_se"] = 100.0 * standard_error
        for field in (
            "prompt_tokens_per_query",
            "completion_tokens_per_query",
            "total_tokens_per_query",
            "embedding_requests_per_query",
        ):
            mean, std, standard_error = mean_std_error(
                [float(run[field]) for run in runs]
            )
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
            row[f"{field}_se"] = standard_error
        rows.append(row)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[key for key in rows[0] if key != "seeds"] if rows else [],
        )
        if rows:
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != "seeds"} for row in rows)
    print(f"aggregated {len(rows)} cells from {len(list(args.input_root.glob('*.metrics.json')))} runs")


if __name__ == "__main__":
    main()
