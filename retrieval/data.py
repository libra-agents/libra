"""Canonical data loading and document formatting for retrieval baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_QUERY_INSTRUCTION = (
    "Given a user task, retrieve tools that are necessary and appropriate "
    "to solve the task."
)


def qwen3_query_text(
    text: str,
    instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> str:
    """Format a retrieval query as prescribed by the Qwen3 embedding model."""
    return f"Instruct: {instruction}\nQuery: {text}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_benchmark(
    path: str | Path,
    *,
    question_key: str = "question",
    gold_key: str = "tool_path",
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in seen_ids:
            raise ValueError(f"{path}: duplicate benchmark id {row_id}")
        seen_ids.add(row_id)
        if question_key not in row or not isinstance(row[question_key], str):
            raise ValueError(f"{path}: id={row_id} has no string {question_key!r}")
        if gold_key not in row or not isinstance(row[gold_key], list):
            raise ValueError(f"{path}: id={row_id} has no list {gold_key!r}")
        if not row[gold_key] or not all(
            isinstance(name, str) and name for name in row[gold_key]
        ):
            raise ValueError(
                f"{path}: id={row_id} has an empty or non-string gold tool"
            )
    return rows


def load_tool_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: tool registry must be a JSON array")
    registry: dict[str, dict[str, Any]] = {}
    for index, tool in enumerate(payload):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError(f"{path}: tool {index} has no string name")
        name = str(tool["name"])
        if name in registry:
            raise ValueError(f"{path}: duplicate tool name {name!r}")
        registry[name] = tool
    return registry


def load_toolde_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        name = str(row.get("name"))
        expanded = row.get("expanded")
        if not isinstance(expanded, dict):
            raise ValueError(f"{path}: profile {name!r} has no expanded object")
        if name in profiles:
            raise ValueError(f"{path}: duplicate Tool-DE profile {name!r}")
        profiles[name] = expanded
    return profiles


def qts_document(tool: dict[str, Any]) -> str:
    """Canonical full-document representation used by QTS, BM25, and TF-IDF."""
    name = str(tool["name"])
    function = str(tool.get("function") or name)
    parts = [
        f"Tool: {name}",
        f"Function: {function}",
        f"Category: {tool.get('category', '')}",
        f"Signature: {function}{tool.get('signature', '')}",
    ]
    docstring = str(tool.get("docstring") or "").strip()
    if docstring:
        parts.append(f"Docstring:\n{docstring}")
    return "\n".join(parts)


TOOLDE_FIELDS = ("function_description", "when_to_use", "limitations", "tags")


def toolde_document(
    tool: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    """Canonical Tool-DE profile representation."""
    name = str(tool["name"])
    signature = str(tool.get("signature") or "")
    category = str(tool.get("category") or "")
    docstring = str(tool.get("docstring") or "").strip()
    parts = [f"{name}{signature}\nCategory: {category}\n{docstring}".strip()]
    for field in TOOLDE_FIELDS:
        value = profile.get(field)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if value:
            parts.append(f"{field.replace('_', ' ')}: {value}")
    return "\n".join(parts)


def validate_dataset(data_dir: str | Path) -> dict[str, int]:
    data_dir = Path(data_dir)
    benchmark = load_benchmark(data_dir / "benchmark.jsonl")
    registry = load_tool_registry(data_dir / "tool_registry.json")
    profiles = load_toolde_profiles(data_dir / "toolde_profiles.jsonl")
    graph = json.loads((data_dir / "tgr_graph.json").read_text(encoding="utf-8"))

    names = set(registry)
    if set(profiles) != names:
        raise ValueError(
            f"{data_dir}: Tool-DE profiles and registry names differ"
        )
    missing_gold = sorted(
        {
            str(name)
            for row in benchmark
            for name in row["tool_path"]
            if str(name) not in names
        }
    )
    if missing_gold:
        raise ValueError(f"{data_dir}: gold tools missing from registry: {missing_gold[:5]}")
    edge_names = {
        str(name)
        for edge in graph.get("undirected_edges", [])
        for name in (edge.get("tool_a"), edge.get("tool_b"))
    }
    missing_edge_names = sorted(edge_names - names)
    if missing_edge_names:
        raise ValueError(
            f"{data_dir}: graph tools missing from registry: {missing_edge_names[:5]}"
        )
    return {
        "queries": len(benchmark),
        "tools": len(registry),
        "toolde_profiles": len(profiles),
        "tgr_edges": len(graph.get("undirected_edges", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate released retrieval data.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    output = {
        dataset: validate_dataset(args.data_root / dataset)
        for dataset in ("appworld", "bfcl")
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
