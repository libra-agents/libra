#!/usr/bin/env python3
"""Convert AppWorld train/dev into the unified retrieval benchmark format.

The converted tool registry uses the common minimal schema:
    {name, category, function, signature, docstring}

For AppWorld, ``name`` is the benchmark-native API path ``app.api``.  We use
train/dev by default because those splits expose full ground truth, including
``required_apis``; AppWorld test splits expose evaluation code but not gold API
sets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_APPWORLD_ROOT = Path("benchmark_code/appworld")
DEFAULT_OUTPUT_DIR = Path("dataset and embedding/appworld")


def normalize_type(value: Any) -> str:
    if value is None:
        return "Any"
    value = str(value)
    return {
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "array",
        "object": "object",
        "string": "str",
    }.get(value.lower(), value)


def normalize_parameter_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "param"
    if value[0].isdigit():
        value = f"param_{value}"
    return value


def build_signature(parameters: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    required = [p for p in parameters if p.get("required")]
    optional = [p for p in parameters if not p.get("required")]
    for param in required + optional:
        name = normalize_parameter_name(param.get("name", "param"))
        if name in seen:
            continue
        seen.add(name)
        typ = normalize_type(param.get("type"))
        if param.get("required"):
            parts.append(f"{name}: {typ}")
        else:
            parts.append(f"{name}: {typ} = None")
    return f"({', '.join(parts)})"


def load_app_descriptions(appworld_root: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    apps_dir = appworld_root / "src" / "appworld" / "apps"
    for info_path in sorted(apps_dir.glob("*/info.toml")):
        app_name = info_path.parent.name
        text = info_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"(?m)^description\s*=\s*[\"'](.*?)[\"']\s*$", text)
        descriptions[app_name] = match.group(1) if match else ""
    return descriptions


def load_tool_registry(appworld_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_docs_dir = appworld_root / "data" / "api_docs" / "standard"
    app_descriptions = load_app_descriptions(appworld_root)
    registry: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    for docs_path in sorted(api_docs_dir.glob("*.json")):
        app_name = docs_path.stem
        api_name_to_doc = json.loads(docs_path.read_text(encoding="utf-8"))
        for api_name, doc in sorted(api_name_to_doc.items()):
            name = f"{app_name}.{api_name}"
            parameters = doc.get("parameters", []) or []
            description = str(doc.get("description") or "")
            doc_parts = []
            app_description = app_descriptions.get(app_name, "")
            if app_description:
                doc_parts.append(f"App: {app_description}")
            if description:
                doc_parts.append(f"API: {description}")
            response_success = (doc.get("response_schemas") or {}).get("success")
            if response_success:
                doc_parts.append(
                    "Success response schema: "
                    + json.dumps(response_success, ensure_ascii=False, sort_keys=True)
                )
            registry.append(
                {
                    "name": name,
                    "category": app_name,
                    "function": api_name,
                    "signature": build_signature(parameters),
                    "docstring": "\n".join(doc_parts),
                }
            )
            metadata[name] = {
                "app_name": app_name,
                "api_name": api_name,
                "path": doc.get("path"),
                "method": doc.get("method"),
                "parameters": parameters,
                "response_schemas": doc.get("response_schemas"),
            }
    return registry, metadata


def load_benchmark(appworld_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        split_path = appworld_root / "data" / "datasets" / f"{split}.txt"
        task_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for task_id in task_ids:
            task_dir = appworld_root / "data" / "tasks" / task_id
            ground_truth_dir = task_dir / "ground_truth"
            specs = json.loads((task_dir / "specs.json").read_text(encoding="utf-8"))
            required_apis_path = ground_truth_dir / "required_apis.json"
            required_apps_path = ground_truth_dir / "required_apps.json"
            if not required_apis_path.exists():
                raise FileNotFoundError(
                    f"{task_id} has no required_apis.json. "
                    "Use train/dev splits for retrieval conversion."
                )
            required_apis = json.loads(required_apis_path.read_text(encoding="utf-8"))
            required_apps = json.loads(required_apps_path.read_text(encoding="utf-8"))
            metadata = json.loads((ground_truth_dir / "metadata.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "id": task_id,
                    "dataset": f"appworld:{split}",
                    "split": split,
                    "question": specs["instruction"],
                    "tool_path": required_apis,
                    "required_apps": required_apps,
                    "metadata": {
                        "difficulty": metadata.get("difficulty"),
                        "num_apps": metadata.get("num_apps"),
                        "num_apis": metadata.get("num_apis"),
                        "num_api_calls": metadata.get("num_api_calls"),
                        "num_solution_code_lines": metadata.get("num_solution_code_lines"),
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(registry: list[dict[str, Any]], benchmark: list[dict[str, Any]]) -> dict[str, Any]:
    names = [tool["name"] for tool in registry]
    name_set = set(names)
    gold = sorted({tool for row in benchmark for tool in row["tool_path"]})
    missing = sorted(set(gold) - name_set)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    empty_signatures = [tool["name"] for tool in registry if tool.get("signature") == "()"]
    return {
        "num_tools": len(registry),
        "num_tasks": len(benchmark),
        "num_unique_gold_tools": len(gold),
        "missing_gold_tools": missing,
        "num_missing_gold_tools": len(missing),
        "duplicate_tool_names": duplicates,
        "num_duplicate_tool_names": len(duplicates),
        "max_gold_tools_per_task": max((len(row["tool_path"]) for row in benchmark), default=0),
        "num_empty_signatures": len(empty_signatures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, default=DEFAULT_APPWORLD_ROOT)
    parser.add_argument("--splits", default="train,dev", help="Comma-separated splits. Default: train,dev")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    registry, metadata = load_tool_registry(args.appworld_root)
    benchmark = load_benchmark(args.appworld_root, splits)
    report = validate(registry, benchmark)
    report["splits"] = splits

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tool_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "benchmark.jsonl", benchmark)
    (args.output_dir / "api_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_dir}")
    if report["num_missing_gold_tools"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
