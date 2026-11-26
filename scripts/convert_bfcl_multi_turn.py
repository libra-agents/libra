"""Convert official BFCL v4 multi-turn base data to the retrieval schema."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


CLASS_FILES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "TwitterAPI": "posting_api",
    "VehicleControlAPI": "vehicle_control",
}
TYPE_NAMES = {
    "array": "list",
    "boolean": "bool",
    "dict": "dict",
    "float": "float",
    "integer": "int",
    "number": "float",
    "string": "str",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def type_name(schema: dict[str, Any]) -> str:
    raw = str(schema.get("type", "Any"))
    if raw == "array" and isinstance(schema.get("items"), dict):
        return f"list[{type_name(schema['items'])}]"
    return TYPE_NAMES.get(raw, raw)


def signature(parameters: dict[str, Any]) -> str:
    properties = parameters["properties"]
    required = set(map(str, parameters.get("required", [])))
    if required - set(properties):
        raise ValueError("Required BFCL parameter is missing its schema")
    names = [
        *[name for name in properties if name in required],
        *[name for name in properties if name not in required],
    ]
    rendered = []
    for name in names:
        schema = properties[name]
        item = f"{name}: {type_name(schema)}"
        if name not in required:
            default = schema.get("default")
            item += f" = {None if default == 'None' else default!r}"
        rendered.append(item)
    return f"({', '.join(rendered)})"


def docstring(tool: dict[str, Any]) -> str:
    parameter_descriptions = {
        name: schema["description"]
        for name, schema in tool["parameters"]["properties"].items()
        if schema.get("description")
    }
    parts = [str(tool["description"]).strip()]
    if parameter_descriptions:
        parts.append(
            "Parameter descriptions: "
            + json.dumps(parameter_descriptions, ensure_ascii=False, sort_keys=True)
        )
    parts.append(
        "Response schema: "
        + json.dumps(tool["response"], ensure_ascii=False, sort_keys=True)
    )
    return "\n".join(parts)


def build_registry(source_root: Path) -> list[dict[str, Any]]:
    registry = []
    for category, file_stem in CLASS_FILES.items():
        path = source_root / "multi_turn_func_doc" / f"{file_stem}.json"
        for source in read_jsonl(path):
            function = str(source["name"])
            registry.append(
                {
                    "name": f"{category}.{function}",
                    "category": category,
                    "function": function,
                    "signature": signature(source["parameters"]),
                    "docstring": docstring(source),
                    "parameters": source["parameters"],
                    "response": source["response"],
                }
            )
    names = [tool["name"] for tool in registry]
    if len(names) != 128 or len(names) != len(set(names)):
        raise ValueError("Expected 128 uniquely qualified BFCL tools")
    return registry


def build_benchmark(
    source_root: Path,
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qualified_by_function = {
        str(tool["function"]): str(tool["name"])
        for tool in registry
    }
    if len(qualified_by_function) != len(registry):
        raise ValueError("BFCL catalog contains ambiguous raw function names")
    answers = {
        str(row["id"]): row
        for row in read_jsonl(
            source_root / "possible_answer" / "BFCL_v4_multi_turn_base.json"
        )
    }
    rows = []
    for source in read_jsonl(source_root / "BFCL_v4_multi_turn_base.json"):
        user_messages = [
            str(message["content"]).strip()
            for turn in source["question"]
            for message in turn
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ]
        answer = answers[str(source["id"])]
        gold = []
        for turn in answer["ground_truth"]:
            for call_text in turn:
                expression = ast.parse(str(call_text), mode="eval").body
                if not isinstance(expression, ast.Call) or not isinstance(
                    expression.func, ast.Name
                ):
                    raise ValueError(
                        f"{source['id']}: unsupported official call {call_text!r}"
                    )
                function = expression.func.id
                if function not in qualified_by_function:
                    raise ValueError(
                        f"{source['id']}: official function {function!r} "
                        "is absent from the catalog"
                    )
                gold.append(qualified_by_function[function])
        rows.append(
            {
                "id": str(source["id"]),
                "dataset": "bfcl_v4:multi_turn_base",
                "split": "test",
                "question": "\n".join(user_messages),
                "tool_path": gold,
                "metadata": {
                    "num_turns": len(source["question"]),
                    "involved_classes": source["involved_classes"],
                    "excluded_function": source.get("excluded_function", []),
                    "declared_path": source["path"],
                },
            }
        )
    if len(rows) != 200:
        raise ValueError(f"Expected 200 BFCL tasks, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bfcl"))
    args = parser.parse_args()

    registry = build_registry(args.source_root)
    benchmark = build_benchmark(
        args.source_root,
        registry,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tool_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "benchmark.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark),
        encoding="utf-8",
    )
    print(json.dumps({"queries": len(benchmark), "tools": len(registry)}, indent=2))


if __name__ == "__main__":
    main()
