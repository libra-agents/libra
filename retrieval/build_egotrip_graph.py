"""Build the offline knowledge graph for the Ego-Trip baseline.

Triples are extracted from public tool metadata with DeepSeek-V4-Flash and
entities are canonicalized by deterministic surface normalization. No
benchmark query or gold tool is read here.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from retrieval.data import load_tool_registry, qts_document


GRAPH_VERSION = "egotrip_adapted_graph_v1"
RELATIONS = (
    "has_line_of_business",
    "has_entity",
    "has_parameter",
    "contains",
    "related_to",
    "used_for",
    "has_attribute",
    "associated_with",
    "managed_by",
    "part_of",
    "required_for",
    "depends_on",
    "produces",
    "receives_from",
    "involved_in",
    "reports_to",
    "responsible_for",
    "affects",
    "includes",
)


def surface_key(text: str) -> str:
    """Apply only deterministic surface normalization, not semantic merging."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def signature_parameters(signature: str) -> list[str]:
    """Read parameter names from the registry's Python-style signature."""
    try:
        tree = ast.parse(f"def tool{signature}:\n    pass\n")
    except SyntaxError:
        return []
    function = tree.body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    arguments = function.args
    names = [arg.arg for arg in arguments.posonlyargs + arguments.args]
    names.extend(arg.arg for arg in arguments.kwonlyargs)
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


def scenario_metadata(tool: dict[str, Any]) -> str:
    """Map registry fields to the scenario metadata used by Ego-Trip.

    The triple extractor receives a scenario description and input
    parameters. Response schemas appended to registry docstrings are excluded.
    """
    docstring = str(tool.get("docstring") or "")
    for marker in ("Success response schema:", "Response schema:"):
        docstring = docstring.split(marker, 1)[0]
    parameters = signature_parameters(str(tool.get("signature") or ""))
    return "\n".join(
        [
            f"Scenario title: {tool['name']}",
            f"Line of business: {tool.get('category', '')}",
            f"Function: {tool.get('function', '')}",
            f"Description: {docstring.strip()}",
            f"Input parameters: {', '.join(parameters)}",
        ]
    )


def extraction_messages(tool: dict[str, Any]) -> list[dict[str, str]]:
    """Build the Ego-Trip triple extraction prompt."""
    system = (
        "Extract a compact knowledge graph from one tool description. Return "
        "only JSON with a 'relationships' array. Every item must contain the "
        "string fields 'head', 'relationship', and 'tail'. Use concise, "
        "consistent entity names; split compound claims; do not duplicate "
        "triples or create self-relations."
    )
    relation_list = ", ".join(RELATIONS)
    user = (
        f"The exact tool title is {json.dumps(str(tool['name']))}. Extract "
        "head-relation-tail triples from the metadata below. Use only these "
        f"relations when applicable: {relation_list}. Link the title to its "
        "category with has_line_of_business, to each input parameter with "
        "has_parameter, and to other salient concepts with has_entity. The "
        "exact tool title may occur only in those three relations. Preserve "
        "the exact title whenever it is a head.\n\n"
        f"{scenario_metadata(tool)}\n\n"
        "Output schema: "
        '{"relationships":[{"head":"...","relationship":"...","tail":"..."}]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_extraction(
    tool: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": extraction_messages(tool),
        "temperature": 0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(
                    f"{tool['name']}: incomplete response with "
                    f"finish_reason={choice.get('finish_reason')!r}"
                )
            content = str(choice["message"]["content"])
            parsed = json.loads(content)
            relationships = parsed.get("relationships")
            if not isinstance(relationships, list):
                raise ValueError("response has no relationships array")
            return {
                "tool": str(tool["name"]),
                "valid": True,
                "relationships": relationships,
                "raw_output": content,
            }
        except (requests.RequestException, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            if attempt == 2:
                return {
                    "tool": str(tool["name"]),
                    "valid": False,
                    "relationships": [],
                    "error": f"{type(error).__name__}: {error}",
                }
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("tool", ""))
            if not name:
                raise ValueError(f"{path}:{line_number}: checkpoint row has no tool")
            rows[name] = row
    return rows


def validated_relationships(tool: dict[str, Any], row: dict[str, Any]) -> list[dict[str, str]]:
    title = str(tool["name"])
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def append(head: str, relationship: str, tail: str) -> None:
        head, relationship, tail = head.strip(), relationship.strip(), tail.strip()
        if not head or not tail or relationship not in RELATIONS:
            return
        key = (surface_key(head), relationship, surface_key(tail))
        if key[0] == key[2] or key in seen:
            return
        seen.add(key)
        triples.append({"head": head, "relationship": relationship, "tail": tail})

    category = str(tool.get("category") or "").strip()
    if category:
        append(title, "has_line_of_business", category)
    for parameter in signature_parameters(str(tool.get("signature") or "")):
        append(title, "has_parameter", parameter)

    raw_relationships = row.get("relationships")
    if not isinstance(raw_relationships, list):
        raise ValueError(f"Extraction for {title!r} has no relationships list")
    for raw in raw_relationships:
        if not isinstance(raw, dict):
            continue
        append(
            str(raw.get("head", "")),
            str(raw.get("relationship", "")),
            str(raw.get("tail", "")),
        )
    return triples


def assemble_graph(
    registry: dict[str, dict[str, Any]],
    extractions: dict[str, dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    nodes: dict[str, dict[str, str]] = {}
    edges: dict[tuple[str, str, str], dict[str, str]] = {}
    tool_keys = {surface_key(name): name for name in registry}

    def add_node(label: str, node_type: str, text: str | None = None) -> str:
        key = surface_key(label)
        node_id = f"{node_type}:{key}"
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "text": text or label,
            }
        return node_id

    for name, tool in registry.items():
        add_node(name, "tool", qts_document(tool))

    for name, tool in registry.items():
        tool_id = f"tool:{surface_key(name)}"
        for triple in validated_relationships(tool, extractions[name]):
            endpoint_ids: list[str] = []
            for field in ("head", "tail"):
                label = triple[field]
                label_key = surface_key(label)
                if label_key == surface_key(name):
                    endpoint_ids.append(tool_id)
                elif label_key in tool_keys:
                    endpoint_ids.append(f"tool:{label_key}")
                else:
                    endpoint_ids.append(add_node(label, "entity"))
            source, target = endpoint_ids
            if source == target:
                continue
            edge_key = (source, triple["relationship"], target)
            edges[edge_key] = {
                "source": source,
                "relationship": triple["relationship"],
                "target": target,
                "source_tool": name,
            }

    return {
        "version": GRAPH_VERSION,
        "method": "Ego-Trip adapted to the released benchmark registries",
        "paper": "Bansal et al., 2025, Findings of IJCNLP-AACL",
        "adaptations": {
            "triple_extractor": f"{model} replaces GPT-4o",
            "canonicalization": "Unicode NFKC, case-folding, and whitespace only; no semantic merging",
            "tool_metadata": "canonical benchmark tool registry replaces the internal enterprise catalog",
        },
        "num_tools": len(registry),
        "invalid_extraction_count": sum(
            not bool(extractions[name].get("valid", True)) for name in registry
        ),
        "invalid_extraction_tools": [
            name for name in registry if not bool(extractions[name].get("valid", True))
        ],
        "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
        "edges": sorted(edges.values(), key=lambda item: (item["source"], item["relationship"], item["target"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-concurrent", type=int, default=12)
    args = parser.parse_args()
    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key environment variable {args.api_key_env}")

    registry = load_tool_registry(args.tools)
    completed = load_checkpoint(args.checkpoint)
    unknown = sorted(set(completed) - set(registry))
    if unknown:
        raise ValueError(f"Checkpoint contains tools absent from registry: {unknown[:5]}")
    pending = [tool for name, tool in registry.items() if name not in completed]
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    with args.checkpoint.open("a", encoding="utf-8") as checkpoint:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = {
                pool.submit(
                    request_extraction,
                    tool,
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                ): str(tool["name"])
                for tool in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                completed[str(row["tool"])] = row
                with write_lock:
                    checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                if index % 25 == 0 or index == len(pending):
                    print(f"extracted {index}/{len(pending)} pending tools", flush=True)

    graph = assemble_graph(registry, completed, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tools": graph["num_tools"],
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
