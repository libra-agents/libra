"""Build a TGR tool dependency graph with an LLM pair judge.

Each tool pair is classified by an OpenAI-compatible chat completion call.

Edge convention:
    directed edge source -> target means source depends on target.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

import requests


GRAPH_VERSION = "tgr_llm_graph_v1"
VALID_LABELS = {"a_depends_on_b", "b_depends_on_a", "no_dependency"}


def load_tool_registry(path: str | Path) -> list[dict[str, str]]:
    """Load supported registry schemas into canonical tool records.

    Internal id must be the exact name used by tool_embeddings.npz and
    benchmark.jsonl tool_path.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: list[dict[str, str]] = []

    if isinstance(data, list):
        iterable = [(None, item) for item in data]
    elif isinstance(data, dict):
        iterable = list(data.items())
    else:
        raise TypeError(f"Unsupported registry root type: {type(data).__name__}")

    for key, raw in iterable:
        if not isinstance(raw, dict):
            raw = {"docstring": str(raw)}
        tool_id = str(raw.get("id") or raw.get("name") or key or "").strip()
        if not tool_id:
            raise ValueError(f"Tool registry row has no id/name: {raw}")
        name = str(raw.get("name") or tool_id)
        function = str(raw.get("function") or raw.get("api_name") or raw.get("api") or "")
        category = str(raw.get("category") or raw.get("tool_name") or raw.get("app") or "")
        signature = str(raw.get("signature") or raw.get("parameters") or raw.get("schema") or "")
        docstring = str(
            raw.get("docstring")
            or raw.get("description")
            or raw.get("summary")
            or raw.get("api_description")
            or ""
        )
        tools.append({
            "id": tool_id,
            "name": name,
            "function": function,
            "category": category,
            "signature": signature,
            "docstring": docstring,
        })

    ids = [tool["id"] for tool in tools]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tool ids in registry: {duplicates[:10]}")
    return tools


def canonical_tool_text(tool: dict[str, str]) -> str:
    return "\n".join([
        f"Tool ID: {tool.get('id', '')}",
        f"Signature: {tool.get('signature', '')}",
        "Docstring:",
        str(tool.get("docstring", "")),
    ])


def build_prompt(tool_a: dict[str, str], tool_b: dict[str, str]) -> list[dict[str, str]]:
    system = (
        "You classify the data-flow dependency between two tools. Judge ONLY "
        "from their I/O schema (the signature gives REQUIRED INPUTS; the "
        "docstring 'Returns'/functionality gives the OUTPUT). A dependency "
        "exists when one tool's OUTPUT is exactly what the other tool needs as "
        "a REQUIRED INPUT (same type AND meaning). The direction points from "
        "the consumer to the producer: 'X depends on Y' means Y produces a "
        "value that X requires as input. Output only strict JSON with one key "
        "'label'. No reasoning. Valid labels: a_depends_on_b, b_depends_on_a, "
        "no_dependency."
    )
    user = (
        "Determine, from inputs/outputs only, whether one tool's output feeds a "
        "required input of the other. Ignore shared category, similar names, or "
        "merely related topics.\n\n"
        "Worked example:\n"
        "Tool A:\n"
        "Tool ID: ConvertingPeptide2SMILES\n"
        "Signature: (peptide: str)\n"
        "Docstring: Convert a peptide sequence into its SMILES string. "
        "Returns: a SMILES string.\n"
        "Tool B:\n"
        "Tool ID: SMILESToInChI\n"
        "Signature: (smiles: str)\n"
        "Docstring: Convert a SMILES string into an InChI. Returns: an InChI "
        "string.\n"
        "Analysis: A outputs a SMILES string; B's required input is a SMILES "
        "string. So B consumes A's output -> B depends on A.\n"
        "Answer: {\"label\":\"b_depends_on_a\"}\n\n"
        "If neither tool's output is a required input of the other, the answer "
        "is {\"label\":\"no_dependency\"}.\n\n"
        "Now classify this pair:\n"
        "Tool A:\n"
        f"{canonical_tool_text(tool_a)}\n\n"
        "Tool B:\n"
        f"{canonical_tool_text(tool_b)}\n\n"
        "Return exactly one JSON object such as {\"label\":\"no_dependency\"}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_label(raw_output: str) -> tuple[str, bool]:
    text = raw_output.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    braced = re.search(r"\{.*\}", text, flags=re.S)
    if braced:
        candidates.append(braced.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        label = str(data.get("label", "")).strip()
        if label in VALID_LABELS:
            return label, True
    return "no_dependency", False


def _chat_completion(messages: list[dict[str, str]], *, base_url: str, api_key: str, model: str) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        # Pairwise dependency judgement is a short classification, so the
        # reasoning trace is disabled and each call returns only the JSON
        # label. Ignored by non-reasoning models.
        "thinking": {"type": "disabled"},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def judge_pair(
    tool_a: dict[str, str],
    tool_b: dict[str, str],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    raw = _chat_completion(build_prompt(tool_a, tool_b), base_url=base_url, api_key=api_key, model=model)
    label, valid = parse_label(raw)
    return {
        "tool_a": tool_a["id"],
        "tool_b": tool_b["id"],
        "label": label,
        "valid": valid,
        "raw_output": raw,
    }


def pair_key(tool_a: str, tool_b: str) -> str:
    return f"{tool_a}\u241f{tool_b}"


def load_checkpoint(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[pair_key(str(row["tool_a"]), str(row["tool_b"]))] = row
    return rows


def graph_from_rows(
    *,
    tools: list[dict[str, str]],
    rows: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    directed_edges: list[dict[str, str]] = []
    undirected_seen: set[tuple[str, str]] = set()
    invalid_outputs: list[dict[str, str]] = []

    for row in rows:
        a = str(row["tool_a"])
        b = str(row["tool_b"])
        label = str(row.get("label", "no_dependency"))
        if not row.get("valid", False):
            invalid_outputs.append({
                "tool_a": a,
                "tool_b": b,
                "raw_output": str(row.get("raw_output", "")),
            })
            continue
        if label == "a_depends_on_b":
            directed_edges.append({"source": a, "target": b, "label": label})
            undirected_seen.add(tuple(sorted((a, b))))
        elif label == "b_depends_on_a":
            directed_edges.append({"source": b, "target": a, "label": label})
            undirected_seen.add(tuple(sorted((a, b))))

    undirected_edges = [
        {"tool_a": a, "tool_b": b}
        for a, b in sorted(undirected_seen)
    ]
    return {
        "version": GRAPH_VERSION,
        "edge_definition": "directed edge source->target means source depends on target",
        "main_adjacency": "symmetric",
        "model": model,
        "num_tools": len(tools),
        "num_pairs_evaluated": len(rows),
        "invalid_label_count": len(invalid_outputs),
        "directed_edges": directed_edges,
        "undirected_edges": undirected_edges,
        "invalid_outputs": invalid_outputs,
    }


def build_llm_graph(
    *,
    tools_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    max_pairs: int | None = None,
    max_concurrent: int = 20,
    resume: bool = False,
    retry_invalid: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    tools = load_tool_registry(tools_path)
    tool_by_id = {tool["id"]: tool for tool in tools}
    all_pairs = [(a["id"], b["id"]) for a, b in combinations(tools, 2)]
    if max_pairs is not None:
        all_pairs = all_pairs[:max_pairs]

    model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-chat"
    base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise ValueError("Missing LLM API key. Set LLM_API_KEY or OPENAI_API_KEY.")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("LLM API key must be the real ASCII key, not a placeholder.") from exc

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    existing = load_checkpoint(checkpoint_path) if resume or retry_invalid else {}

    rows_by_key: dict[str, dict[str, Any]] = {}
    to_run: list[tuple[str, str]] = []
    for a, b in all_pairs:
        key = pair_key(a, b)
        row = existing.get(key)
        if row and (row.get("valid", False) or not retry_invalid):
            rows_by_key[key] = row
        else:
            to_run.append((a, b))

    lock = threading.Lock()
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def run_one(a: str, b: str) -> dict[str, Any]:
        try:
            return judge_pair(
                tool_by_id[a],
                tool_by_id[b],
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        except Exception as exc:
            return {
                "tool_a": a,
                "tool_b": b,
                "label": "no_dependency",
                "valid": False,
                "raw_output": "",
                "error": str(exc),
            }

    completed = 0
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_to_pair = {pool.submit(run_one, a, b): (a, b) for a, b in to_run}
        for future in as_completed(future_to_pair):
            a, b = future_to_pair[future]
            row = future.result()
            rows_by_key[pair_key(a, b)] = row
            completed += 1
            if checkpoint_path:
                with lock, checkpoint_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 100 == 0 or completed == len(to_run):
                print(f"  judged {completed}/{len(to_run)} new pairs")

    ordered_rows = [rows_by_key[pair_key(a, b)] for a, b in all_pairs if pair_key(a, b) in rows_by_key]
    graph = graph_from_rows(tools=tools, rows=ordered_rows, model=model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TGR-style dependency graph using an LLM pair judge.")
    parser.add_argument("--tools", required=True, help="Tool registry JSON.")
    parser.add_argument("--output", required=True, help="Output graph JSON.")
    parser.add_argument("--checkpoint", help="Checkpoint JSONL.")
    parser.add_argument("--max-pairs", type=int, help="Limit number of unordered pairs for preview/debug.")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--base-url", help="OpenAI-compatible chat base URL.")
    parser.add_argument("--api-key", help="Chat API key.")
    parser.add_argument("--model", help="Chat model.")
    args = parser.parse_args()

    graph = build_llm_graph(
        tools_path=args.tools,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        max_pairs=args.max_pairs,
        max_concurrent=args.max_concurrent,
        resume=args.resume,
        retry_invalid=args.retry_invalid,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    print(json.dumps({
        "num_tools": graph["num_tools"],
        "num_pairs_evaluated": graph["num_pairs_evaluated"],
        "directed_edges": len(graph["directed_edges"]),
        "undirected_edges": len(graph["undirected_edges"]),
        "invalid_label_count": graph["invalid_label_count"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved graph: {args.output}")


if __name__ == "__main__":
    main()
