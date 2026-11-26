"""Reduce a discovery run to the compact graph used by retrieval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.adapters.appworld import load_appworld_tools
from retrieval.core.contracts import load_contracts, write_contracts
from retrieval.core.fields import (
    SQLiteFieldEmbeddingStore,
    SQLiteFieldPriorityIndex,
)
from retrieval.core.graph import Observation, ToolDefinition
from retrieval.core.registry import load_registry_tools
from retrieval.core.retrieval import DependencyEdge, write_graph


def _raw_edges(path: str | Path) -> list[dict[str, Any]]:
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    edges = graph.get("tool_projection", {}).get("edges")
    if not isinstance(edges, list):
        raise ValueError("raw graph is missing tool_projection.edges")
    return edges


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        return set().union(*(
            _schema_types(row)
            for row in alternatives
            if isinstance(row, Mapping)
        ))
    value = schema.get("type")
    if isinstance(value, list):
        return set(map(str, value))
    return {str(value)} if isinstance(value, str) else {"any"}


def _compatible(output: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    output_types = _schema_types(output)
    target_types = _schema_types(target)
    return (
        "any" in output_types
        or "any" in target_types
        or bool(output_types & target_types)
        or bool(
            {"integer", "number"} & output_types
            and {"integer", "number"} & target_types
        )
    )


def _input_edges(
    *,
    raw_graph: str | Path,
    definitions: Sequence[ToolDefinition],
    contracts_path: str | Path,
    field_embeddings_path: str | Path,
    required: bool,
) -> list[DependencyEdge]:
    contracts = load_contracts(contracts_path)
    parameters = {
        (tool.tool, parameter.name): parameter
        for tool in definitions
        for parameter in tool.parameters
    }
    exports: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for tool, contract in contracts.items():
        for export in contract.exports:
            exports[(tool, export.source, export.name)].append(export)

    store = SQLiteFieldEmbeddingStore(field_embeddings_path, read_only=True)
    try:
        index = SQLiteFieldPriorityIndex(store, definitions)
        candidates = []
        for edge in _raw_edges(raw_graph):
            if edge.get("kind") != "input" or bool(edge.get("required")) != required:
                continue
            source = str(edge["source"])
            target = str(edge["target"])
            target_input = str(edge.get("target_parameter") or "")
            parameter = parameters.get((target, target_input))
            contract = contracts.get(source)
            if parameter is None or contract is None:
                continue
            scores: dict[str, list[float]] = defaultdict(list)
            for row in edge.get("source_observations") or ():
                location = str(row.get("location") or "")
                export_name = str(row.get("export") or "")
                for export in exports.get((source, location, export_name), ()):
                    observation = Observation.create(
                        None,
                        source_tool=source,
                        source_call="publish",
                        source_revision=contract.revision,
                        location=export.source,
                        export_name=export.name,
                        path=str(row.get("path") or export.path),
                        field_path=export.path,
                        value_schema=export.value_schema,
                    )
                    score = index.score(
                        observation,
                        target_tool=target,
                        target_parameter=parameter,
                    )
                    if score is not None:
                        scores[export_name].append(float(score))
            export_scores = {
                name: max(values) for name, values in scores.items()
            }
            candidates.append((edge, export_scores))

        best_target: dict[tuple[str, str], tuple[float, str]] = {}
        for edge, export_scores in candidates:
            for export, score in export_scores.items():
                key = (str(edge["source"]), export)
                priority = (score, str(edge.get("target_parameter") or ""))
                if key not in best_target or priority > best_target[key]:
                    best_target[key] = priority

        result = []
        for edge, export_scores in candidates:
            target_input = str(edge.get("target_parameter") or "")
            if export_scores and not any(
                best_target[(str(edge["source"]), export)][1] == target_input
                for export in export_scores
            ):
                continue
            result.append(DependencyEdge(
                edge_id=str(edge["id"]),
                provider=str(edge["source"]),
                consumer=str(edge["target"]),
                consumer_slot=target_input,
                kind="input",
                field_score=(max(export_scores.values()) if export_scores else None),
            ))
        return result
    finally:
        store.close()


def _state_edges(raw_graph: str | Path) -> list[DependencyEdge]:
    return [
        DependencyEdge(
            edge_id=str(edge["id"]),
            provider=str(edge["source"]),
            consumer=str(edge["target"]),
            consumer_slot=f"__state__:{edge['source']}",
            kind="state",
        )
        for edge in _raw_edges(raw_graph)
        if edge.get("kind") == "state"
    ]


def _contract_edges(
    *,
    definitions: Sequence[ToolDefinition],
    contracts_path: str | Path,
    field_embeddings_path: str | Path,
) -> list[DependencyEdge]:
    contracts = load_contracts(contracts_path)
    store = SQLiteFieldEmbeddingStore(field_embeddings_path, read_only=True)
    try:
        index = SQLiteFieldPriorityIndex(store, definitions)
        result = []
        for source, contract in contracts.items():
            for export in contract.exports:
                if not export.propagate or export.path == "$..[*]":
                    continue
                observation = Observation.create(
                    None,
                    source_tool=source,
                    source_call="publish",
                    source_revision=contract.revision,
                    location=export.source,
                    export_name=export.name,
                    path=export.path,
                    field_path=export.path,
                    value_schema=export.value_schema,
                )
                best: tuple[tuple[float, str, str], DependencyEdge] | None = None
                for target in definitions:
                    if target.tool == source:
                        continue
                    for parameter in target.parameters:
                        if not _compatible(export.value_schema, parameter.schema):
                            continue
                        score = index.score(
                            observation,
                            target_tool=target.tool,
                            target_parameter=parameter,
                        )
                        if score is None:
                            continue
                        edge = DependencyEdge(
                            edge_id=(
                                f"contract-field:{source}:{export.name}:"
                                f"{target.tool}:{parameter.name}"
                            ),
                            provider=source,
                            consumer=target.tool,
                            consumer_slot=parameter.name,
                            kind="contract",
                            field_score=float(score),
                        )
                        candidate = ((float(score), target.tool, parameter.name), edge)
                        if best is None or candidate[0] > best[0]:
                            best = candidate
                if best is not None:
                    result.append(best[1])
        return result
    finally:
        store.close()


def publish(
    benchmark: str,
    *,
    data_dir: Path,
    raw_graph: Path,
    contracts: Path,
    field_embeddings: Path,
    output: Path,
    contracts_output: Path,
) -> int:
    if benchmark == "appworld":
        definitions, _, _, _ = load_appworld_tools(data_dir / "api_metadata.json")
        edges = _input_edges(
            raw_graph=raw_graph,
            definitions=definitions,
            contracts_path=contracts,
            field_embeddings_path=field_embeddings,
            required=True,
        )
    else:
        definitions, _, _ = load_registry_tools(data_dir / "tool_registry.json")
        if benchmark == "scitool":
            edges = _input_edges(
                raw_graph=raw_graph,
                definitions=definitions,
                contracts_path=contracts,
                field_embeddings_path=field_embeddings,
                required=False,
            )
        elif benchmark == "bfcl":
            edges = [
                *_state_edges(raw_graph),
                *_contract_edges(
                    definitions=definitions,
                    contracts_path=contracts,
                    field_embeddings_path=field_embeddings,
                ),
            ]
        else:
            raise ValueError(f"unknown benchmark: {benchmark}")
    write_graph(edges, output)
    selected_contracts = load_contracts(contracts)
    write_contracts(selected_contracts, contracts_output)
    print(json.dumps({
        "benchmark": benchmark,
        "edges": len(edges),
        "contracts": len(selected_contracts),
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("appworld", "scitool", "bfcl"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-graph", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--field-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contracts-output", type=Path, required=True)
    args = parser.parse_args()
    return publish(
        args.benchmark,
        data_dir=args.data_dir,
        raw_graph=args.raw_graph,
        contracts=args.contracts,
        field_embeddings=args.field_embeddings,
        output=args.output,
        contracts_output=args.contracts_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
