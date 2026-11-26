"""Load benchmark tool registries into the common dependency-graph schema."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping

from retrieval.core.graph import ToolDefinition, ToolParameter


def _type_name_schema(kind: str) -> dict[str, Any]:
    return {
        "str": {"type": "string"},
        "string": {"type": "string"},
        "int": {"type": "integer"},
        "integer": {"type": "integer"},
        "float": {"type": "number"},
        "number": {"type": "number"},
        "bool": {"type": "boolean"},
        "boolean": {"type": "boolean"},
        "list": {"type": "array"},
        "array": {"type": "array"},
        "tuple": {"type": "array"},
        "set": {"type": "array"},
        "dict": {"type": "object"},
        "mapping": {"type": "object"},
        "object": {"type": "object"},
    }.get(kind.casefold(), {})


def _schema(annotation: ast.expr | None) -> dict[str, Any]:
    if annotation is None:
        return {}
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return {"type": "null"}
    if isinstance(annotation, ast.Name):
        kind = annotation.id.lower()
    elif isinstance(annotation, ast.Attribute):
        kind = annotation.attr.lower()
    elif isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            kind = annotation.value.id.lower()
        elif isinstance(annotation.value, ast.Attribute):
            kind = annotation.value.attr.lower()
        else:
            kind = ""
    elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        alternatives = [_schema(annotation.left), _schema(annotation.right)]
        types = sorted({
            typ
            for row in alternatives
            for typ in (
                row.get("type")
                if isinstance(row.get("type"), list)
                else [row.get("type")]
            )
            if typ
        })
        return {"type": types} if types else {}
    else:
        kind = ""
    return _type_name_schema(kind)


def return_schema_from_signature(signature: str) -> dict[str, Any]:
    """Read the public callable's return annotation."""

    tree = ast.parse(f"def _tool{signature}:\n    pass\n")
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError(f"invalid tool signature: {signature!r}")
    return _schema(function.returns)


def _documented_return_schema(docstring: str) -> dict[str, Any]:
    """Read a conventional ``Returns: str`` public declaration."""

    match = re.search(
        r"(?mi)^\s*returns?\s*:\s*(?:\n\s*)?"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        docstring,
    )
    return _type_name_schema(match.group(1)) if match else {}


def parameters_from_signature(
    signature: str,
    *,
    documented_optional: frozenset[str] = frozenset(),
) -> tuple[ToolParameter, ...]:
    """Parse the registry's public Python signature without importing a tool."""

    tree = ast.parse(f"def _tool{signature}:\n    pass\n")
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError(f"invalid tool signature: {signature!r}")
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = (
        [None] * (len(positional) - len(function.args.defaults))
        + list(function.args.defaults)
    )
    result = [
        ToolParameter(
            name=argument.arg,
            required=default is None and argument.arg not in documented_optional,
            schema=_schema(argument.annotation),
        )
        for argument, default in zip(positional, positional_defaults, strict=True)
    ]
    result.extend(
        ToolParameter(
            name=argument.arg,
            required=default is None and argument.arg not in documented_optional,
            schema=_schema(argument.annotation),
        )
        for argument, default in zip(
            function.args.kwonlyargs,
            function.args.kw_defaults,
            strict=True,
        )
    )
    return tuple(result)


def _documented_optional_parameters(docstring: str) -> frozenset[str]:
    """Read conventional ``name (type, optional):`` public documentation."""

    return frozenset(
        match.group(1)
        for match in re.finditer(
            r"(?mi)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*"
            r"(?:\([^)\n]*\boptional\b[^)\n]*\)|:[^\n]*\boptional\b)",
            docstring,
        )
    )


def _api_parameter_schema(parameter: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the public API registry's primitive type vocabulary."""

    kind = str(parameter.get("type") or "").strip().casefold()
    if kind in {"number", "float", "double"}:
        return {"type": "number"}
    if kind in {"integer", "int"}:
        return {"type": "integer"}
    if kind in {"boolean", "bool"}:
        return {"type": "boolean"}
    if kind in {"array", "list"}:
        return {"type": "array"}
    if kind in {"object", "dict", "mapping"}:
        return {"type": "object"}
    return {"type": "string"}


def _api_parameters(api: Mapping[str, Any]) -> tuple[ToolParameter, ...]:
    """Use the exact JSON-object interface consumed by a virtual API runtime."""

    parameters: list[ToolParameter] = []
    seen: set[str] = set()
    for required, rows in (
        (True, api.get("required_parameters") or ()),
        (False, api.get("optional_parameters") or ()),
    ):
        for row in rows:
            name = str(row.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            parameters.append(ToolParameter(
                name=name,
                required=required,
                schema=_api_parameter_schema(row),
            ))
    return tuple(parameters)


def _normalize_json_schema(value: Any) -> Any:
    """Translate Python-style public schema names to JSON Schema names."""

    if isinstance(value, list):
        return [_normalize_json_schema(child) for child in value]
    if not isinstance(value, Mapping):
        return value
    normalized = {
        str(key): _normalize_json_schema(child)
        for key, child in value.items()
    }
    kind = normalized.get("type")
    if isinstance(kind, str):
        normalized["type"] = {
            "bool": "boolean",
            "dict": "object",
            "float": "number",
            "int": "integer",
            "list": "array",
            "str": "string",
        }.get(kind, kind)
    return normalized


def _json_schema_parameters(
    schema: Mapping[str, Any],
) -> tuple[ToolParameter, ...]:
    """Read a callable's parameters from its public JSON object schema."""

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    required = set(map(str, schema.get("required") or ()))
    return tuple(
        ToolParameter(
            name=str(name),
            required=str(name) in required,
            schema=(
                _normalize_json_schema(value)
                if isinstance(value, Mapping)
                else {}
            ),
        )
        for name, value in properties.items()
    )


def _representative_response(value: Any) -> Any:
    """Keep public field structure while removing repeated response rows."""

    if isinstance(value, list):
        return [] if not value else [_representative_response(value[0])]
    if isinstance(value, Mapping):
        if value and all(str(key).isdigit() for key in value):
            return [_representative_response(next(iter(value.values())))]
        return {
            str(key): _representative_response(child)
            for key, child in value.items()
        }
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return 1
    if isinstance(value, float):
        return 1.0
    if isinstance(value, str):
        return "string"
    return None


def _schema_example(schema: Mapping[str, Any]) -> Any:
    """Turn a public JSON schema into the shape expected by contract exports."""

    if "example" in schema:
        return schema["example"]
    if schema.get("enum"):
        return schema["enum"][0]
    for keyword in ("oneOf", "anyOf"):
        choices = schema.get(keyword)
        if isinstance(choices, list) and choices:
            return _schema_example(choices[0])
    kind = schema.get("type")
    if kind == "object" or isinstance(schema.get("properties"), Mapping):
        return {
            str(key): _schema_example(child)
            for key, child in (schema.get("properties") or {}).items()
            if isinstance(child, Mapping)
        }
    if kind == "array":
        items = schema.get("items")
        return [_schema_example(items)] if isinstance(items, Mapping) else []
    return {
        "string": "string",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "null": None,
    }.get(str(kind), None)


def load_registry_tools(
    registry_path: str | Path,
    api_metadata_path: str | Path | None = None,
) -> tuple[
    list[ToolDefinition],
    dict[str, Mapping[str, Any]],
    str,
]:
    """Return common definitions plus model-visible public metadata."""

    registry_file = Path(registry_path)
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    sidecar: Mapping[str, Any] = {}
    if api_metadata_path is not None:
        sidecar = json.loads(
            Path(api_metadata_path).read_text(encoding="utf-8")
        )

    definitions: list[ToolDefinition] = []
    metadata: dict[str, Mapping[str, Any]] = {}
    parameter_count = 0
    for row in registry:
        tool = str(row["name"])
        docstring = str(row.get("docstring") or "")
        signature = str(row.get("signature") or "()")
        raw = dict(sidecar.get(tool) or {})
        api = dict(raw.get("api") or {})
        has_api_interface = (
            "required_parameters" in api or "optional_parameters" in api
        )
        public_parameter_schema = row.get("parameters")
        if has_api_interface:
            parameters = _api_parameters(api)
        elif isinstance(public_parameter_schema, Mapping):
            parameters = _json_schema_parameters(public_parameter_schema)
        else:
            parameters = parameters_from_signature(
                signature,
                documented_optional=_documented_optional_parameters(docstring),
            )
        parameter_count += len(parameters)
        if has_api_interface:
            documented_parameters = {
                str(item.get("name") or ""): item
                for item in (
                list(api.get("required_parameters") or ())
                + list(api.get("optional_parameters") or ())
            )
            }
        elif isinstance(public_parameter_schema, Mapping):
            documented_parameters = {
                str(name): value
                for name, value in (
                    public_parameter_schema.get("properties") or {}
                ).items()
                if isinstance(value, Mapping)
            }
        else:
            documented_parameters = {}
        public_parameters = [{
            "name": parameter.name,
            "required": parameter.required,
            "description": str(
                (documented_parameters.get(parameter.name) or {}).get(
                    "description"
                )
                or ""
            ),
            "schema": dict(parameter.schema),
            "default": (
                documented_parameters.get(parameter.name) or {}
            ).get("default"),
        } for parameter in parameters]
        response_example = None
        if "body" in api:
            response_example = _representative_response(api["body"])
        elif isinstance(api.get("schema"), Mapping) and api["schema"]:
            response_example = _schema_example(api["schema"])
        elif isinstance(row.get("response"), Mapping):
            response_example = _schema_example(
                _normalize_json_schema(row["response"])
            )
        elif not has_api_interface:
            return_schema = (
                return_schema_from_signature(signature)
                or _documented_return_schema(docstring)
            )
            response_example = _schema_example(return_schema)
        public = {
            "tool": tool,
            "method": api.get("method"),
            "path": api.get("url"),
            "description": docstring,
            "category": row.get("category"),
            "function": row.get("function"),
            "parameters": public_parameters,
            "calling_convention": (
                {
                    "kind": "json_object",
                    "argument_names": "exact_public_api_names",
                }
                if has_api_interface
                else {
                    "kind": "declared_signature",
                    "argument_names": "signature_names",
                }
            ),
            "response_schemas": (
                {"success": response_example}
                if response_example not in (None, {}, [])
                else {}
            ),
        }
        definitions.append(ToolDefinition(
            tool=tool,
            parameters=parameters,
            public_metadata=public,
        ))
        metadata[tool] = public

    # Fresh registry runs create contracts and both field stores together.
    # A descriptive run-local identity is sufficient; no artifact checksum or
    # persisted hash machinery is needed.
    metadata_id = (
        f"registry_metadata_v1:{registry_file.parent.name}:"
        f"{len(definitions)}:{parameter_count}"
    )
    return definitions, metadata, metadata_id
