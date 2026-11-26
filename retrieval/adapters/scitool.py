"""SciTool adapter using the tool server's ``/invoke`` endpoint."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from retrieval.adapters.registry_contracts import REGISTRY_CONTRACT_ADAPTER
from retrieval.core.registry import load_registry_tools
from retrieval.core.runtime import (
    BatchInvocationPool,
    add_graph_arguments,
    build_stateless_graph,
    unchanged_result,
)


ADAPTER_VERSION = "scitool_baseline_container_v2"
_MARKDOWN_FIELD = re.compile(
    r"^\s*\*\*[^*\n]+:\*\*\s*(?P<value>.+?)\s*$"
)
_FAILURE_PREFIXES = (
    "an error occurred",
    "error:",
    "failed:",
    "invalid ",
)


def terminal_markdown_value(value: Any) -> Any:
    """Remove SciTool's display wrapper and publish its terminal scalar."""

    if not isinstance(value, str):
        return value
    lines = [
        line.strip()
        for line in value.splitlines()
        if line.strip() and line.strip() not in {"```", "~~~", "---"}
    ]
    if not lines:
        return ""
    candidate = lines[-1]
    match = _MARKDOWN_FIELD.fullmatch(candidate)
    if match is not None:
        candidate = match.group("value").strip()
    if (
        len(candidate) >= 2
        and candidate[0] == candidate[-1]
        and candidate[0] in {"`", '"', "'"}
    ):
        candidate = candidate[1:-1].strip()
    return candidate


def is_failure_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold().startswith(
        _FAILURE_PREFIXES
    )


@dataclass(frozen=True)
class SciToolInvoker:
    service_urls: tuple[str, ...]
    timeout_seconds: int = 120
    _services: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_services", itertools.cycle(self.service_urls))

    def __call__(self, tool: str, args: Mapping[str, Any]):
        values = list(args.values())
        input_string = (
            ""
            if not values
            else str(values[0])
            if len(values) == 1
            else json.dumps(dict(args), ensure_ascii=False)
        )

        try:
            response = requests.post(
                next(self._services),
                json={
                    "tool_name": tool,
                    "input": input_string,
                    "timeout": self.timeout_seconds,
                },
                timeout=self.timeout_seconds + 10,
            )
        except requests.RequestException as exc:
            return unchanged_result(
                "error",
                None,
                {
                    "entry": "/invoke -> tool_runner.run_tool",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc)[:1000],
                },
            )
        try:
            body = response.json()
        except ValueError:
            body = {
                "output": None,
                "error": f"non-JSON HTTP {response.status_code}",
            }
        error = body.get("error")
        output = terminal_markdown_value(body.get("output"))
        failure_text = is_failure_text(output)
        return unchanged_result(
            "success" if response.ok and not error and not failure_text else "error",
            output,
            {
                "entry": "/invoke -> loaded SciTool callable(input_str)",
                "status_code": response.status_code,
                "server_error": str(
                    error or ("tool returned failure text" if failure_text else "")
                )[:1000],
                "response_policy": "terminal_markdown_scalar",
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--service-url", action="append", required=True)
    parser.add_argument("--tool-timeout", type=int, default=120)
    add_graph_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    definitions, metadata, metadata_id = load_registry_tools(args.registry)
    pool = BatchInvocationPool(
        SciToolInvoker(tuple(args.service_url), args.tool_timeout),
        workers=96,
        processes=False,
    )
    graph = build_stateless_graph(
        args,
        definitions=definitions,
        metadata=metadata,
        metadata_id=metadata_id,
        runtime_id=ADAPTER_VERSION,
        runtime_metadata={
            "adapter": ADAPTER_VERSION,
            "entry": (
                f"{len(args.service_url)} x /invoke -> "
                "loaded SciTool callable(input_str)"
            ),
            "result_policy": "terminal_markdown_scalar",
            "imports_task_or_gold_data": False,
        },
        invocation_pool=pool,
        contract_adapter=REGISTRY_CONTRACT_ADAPTER,
        calls_per_worker_round=8,
        worker_ipc_batch_size=1,
    )
    print(json.dumps({
        "adapter": ADAPTER_VERSION,
        "tools": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "usage": graph["build"]["usage"],
        "stop_reason": graph["build"]["stop_reason"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
