"""BFCL adapter that executes the official in-process multi-turn tools."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib
import json
import multiprocessing
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.core.registry import load_registry_tools
from retrieval.adapters.registry_contracts import (
    STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER,
)
from retrieval.core.graph import (
    InvocationResult,
    StateSnapshot,
    TransitionDelta,
    classify_outcome_schemas,
)
from retrieval.core.runtime import (
    add_graph_arguments,
    build_stateless_graph,
)


ADAPTER_VERSION = "bfcl_official_default_state_v1"


def _add_bfcl_path(root: str | Path) -> None:
    path = str(Path(root).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _new_instances(
    bfcl_root: str | Path,
    categories: Sequence[str],
) -> dict[str, Any]:
    _add_bfcl_path(bfcl_root)
    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING,
        STATELESS_CLASSES,
    )

    instances = {}
    for category in categories:
        module = importlib.import_module(CLASS_FILE_PATH_MAPPING[category])
        instance = getattr(module, category)()
        if category not in STATELESS_CLASSES:
            instance._load_scenario({})
        instances[category] = instance
    return instances


def _state_value(value: Any, seen: set[int]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    identity = id(value)
    if identity in seen:
        return f"<{type(value).__name__}:cycle>"
    seen.add(identity)
    if isinstance(value, Mapping):
        return {
            str(key): _state_value(child, seen)
            for key, child in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_state_value(child, seen) for child in value]
    if isinstance(value, set):
        return sorted(
            (_state_value(child, seen) for child in value),
            key=lambda child: json.dumps(child, sort_keys=True, default=str),
        )
    if hasattr(value, "__dict__"):
        return {
            str(key): _state_value(child, seen)
            for key, child in sorted(vars(value).items())
            if key not in {"_api_description", "_random", "parent"}
        }
    return str(value)


def _state_view(instances: Mapping[str, Any]) -> dict[str, Any]:
    return {
        category: _state_value(instance, set())
        for category, instance in sorted(instances.items())
    }


def _invoke_one(
    instances: dict[str, Any],
    request: Mapping[str, Any],
) -> tuple[InvocationResult, bytes | None]:
    tool = str(request["tool"])
    category, method = tool.split(".", 1)
    before = _state_view(instances)
    try:
        response = getattr(instances[category], method)(
            **dict(request.get("args") or {})
        )
        status = (
            "rejected"
            if isinstance(response, Mapping) and response.get("error")
            else "success"
        )
    except Exception as exc:
        status = "rejected"
        response = {
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }
    after = _state_view(instances)
    changes = {
        category: value
        for category, value in after.items()
        if value != before[category]
    }
    changed = bool(changes)
    after_digest = (
        str(request.get("save_state_id") or f"{tool}:changed")
        if changed
        else str(request["state_digest"])
    )
    result = InvocationResult(
        status=status,
        response=response,
        machine_status={
            "runtime": ADAPTER_VERSION,
            "category": category,
            "method": method,
        },
        transition=TransitionDelta(
            str(request["state_digest"]),
            after_digest,
            changes,
        ),
    )
    return result, pickle.dumps(instances) if changed else None


def _initialize_worker(bfcl_root: str) -> None:
    _add_bfcl_path(bfcl_root)


def _invoke_batch(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows = []
    for request in requests:
        try:
            instances = pickle.loads(request["state_blob"])
            result, state_blob = _invoke_one(instances, request)
            outcome = classify_outcome_schemas(
                dict(request["success_schema"]),
                dict(request["failure_schema"]),
                result,
            )
            rows.append({
                "ok": True,
                "result": result.to_dict(),
                "state_blob": state_blob,
                "outcome": outcome,
            })
        except Exception as exc:
            rows.append({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
    return rows, os.getpid()


class BFCLRuntime:
    runtime_id = ADAPTER_VERSION

    def __init__(self, bfcl_root: Path, categories: Sequence[str]) -> None:
        self.categories = tuple(categories)
        self.states: dict[str, bytes] = {}
        self.root_blob = pickle.dumps(
            _new_instances(bfcl_root, self.categories)
        )
        self.root_handle: str | None = None

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            "bfcl-default-root",
            {"classes": list(self.categories)},
        )

    def save_state(self, state_id: str) -> str:
        self.states[state_id] = self.root_blob
        self.root_handle = state_id
        return state_id

    def load_state(self, state_handle: str) -> None:
        if state_handle not in self.states:
            raise KeyError(state_handle)

    def state_handle_available(self, state_handle: str) -> bool:
        return state_handle in self.states

    def discard_state(self, state_handle: str) -> None:
        if state_handle != self.root_handle:
            self.states.pop(state_handle, None)

    def invoke(self, tool: str, args: Mapping[str, Any]) -> InvocationResult:
        raise RuntimeError(f"{tool} was sent to the coordinator")

    def source_metadata(self) -> Mapping[str, Any]:
        return {
            "adapter": ADAPTER_VERSION,
            "initial_state": "official public defaults",
            "imports_task_or_gold_data": False,
        }


class BFCLInvocationPool:
    def __init__(
        self,
        runtime: BFCLRuntime,
        *,
        bfcl_root: Path,
        workers: int,
    ) -> None:
        self.runtime = runtime
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(str(bfcl_root.resolve()),),
        )

    async def invoke_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        avoid_worker: int | None = None,
    ) -> tuple[
        list[tuple[
            InvocationResult | None,
            str | None,
            Mapping[str, Any] | None,
            str | None,
        ]],
        int,
    ]:
        del avoid_worker
        payloads = []
        for request in requests:
            handle = str(request["state_handle"])
            payloads.append({
                **dict(request),
                "state_blob": self.runtime.states[handle],
            })
        loop = asyncio.get_running_loop()
        raw_rows, worker = await loop.run_in_executor(
            self.executor, _invoke_batch, payloads
        )
        rows = []
        for request, raw in zip(requests, raw_rows, strict=True):
            if not raw["ok"]:
                rows.append((None, None, raw, None))
                continue
            result = InvocationResult.from_dict(raw["result"])
            saved_handle = None
            state_blob = raw.get("state_blob")
            if state_blob is not None and request.get("save_state_id"):
                saved_handle = str(request["save_state_id"])
                self.runtime.states[saved_handle] = state_blob
            rows.append((result, saved_handle, None, raw.get("outcome")))
        return rows, worker

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bfcl-root", type=Path, required=True)
    add_graph_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    definitions, metadata, metadata_id = load_registry_tools(args.registry)
    categories = sorted({
        str(row["category"]) for row in metadata.values()
    })
    runtime = BFCLRuntime(args.bfcl_root, categories)
    pool = BFCLInvocationPool(
        runtime,
        bfcl_root=args.bfcl_root,
        workers=96,
    )
    graph = build_stateless_graph(
        args,
        definitions=definitions,
        metadata=metadata,
        metadata_id=metadata_id,
        invocation_pool=pool,
        runtime=runtime,
        max_states=2_000,
        contract_adapter=STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER,
        state_domains={
            tool: str(row["category"])
            for tool, row in metadata.items()
        },
    )
    print(json.dumps({
        "adapter": ADAPTER_VERSION,
        "tools": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "usage": graph["build"]["usage"],
        "stop_reason": graph["build"]["stop_reason"],
    }, ensure_ascii=False, indent=2))
    fatal_stop_reasons = {
        "coordinator_error",
        "field_embedding_error",
        "max_external_model_calls",
        "max_retries",
        "max_storage_bytes",
        "worker_failure_rate",
    }
    return 2 if graph["build"]["stop_reason"] in fatal_stop_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
