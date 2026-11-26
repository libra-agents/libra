"""AppWorld adapter and CLI for the generic dependency discovery core."""

from __future__ import annotations

import argparse
import asyncio
import ast
import atexit
import concurrent.futures
import importlib.util
import json
import math
import multiprocessing
import os
import re
import shutil
import stat
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from retrieval.core.graph import (
    BuildConfig,
    InvocationResult,
    SearchBudget,
    SQLiteBuildStore,
    StateSnapshot,
    ToolDefinition,
    ToolParameter,
    TransitionDelta,
    classify_outcome_schemas,
    json_safe,
    stable_id,
)
from retrieval.core.fields import (
    QwenVllmTextEmbedder,
    SQLiteFieldEmbeddingStore,
    SQLiteFieldPriorityIndex,
)
from retrieval.adapters.appworld_contracts import APPWORLD_CONTRACT_ADAPTER
from retrieval.core.contracts import (
    ContractGenerator,
    DEFAULT_FEEDBACK_POLICY,
    DeepSeekFlash,
    MINIMAL_SUCCESS_TEMPLATE_POLICY,
    OnlineContractSet,
    load_active_contract_manifest,
    select_template_policy,
)
from retrieval.core.parallel import (
    ParallelBuildConfig,
    ParallelDependencyGraphBuilder,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPWORLD_ROOT = REPO_ROOT / "benchmark_code" / "appworld"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset and embedding" / "appworld"
DEFAULT_METADATA = DEFAULT_DATASET_DIR / "api_metadata.json"
APPWORLD_ADAPTER_VERSION = "appworld_branching_adapter_v3_direct_state_load"
BASE_FIXTURE = "_base_"


_PROCESS_RUNTIME: "AppWorldStatefulRuntime | None" = None


def _close_process_runtime() -> None:
    global _PROCESS_RUNTIME
    if _PROCESS_RUNTIME is not None:
        _PROCESS_RUNTIME.close()
        _PROCESS_RUNTIME = None


def _initialize_process_runtime(
    appworld_root: str,
    experiment_name: str,
    metadata_path: str,
    checkpoint_root: str,
    worker_root: str,
    unsupported_constraints: Sequence[Mapping[str, str]],
) -> None:
    global _PROCESS_RUNTIME
    warnings.filterwarnings(
        "ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*"
    )
    worker_id = os.getpid()
    worker_directory = Path(worker_root) / f"worker-{worker_id}"
    worker_directory.mkdir(parents=True, exist_ok=True)
    os.environ["APPWORLD_CACHE"] = str(worker_directory / "cache")
    _PROCESS_RUNTIME = AppWorldStatefulRuntime(
        appworld_root=appworld_root,
        experiment_name=f"{experiment_name}_worker_{worker_id}",
        metadata_path=metadata_path,
        checkpoint_root=checkpoint_root,
        unsupported_constraints=unsupported_constraints,
    )
    _PROCESS_RUNTIME.__enter__()
    _PROCESS_RUNTIME.warm_request_metadata()
    atexit.register(_close_process_runtime)


def _process_invoke(
    request: Mapping[str, Any],
    *,
    load_state: bool = True,
) -> tuple[dict[str, Any], str | None]:
    if _PROCESS_RUNTIME is None:
        raise RuntimeError("parallel AppWorld worker was not initialized")
    if load_state:
        _PROCESS_RUNTIME.load_state(str(request["state_handle"]))
    invoke_from_state = getattr(_PROCESS_RUNTIME, "invoke_from_state", None)
    if invoke_from_state is not None and request.get("state_digest") is not None:
        result = invoke_from_state(
            str(request["tool"]),
            dict(request.get("args") or {}),
            before_digest=str(request["state_digest"]),
        )
    else:
        result = _PROCESS_RUNTIME.invoke(
            str(request["tool"]), dict(request.get("args") or {})
        )
    saved_handle = None
    save_state_id = request.get("save_state_id")
    if save_state_id is not None and result.transition.changed:
        saved_handle = _PROCESS_RUNTIME.save_state(str(save_state_id))
    return result.to_dict(), saved_handle


def _process_invoke_batch(
    requests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute one coordinator-assigned k-sized batch in one worker IPC."""

    rows = []
    loaded_state_handle: str | None = None
    loaded_state_is_pristine = False
    for request in requests:
        try:
            state_handle = str(request["state_handle"])
            reuse_loaded_state = (
                loaded_state_is_pristine
                and loaded_state_handle == state_handle
            )
            reuse_fresh_base_state = (
                loaded_state_handle is None
                and bool(request.get("state_is_initial", False))
                and getattr(
                    _PROCESS_RUNTIME, "_current_checkpoint_handle", None
                ) is None
            )
            raw, saved_handle = _process_invoke(
                request,
                load_state=not (reuse_loaded_state or reuse_fresh_base_state),
            )
            loaded_state_handle = state_handle
            loaded_state_is_pristine = not bool(
                raw.get("transition", {}).get("changed", False)
            )
            outcome = None
            if "success_schema" in request and "failure_schema" in request:
                outcome = classify_outcome_schemas(
                    dict(request["success_schema"]),
                    dict(request["failure_schema"]),
                    InvocationResult.from_dict(raw),
                )
            rows.append({
                "ok": True,
                "result": raw,
                "saved_handle": saved_handle,
                "outcome": outcome,
            })
        except Exception as exc:
            # The failed call may have partially mutated its runtime. Reload
            # the requested immutable handle before any following call.
            loaded_state_is_pristine = False
            save_state_id = request.get("save_state_id")
            discard_state = getattr(_PROCESS_RUNTIME, "discard_state", None)
            if save_state_id is not None and discard_state is not None:
                discard_state(str(save_state_id))
            rows.append({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
    return rows


class AppWorldProcessInvocationPool:
    """Persistent spawn workers that never open the coordinator checkpoint."""

    def __init__(
        self,
        *,
        workers: int,
        appworld_root: str | Path,
        experiment_name: str,
        metadata_path: str | Path,
        checkpoint_root: str | Path,
        worker_root: str | Path,
        unsupported_constraints: Sequence[Mapping[str, str]],
    ) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._initargs = (
            str(appworld_root),
            experiment_name,
            str(metadata_path),
            str(checkpoint_root),
            str(worker_root),
            tuple(dict(row) for row in unsupported_constraints),
        )
        # One single-process executor per logical worker lets retries exclude
        # the worker that just failed while retaining persistent AppWorld
        # runtimes. A shared ProcessPoolExecutor cannot make that guarantee.
        self.executors = [
            self._new_executor()
            for _ in range(workers)
        ]
        # A blocking availability queue avoids the coordinator-wide 1 ms
        # polling loop that otherwise wakes every pending invocation while all
        # workers are busy.  At 2,880 calls per round that polling alone can
        # dominate short AppWorld invocations.
        self._available_workers: asyncio.Queue[int] = asyncio.Queue()
        self._worker_acquire_lock = asyncio.Lock()
        for worker in range(workers):
            self._available_workers.put_nowait(worker)

    def _new_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=self._context,
            initializer=_initialize_process_runtime,
            initargs=self._initargs,
        )

    async def _acquire_worker(self, avoid_worker: int | None) -> int:
        if avoid_worker is None or len(self.executors) <= 1:
            return await self._available_workers.get()
        # Serialize the tiny queue selection so concurrent retries cannot each
        # hold their failed worker while waiting for a replacement and deadlock.
        async with self._worker_acquire_lock:
            worker = await self._available_workers.get()
            if worker == avoid_worker:
                replacement = await self._available_workers.get()
                self._available_workers.put_nowait(worker)
                worker = replacement
            return worker

    async def invoke_with_worker(
        self,
        *,
        state_handle: str,
        state_digest: str | None = None,
        tool: str,
        args: Mapping[str, Any],
        save_state_id: str | None,
        avoid_worker: int | None = None,
    ) -> tuple[InvocationResult, str | None, int]:
        worker = await self._acquire_worker(avoid_worker)
        loop = asyncio.get_running_loop()
        try:
            raw, saved_handle = await loop.run_in_executor(
                self.executors[worker],
                _process_invoke,
                {
                    "state_handle": state_handle,
                    "state_digest": state_digest,
                    "tool": tool,
                    "args": json_safe(args),
                    "save_state_id": save_state_id,
                },
            )
        except Exception as exc:
            self.executors[worker].shutdown(wait=False, cancel_futures=True)
            self.executors[worker] = self._new_executor()
            setattr(exc, "worker_id", worker)
            raise
        finally:
            self._available_workers.put_nowait(worker)
        return InvocationResult.from_dict(raw), saved_handle, worker

    async def invoke(
        self,
        *,
        state_handle: str,
        state_digest: str | None = None,
        tool: str,
        args: Mapping[str, Any],
        save_state_id: str | None,
    ) -> tuple[InvocationResult, str | None]:
        result, saved_handle, _ = await self.invoke_with_worker(
            state_handle=state_handle,
            state_digest=state_digest,
            tool=tool,
            args=args,
            save_state_id=save_state_id,
        )
        return result, saved_handle

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
        """Send up to k calls to one persistent process in one IPC request."""

        worker = await self._acquire_worker(avoid_worker)
        loop = asyncio.get_running_loop()
        try:
            raw_rows = await loop.run_in_executor(
                self.executors[worker],
                _process_invoke_batch,
                [dict(request) for request in requests],
            )
        except Exception as exc:
            self.executors[worker].shutdown(wait=False, cancel_futures=True)
            self.executors[worker] = self._new_executor()
            setattr(exc, "worker_id", worker)
            raise
        finally:
            self._available_workers.put_nowait(worker)
        rows: list[
            tuple[
                InvocationResult | None,
                str | None,
                Mapping[str, Any] | None,
                str | None,
            ]
        ] = []
        for raw in raw_rows:
            if bool(raw.get("ok")):
                rows.append((
                    InvocationResult.from_dict(dict(raw["result"])),
                    raw.get("saved_handle"),
                    None,
                    raw.get("outcome"),
                ))
            else:
                rows.append((None, None, dict(raw), None))
        return rows, worker

    def close(self) -> None:
        # AppWorld teardown does real per-process work, so executors are shut
        # down concurrently rather than one after another.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, len(self.executors))
        ) as shutdown_pool:
            list(shutdown_pool.map(
                lambda executor: executor.shutdown(
                    wait=True, cancel_futures=True
                ),
                self.executors,
            ))


def compose_checkpoint_directory(
    source: Path,
    target: Path,
    *,
    parent: Path | None,
) -> None:
    shutil.rmtree(target, ignore_errors=True)
    if parent is not None:
        shutil.copytree(parent, target)
    else:
        target.mkdir(parents=True, exist_ok=True)
    for source_file in source.rglob("*"):
        if not source_file.is_file():
            continue
        relative = source_file.relative_to(source)
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if source_file.suffix == ".jsonl" and target_file.exists():
            with source_file.open("rb") as input_file, target_file.open("ab") as output_file:
                shutil.copyfileobj(input_file, output_file)
        else:
            shutil.copy2(source_file, target_file)


def regular_file_storage_bytes(*roots: Path) -> int:
    """Count unique regular-file bytes below explicit builder-owned roots."""

    total = 0
    visited: set[tuple[int, int]] = set()
    stack = [Path(root) for root in roots]
    while stack:
        path = stack.pop()
        try:
            metadata = path.lstat()
        except OSError:
            continue
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in visited:
            continue
        visited.add(identity)
        if stat.S_ISREG(metadata.st_mode):
            total += int(metadata.st_size)
        elif stat.S_ISDIR(metadata.st_mode):
            try:
                stack.extend(path.iterdir())
            except OSError:
                continue
    return total


def _schema_from_type(raw_type: Any) -> dict[str, Any]:
    aliases = {
        "str": "string", "string": "string",
        "int": "integer", "integer": "integer",
        "float": "number", "number": "number",
        "bool": "boolean", "boolean": "boolean",
        "dict": "object", "object": "object",
        "none": "null", "null": "null", "any": None,
    }
    parts = [part.strip().lower() for part in str(raw_type or "any").split("|")]
    schemas: list[dict[str, Any]] = []
    for part in parts:
        list_match = re.fullmatch(r"list\s*\[\s*([^\]]+)\s*\]", part)
        if list_match:
            schemas.append({"type": "array", "items": _schema_from_type(list_match.group(1))})
            continue
        json_type = aliases.get(part)
        schemas.append({} if json_type is None else {"type": json_type})
    if len(schemas) == 1:
        return schemas[0]
    return {"anyOf": schemas}


def _apply_constraints(
    schema: Mapping[str, Any], constraints: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    result = dict(schema)
    unsupported: list[str] = []
    integer = result.get("type") == "integer"
    for raw in constraints:
        text = str(raw).strip()
        enum_match = re.fullmatch(r"value\s+in\s+(.+)", text)
        if enum_match:
            try:
                values = ast.literal_eval(enum_match.group(1))
            except (SyntaxError, ValueError):
                values = None
            if isinstance(values, list | tuple):
                result["enum"] = json_safe(values)
                continue
        pieces = [piece.strip() for piece in text.split(",")]
        length_pattern = r"(?:length\s*)?(>=|<=|>|<)\s*(\d+)"
        if pieces and pieces[0].startswith("length") and all(
            re.fullmatch(length_pattern, piece) for piece in pieces
        ):
            for piece in pieces:
                operator, raw_value = re.fullmatch(length_pattern, piece).groups()  # type: ignore[union-attr]
                value = int(raw_value)
                if operator in {">", ">="}:
                    minimum = value + int(operator == ">")
                    result["minLength"] = max(int(result.get("minLength", 0)), minimum)
                else:
                    maximum = value - int(operator == "<")
                    result["maxLength"] = min(int(result.get("maxLength", maximum)), maximum)
            continue
        number_pattern = r"(?:value\s*)?(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)"
        if pieces and pieces[0].startswith("value") and all(
            re.fullmatch(number_pattern, piece) for piece in pieces
        ):
            for piece in pieces:
                operator, raw_value = re.fullmatch(number_pattern, piece).groups()  # type: ignore[union-attr]
                value = float(raw_value)
                if operator == ">":
                    value = value + 1 if integer else math.nextafter(value, math.inf)
                    result["minimum"] = max(float(result.get("minimum", value)), value)
                elif operator == ">=":
                    result["minimum"] = max(float(result.get("minimum", value)), value)
                elif operator == "<":
                    value = value - 1 if integer else math.nextafter(value, -math.inf)
                    result["maximum"] = min(float(result.get("maximum", value)), value)
                else:
                    result["maximum"] = min(float(result.get("maximum", value)), value)
            continue
        unsupported.append(text)
    return result, unsupported


def load_appworld_tools(
    metadata_path: str | Path = DEFAULT_METADATA,
    *,
    selected_tools: set[str] | None = None,
) -> tuple[list[ToolDefinition], dict[str, Mapping[str, Any]], str, list[dict[str, str]]]:
    path = Path(metadata_path)
    raw_metadata = json.loads(path.read_text(encoding="utf-8"))
    definitions: list[ToolDefinition] = []
    public: dict[str, Mapping[str, Any]] = {}
    unsupported: list[dict[str, str]] = []
    for tool, row in sorted(raw_metadata.items()):
        if selected_tools is not None and tool not in selected_tools:
            continue
        parameters: list[ToolParameter] = []
        public_parameters: list[dict[str, Any]] = []
        for parameter in row.get("parameters") or ():
            schema, missing = _apply_constraints(
                _schema_from_type(parameter.get("type")),
                tuple(map(str, parameter.get("constraints") or ())),
            )
            name = str(parameter.get("name") or "")
            required = bool(parameter.get("required", False))
            if not required and parameter.get("default") is not None:
                schema = {**schema, "default": json_safe(parameter.get("default"))}
            parameters.append(ToolParameter(name=name, required=required, schema=schema))
            public_parameters.append({
                "name": name,
                "required": required,
                "description": str(parameter.get("description") or ""),
                "schema": schema,
            })
            unsupported.extend(
                {"tool": tool, "parameter": name, "constraint": constraint}
                for constraint in missing
            )
        public_row = {
            "tool": tool,
            "method": row.get("method"),
            "path": row.get("path"),
            "parameters": public_parameters,
            "response_schemas": json_safe(row.get("response_schemas") or {}),
        }
        definitions.append(ToolDefinition(
            tool=tool,
            parameters=tuple(parameters),
            public_metadata=public_row,
        ))
        public[tool] = public_row
    if selected_tools is not None:
        missing_tools = selected_tools - set(public)
        if missing_tools:
            raise ValueError(f"unknown selected tools: {sorted(missing_tools)}")
    metadata_id = f"appworld_metadata_v1:{path.parent.name}:{len(raw_metadata)}"
    return definitions, public, metadata_id, unsupported


class AppWorldStatefulRuntime:
    runtime_id = APPWORLD_ADAPTER_VERSION

    def __init__(
        self,
        *,
        appworld_root: str | Path = DEFAULT_APPWORLD_ROOT,
        experiment_name: str = "tool_dependency_v1",
        metadata_path: str | Path = DEFAULT_METADATA,
        checkpoint_root: str | Path | None = None,
        unsupported_constraints: Sequence[Mapping[str, str]] = (),
    ) -> None:
        self.appworld_root = Path(appworld_root)
        self.experiment_name = experiment_name
        self.metadata_path = Path(metadata_path)
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root else None
        self.unsupported_constraints = tuple(dict(row) for row in unsupported_constraints)
        self.world: Any | None = None
        self._current_checkpoint_handle: str | None = None
        self._checkpoint_handle_bytes: dict[str, int] = {}
        self._checkpoint_root_file_bytes = 0
        self._checkpoint_storage_bytes = 0
        self._checkpoint_storage_initialized = False
        self._accounted_checkpoint_root: Path | None = None
        self._storage_accounting_enabled = False

    def __enter__(self) -> "AppWorldStatefulRuntime":
        src_path = str(self.appworld_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        warnings.filterwarnings("ignore", message=".*ORJSONResponse is deprecated.*")
        warnings.filterwarnings("ignore", message=".*HMAC key is .* below the minimum recommended length.*")
        warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
        os.environ.setdefault("APPWORLD_ROOT", str(self.appworld_root))
        missing = [
            package for package in ("orjson", "fastapi", "sqlmodel")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise RuntimeError("missing AppWorld dependencies: " + ", ".join(missing))
        from appworld import AppWorld

        self.world = AppWorld(
            task_id=BASE_FIXTURE,
            experiment_name=self.experiment_name,
            load_ground_truth=False,
            raise_on_failure=False,
            max_api_calls_per_interaction=100_000,
            timeout_seconds=60,
            show_api_response_schemas=True,
            # Invocation arguments are constructed exclusively from the
            # frozen registry ToolDefinition. Rebuilding a second OpenAPI view
            # merely to re-check their names costs hundreds of milliseconds
            # per newly seen app in every process. FastAPI still performs the
            # authoritative request/body/path validation below.
            raise_on_extra_parameters=False,
        )
        self._reset_db_trackers()
        return self

    def _refresh_checkpoint_storage_bytes(self) -> int:
        """Initialize immutable-state accounting with one explicit root scan."""

        self._checkpoint_handle_bytes = {}
        self._checkpoint_root_file_bytes = 0
        if self.checkpoint_root is not None and self.checkpoint_root.exists():
            for child in self.checkpoint_root.iterdir():
                if child.is_dir():
                    self._checkpoint_handle_bytes[child.name] = (
                        regular_file_storage_bytes(child)
                    )
                else:
                    self._checkpoint_root_file_bytes += (
                        regular_file_storage_bytes(child)
                    )
        self._checkpoint_storage_bytes = sum(
            self._checkpoint_handle_bytes.values()
        ) + self._checkpoint_root_file_bytes
        self._checkpoint_storage_initialized = True
        self._accounted_checkpoint_root = self.checkpoint_root
        return self._checkpoint_storage_bytes

    def storage_usage_bytes(self) -> int:
        """Return cached bytes owned by the immutable checkpoint root.

        A save scans only its newly composed handle and a discard subtracts
        that handle's cached size.  Budget checks therefore never recursively
        walk the complete state tree on each tool invocation.
        """

        self._storage_accounting_enabled = True
        if (
            not self._checkpoint_storage_initialized
            or self._accounted_checkpoint_root != self.checkpoint_root
        ):
            return self._refresh_checkpoint_storage_bytes()
        return self._checkpoint_storage_bytes

    def _changed_app_names(self) -> tuple[str, ...]:
        if self.world is None or self.world.models is None:
            return ()
        return tuple(sorted(
            str(owner)
            for owner, models in self.world.models.items()
            if models.SQLModel.db.tracker.changes
        ))

    @staticmethod
    def _validate_changed_app_deltas(
        source: Path, changed_apps: Sequence[str]
    ) -> None:
        """Reject a missing/empty raw delta before composing a state handle."""

        for owner in changed_apps:
            delta = source / f"{owner}.jsonl"
            try:
                size = delta.stat().st_size
            except FileNotFoundError as error:
                raise RuntimeError(
                    f"missing changed-app checkpoint delta: {owner}"
                ) from error
            if size == 0:
                raise RuntimeError(
                    f"zero-byte changed-app checkpoint delta: {owner}"
                )

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        self.close()

    def save_state(self, state_id: str) -> str:
        if self.world is None:
            raise RuntimeError("AppWorld runtime is not open")
        if self._storage_accounting_enabled:
            self.storage_usage_bytes()
        changed_apps = self._changed_app_names()
        output_root = Path(self.world.output_checkpoints_directory)
        if (
            self.checkpoint_root is not None
            and output_root.resolve() == self.checkpoint_root.resolve()
        ):
            raise RuntimeError("private and external checkpoint roots overlap")
        try:
            handle = self.world.save_state(state_id)
        except BaseException:
            self.discard_state(state_id)
            raise
        if not isinstance(handle, str):
            self.discard_state(state_id)
            raise RuntimeError("AppWorld did not return a checkpoint handle")
        source = output_root / handle
        try:
            self._validate_changed_app_deltas(source, changed_apps)
        except BaseException:
            self.discard_state(handle)
            raise
        if self.checkpoint_root is not None:
            target = self.checkpoint_root / handle
            parent = (
                self.checkpoint_root / self._current_checkpoint_handle
                if self._current_checkpoint_handle else None
            )
            if source.resolve() == target.resolve():
                raise RuntimeError("private and external checkpoint paths overlap")
            try:
                compose_checkpoint_directory(source, target, parent=parent)
                # AppWorld's private directory contains only this call's delta.
                # Once the complete immutable checkpoint is durable externally,
                # retaining that delta only consumes each worker's jobtmp quota.
                shutil.rmtree(source)
            except BaseException:
                self.discard_state(handle)
                raise
            if self._storage_accounting_enabled:
                previous_bytes = self._checkpoint_handle_bytes.get(handle, 0)
                handle_bytes = regular_file_storage_bytes(target)
                self._checkpoint_handle_bytes[handle] = handle_bytes
                self._checkpoint_storage_bytes += handle_bytes - previous_bytes
        self._reset_db_trackers()
        self._current_checkpoint_handle = handle
        return handle

    def discard_state(self, state_handle: str) -> None:
        if not state_handle or Path(state_handle).name != state_handle:
            raise ValueError(f"unsafe checkpoint handle: {state_handle!r}")
        if self._storage_accounting_enabled:
            self.storage_usage_bytes()
        if self.checkpoint_root is not None:
            cached_bytes = self._checkpoint_handle_bytes.get(state_handle, 0)
            try:
                shutil.rmtree(self.checkpoint_root / state_handle)
            except FileNotFoundError:
                pass
            if self._storage_accounting_enabled:
                self._checkpoint_handle_bytes.pop(state_handle, None)
                self._checkpoint_storage_bytes -= cached_bytes
        if self.world is not None:
            shutil.rmtree(
                Path(self.world.output_checkpoints_directory) / state_handle,
                ignore_errors=True,
            )
        if self._current_checkpoint_handle == state_handle:
            self._current_checkpoint_handle = None

    def state_handle_available(self, state_handle: str) -> bool:
        if self.checkpoint_root is not None:
            return (self.checkpoint_root / state_handle).is_dir()
        if self.world is None:
            return False
        return (
            Path(self.world.output_checkpoints_directory) / state_handle
        ).is_dir()

    def load_state(self, state_handle: str) -> None:
        if self.world is None:
            raise RuntimeError("AppWorld runtime is not open")
        output_root = self.world.output_checkpoints_directory
        if self.checkpoint_root is not None:
            source = self.checkpoint_root / state_handle
            if not source.exists():
                raise FileNotFoundError(f"missing external checkpoint: {source}")
            # AppWorld.load_state only reads
            # output_checkpoints_directory/<handle>.  Point that public lookup
            # at the shared immutable checkpoint root for this synchronous
            # load, then restore the worker's private output directory before
            # any later save.  This avoids copying or linking a state tree into
            # every worker and leaves AppWorld itself unchanged.
            self.world.output_checkpoints_directory = str(self.checkpoint_root)
        try:
            self.world.load_state(state_handle)
        finally:
            self.world.output_checkpoints_directory = output_root
        self.world._set_datetime()
        # Loading an immutable state is orchestration, not part of the next
        # tool's transition. Keep SQLAlchemy listeners installed and discard
        # only any bookkeeping changes caused by the load itself.
        self._reset_db_trackers()
        self._current_checkpoint_handle = state_handle

    def snapshot(self) -> StateSnapshot:
        if self.world is None or self.world.models is None:
            raise RuntimeError("AppWorld model state is unavailable")
        components = {
            str(name): value
            for name, value in sorted(self.world.models.model_hashes().items())
        }
        return StateSnapshot(
            digest=stable_id("snapshot", components),
            components=components,
        )

    def _reset_db_trackers(self) -> None:
        if self.world is None or self.world.models is None:
            return
        for models in self.world.models.values():
            models.SQLModel.db.tracker.reset()

    def _changes(self) -> dict[str, Any]:
        if self.world is None or self.world.models is None:
            return {}
        changes: dict[str, Any] = {}
        # ModelCollection.restart_db_tracker() removes and re-adds a SQLAlchemy
        # listener for every AppWorld app. Doing that before every invocation
        # dominates short calls. The listener can remain installed: serialize
        # the non-empty lists here and let save_state/load_state clear them at
        # the persistence boundary.
        for owner, models in sorted(self.world.models.items()):
            tracker = models.SQLModel.db.tracker
            if tracker.changes:
                changes[str(owner)] = json_safe(tracker.changes)
            # AppWorld.save_state(format="changes") consumes this same
            # tracker immediately after a changed invocation. Clearing it
            # here makes the externally mirrored checkpoint an empty delta,
            # even though the transition evidence above is correct. The
            # tracker is cleared by save_state() after the delta is durable,
            # or by the next load_state() for calls whose branch is discarded.
        return changes

    def invoke(self, tool: str, args: Mapping[str, Any]) -> InvocationResult:
        """Invoke without a trusted coordinator state identity."""

        return self._invoke_with_before_digest(
            tool,
            args,
            before_digest=self.snapshot().digest,
            skip_unchanged_snapshot=False,
        )

    def invoke_from_state(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        before_digest: str,
    ) -> InvocationResult:
        """Invoke from an immutable handle whose digest is already known.

        The coordinator owns immutable search states and passes their recorded
        digest with every work item. AppWorld's DB tracker is also the source
        used to serialize transition changes, so an empty change set can reuse
        that digest without hashing every model database before and after each
        call. A real mutation still computes the post-call model snapshot.
        """

        return self._invoke_with_before_digest(
            tool,
            args,
            before_digest=before_digest,
            skip_unchanged_snapshot=True,
        )

    def _invoke_with_before_digest(
        self,
        tool: str,
        args: Mapping[str, Any],
        *,
        before_digest: str,
        skip_unchanged_snapshot: bool,
    ) -> InvocationResult:
        if self.world is None or self.world.models is None:
            raise RuntimeError("AppWorld runtime is not open")
        if "." not in tool:
            raise ValueError(f"invalid AppWorld dispatch id: {tool!r}")
        group, operation = tool.split(".", 1)
        # Trackers were cleared after the previous invocation or state load;
        # their SQLAlchemy listeners stay active across calls.
        response: Any = None
        machine: dict[str, Any]
        try:
            raw = self.world.requester._request(
                _app_name=group,
                _api_name=operation,
                raise_on_failure=False,
                track=False,
                **dict(args),
            )
            status_code = int(raw.status_code)
            response = self.world.requester.response_to_json(raw)
            status = "success" if 200 <= status_code < 300 else "rejected"
            machine = {"status_code": status_code}
        except Exception as exc:
            status = "error"
            machine = {"exception_type": type(exc).__name__, "exception": str(exc)}
        changes = self._changes()
        after_digest = (
            before_digest
            if skip_unchanged_snapshot and not changes
            else self.snapshot().digest
        )
        return InvocationResult(
            status=status,
            response=json_safe(response),
            machine_status=machine,
            transition=TransitionDelta(before_digest, after_digest, changes),
        )

    def source_metadata(self) -> Mapping[str, Any]:
        return {
            "adapter": APPWORLD_ADAPTER_VERSION,
            "base_fixture": BASE_FIXTURE,
            "base_fixture_has_instruction": False,
            "metadata_id": f"appworld_metadata_v1:{self.metadata_path.parent.name}",
            "unsupported_formal_constraints": len(self.unsupported_constraints),
        }

    def warm_request_metadata(self) -> None:
        """Populate AppWorld's per-process API-doc caches before measured work.

        Requester lazily builds two independent FastAPI/Pydantic documentation
        views the first time each app is invoked. A short first graph frontier
        cannot touch every app on every persistent worker, so without explicit
        warming those multi-second cache misses leak into round two. This uses
        registry metadata only and executes no tool or benchmark task.
        """

        if self.world is None:
            raise RuntimeError("AppWorld runtime is not open")
        from appworld.apps import get_all_apps
        from appworld.requester import _get_api_name_to_doc

        for app_name in get_all_apps(
            skip_admin=False,
            skip_api_docs=False,
            skip_supervisor=False,
        ):
            _get_api_name_to_doc(app_name, include_private_apis=True)

    def close(self) -> None:
        if self.world is None:
            return
        try:
            self.world.close()
        except Exception as exc:
            warnings.warn(f"AppWorld cleanup did not finish cleanly: {exc}", RuntimeWarning)
        finally:
            self.world = None


def build_graph(args: argparse.Namespace) -> dict[str, Any]:
    observation_store = args.work_dir / "observations.sqlite"
    state_root = args.work_dir / "states"
    worker_root = args.work_dir / "workers"
    shutil.rmtree(state_root, ignore_errors=True)
    shutil.rmtree(worker_root, ignore_errors=True)
    observation_store.unlink(missing_ok=True)
    definitions, metadata, metadata_id, unsupported = load_appworld_tools(
        args.metadata
    )
    manifest, _ = load_active_contract_manifest(
        args.contracts,
        metadata,
        metadata_id=metadata_id,
        adapter=APPWORLD_CONTRACT_ADAPTER,
    )
    active_tools = {definition.tool for definition in definitions}
    manifest = manifest.model_copy(update={
        "contracts": {
            tool: contract for tool, contract in manifest.contracts.items()
            if tool in active_tools
        },
        "unknown": {
            tool: error for tool, error in manifest.unknown.items()
            if tool in active_tools
        },
        "audit": {
            tool: rows for tool, rows in manifest.audit.items()
            if tool in active_tools
        },
    })
    generator = None
    if args.mode == "fresh":
        generator = ContractGenerator(
            DeepSeekFlash(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
                model="deepseek-v4-flash",
            ),
            feedback_policy=DEFAULT_FEEDBACK_POLICY,
            template_policy=MINIMAL_SUCCESS_TEMPLATE_POLICY,
            adapter=APPWORLD_CONTRACT_ADAPTER,
        )
    select_template_policy(manifest, MINIMAL_SUCCESS_TEMPLATE_POLICY)
    manifest.feedback_policy = DEFAULT_FEEDBACK_POLICY
    contract_set = OnlineContractSet(
        manifest=manifest,
        metadata=metadata,
        generator=generator,
        count_all_history=True,
        max_total_llm_calls=3 * len(definitions),
    )
    field_store: SQLiteFieldEmbeddingStore | None = None
    scheduling_store: SQLiteFieldEmbeddingStore | None = None
    store: SQLiteBuildStore | None = None
    try:
        field_store = SQLiteFieldEmbeddingStore(
            args.field_embeddings,
            base_url=args.embedding_base_url,
        )
        if field_store.descriptor_mode != "field":
            raise ValueError("argument binding requires field descriptors")
        embedder = QwenVllmTextEmbedder(base_url=field_store.base_url)
        field_priority = SQLiteFieldPriorityIndex(
            field_store,
            definitions,
            embedder=embedder,
        )
        scheduling_store = SQLiteFieldEmbeddingStore(
            args.scheduling_field_embeddings,
            base_url=args.embedding_base_url,
        )
        if scheduling_store.descriptor_mode != "context":
            raise ValueError("global scheduling requires context descriptors")
        scheduling_embedder = QwenVllmTextEmbedder(
            base_url=scheduling_store.base_url
        )
        scheduling_priority = SQLiteFieldPriorityIndex(
            scheduling_store,
            definitions,
            embedder=scheduling_embedder,
        )
        store = SQLiteBuildStore(
            observation_store,
            persist_tasks=False,
            persist_calls=False,
        )
        with AppWorldStatefulRuntime(
            appworld_root=args.appworld_root,
            experiment_name=f"dependency-{args.output.parent.name}",
            metadata_path=args.metadata,
            checkpoint_root=state_root,
            unsupported_constraints=unsupported,
        ) as runtime:
            build_config = BuildConfig(
                budget=SearchBudget(
                    max_calls=args.max_calls,
                    max_states=2_000,
                    max_depth=12,
                    max_seconds=82_800,
                    max_external_model_calls=3 * len(definitions),
                    max_contract_repair_calls=(
                        0 if generator is None else max(
                            0,
                            3 * len(definitions)
                            - contract_set.total_llm_calls(),
                        )
                    ),
                    max_retries=100_000,
                ),
                trial_repetitions=2,
                max_observation_bytes=65_536,
                serial_state_batch_size=1,
            )
            worker_root.mkdir(parents=True, exist_ok=True)
            invocation_pool = AppWorldProcessInvocationPool(
                workers=96,
                appworld_root=args.appworld_root,
                experiment_name=f"dependency-{args.output.parent.name}",
                metadata_path=args.metadata,
                checkpoint_root=state_root,
                worker_root=worker_root,
                unsupported_constraints=unsupported,
            )
            builder = ParallelDependencyGraphBuilder(
                runtime,
                invocation_pool,
                definitions,
                contract_set,
                store=store,
                config=build_config,
                parallel=ParallelBuildConfig(
                    workers=96,
                    calls_per_worker_epoch=30,
                    worker_ipc_batch_size=30,
                    prefetch_rounds=1,
                    prefetch_calls_per_worker=8,
                    contract_epoch_calls=2_880,
                    fairness_share=0.20,
                    worker_retries=2,
                    worker_failure_threshold=0.02,
                    telemetry_path=args.telemetry_output,
                    heartbeat_interval_seconds=60.0,
                    storage_paths=(args.work_dir,),
                    contract_repair_concurrency=16,
                ),
                field_priority=field_priority,
                scheduling_priority=scheduling_priority,
            )
            try:
                graph = asyncio.run(builder.run_async())
            finally:
                invocation_pool.close()
            builder.write_outputs(
                graph,
                graph_path=args.output,
                evidence_path=args.work_dir / "evidence.jsonl",
            )
            contract_set.manifest.write(args.contracts)
            return graph
    finally:
        if scheduling_store is not None:
            scheduling_store.close()
        if field_store is not None:
            field_store.close()
        if store is not None:
            store.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "warm"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--appworld-root", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--field-embeddings", type=Path, required=True)
    parser.add_argument("--scheduling-field-embeddings", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--embedding-base-url", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    graph = build_graph(args)
    required = sum(bool(edge["required"]) for edge in graph["edges"])
    value_required = sum(
        bool(edge["required"]) and edge["kind"] == "input"
        for edge in graph["edges"]
    )
    state_required = sum(edge["kind"] == "state" for edge in graph["edges"])
    print(json.dumps({
        "version": graph["version"],
        "saved_graph": args.output,
        "saved_evidence": args.work_dir / "evidence.jsonl",
        "tools": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "required_edges": required,
        "required_value_edges": value_required,
        "required_state_edges": state_required,
        "compatible_edges": len(graph["edges"]) - required,
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
