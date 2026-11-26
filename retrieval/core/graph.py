"""Environment-independent discovery of contract-verified tool dependencies.

Contracts provide caller-constructible seed arguments, expose observation
holes, and select reusable values with JSONPath. Runtime search is systematic:
each parameter domain is ordered by field similarity and complete assignments
are dovetailed by the sum of their per-field ranks. Tool names, application
namespaces, gold data, and graph statistics do not otherwise influence
candidates.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import math
import os
import re
import sqlite3
import time
import zlib
from copy import deepcopy
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator, SchemaError

from retrieval.core.contracts import (
    CallTemplate,
    MINIMAL_SUCCESS_TEMPLATE_POLICY,
    OnlineContractSet,
    ToolCallContract,
    canonical_json,
    select_template_policy,
    select_export_values,
    with_minimal_success_baseline,
)


GRAPH_VERSION = "tool_dependency_graph_v5"
EVIDENCE_VERSION = "tool_dependency_evidence_v12"
INVOCATION_STATUSES = {"success", "rejected", "error"}
OUTCOMES = {"success", "failure", "unknown", "conflict"}

# These words describe JSON structure, not the entity carried by a value.
# Matching ``id`` against every ``*_id`` would create false edges (for example
# album_id -> label_id).
GENERIC_FIELD_TOKENS = frozenset({
    "argument", "count", "id", "ids", "index", "item", "items",
    "name", "number", "response", "value",
})
_JSONPATH_FIELD_COMPONENT = re.compile(
    r"\.([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_-])"
    r'''|\[\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*\]'''
)
_JSONPATH_COLLECTION_SELECTOR = re.compile(r"\[\s*(?:\*|[0-9]+)\s*\]")
_UNSUPPORTED_PATH_CONTEXT = ("..", "?", "@", ":", ",", "(", ")", "|")


def _normalize_collection_component(value: str) -> str:
    lowered = value.lower()
    if (
        len(value) > 2
        and lowered.endswith("s")
        and not lowered.endswith(("ss", "es", "is", "us"))
    ):
        return value[:-1]
    return value


@lru_cache(maxsize=65_536)
def specific_field_tokens(value: str) -> frozenset[str]:
    """Return the entity-bearing tokens in a public field label."""

    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return frozenset(
        token
        for raw in re.findall(r"[A-Za-z]+|[0-9]+", separated)
        if (token := raw.lower()) not in GENERIC_FIELD_TOKENS
    )


@lru_cache(maxsize=65_536)
def specific_jsonpath_tokens(path: str) -> frozenset[str]:
    """Return entity tokens from the last two fields of a simple selector."""

    if not path or any(marker in path for marker in _UNSUPPORTED_PATH_CONTEXT):
        return frozenset()
    matches = tuple(_JSONPATH_FIELD_COMPONENT.finditer(path))
    if not matches:
        return frozenset()
    tokens: set[str] = set()
    for index in range(max(0, len(matches) - 2), len(matches)):
        match = matches[index]
        name = str(match.group(1) or match.group(2))
        next_start = (
            matches[index + 1].start()
            if index + 1 < len(matches) else len(path)
        )
        if _JSONPATH_COLLECTION_SELECTOR.search(path[match.end():next_start]):
            name = _normalize_collection_component(name)
        tokens.update(specific_field_tokens(name))
    return frozenset(tokens)


def fields_share_specific_semantics(
    source_export: str,
    source_declared_path: str,
    target_parameter: str,
) -> bool:
    """Whether source and target name the same non-structural entity."""

    source = specific_field_tokens(source_export)
    source |= specific_jsonpath_tokens(source_declared_path)
    return bool(source & specific_field_tokens(target_parameter))


def stable_id(prefix: str, payload: Any) -> str:
    serialized = canonical_json(payload).encode("utf-8")
    encoded = base64.urlsafe_b64encode(
        zlib.compress(serialized, level=9)
    ).decode("ascii").rstrip("=")
    return f"{prefix}_{encoded}"


def selected_template_fingerprint(template: CallTemplate) -> str:
    """Identify the exact template that selected an invocation's arguments."""

    return stable_id(
        "selected_template",
        template.model_dump(mode="json", by_alias=True),
    )


def output_field_id(
    *,
    tool: str,
    location: str,
    export_name: str,
    field_path: str,
    value_schema: Mapping[str, Any],
) -> str:
    """Identify one declared output field independently of contract revision."""

    return stable_id("output_field", {
        "tool": tool,
        "location": location,
        "export_name": export_name,
        "field_path": field_path,
        "schema": json_safe(value_schema),
    })


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted((json_safe(child) for child in value), key=canonical_json)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return repr(value)


def system_monotonic() -> float:
    clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    return time.clock_gettime(clock) if clock is not None else time.monotonic()


def iter_json_values(value: Any, pointer: str = "") -> Iterator[tuple[str, Any]]:
    safe = json_safe(value)
    yield pointer or "/", safe
    if isinstance(safe, Mapping):
        for key, child in safe.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from iter_json_values(child, f"{pointer}/{escaped}")
    elif isinstance(safe, list):
        for index, child in enumerate(safe):
            yield from iter_json_values(child, f"{pointer}/{index}")


def iter_scalar_jsonpaths(value: Any, path: str = "$") -> Iterator[dict[str, Any]]:
    """Describe observable scalar locations without exposing their values."""
    safe = json_safe(value)
    if isinstance(safe, Mapping):
        for key, child in safe.items():
            yield from iter_scalar_jsonpaths(
                child, f"{path}[{json.dumps(str(key), ensure_ascii=False)}]"
            )
    elif isinstance(safe, list):
        for index, child in enumerate(safe):
            yield from iter_scalar_jsonpaths(child, f"{path}[{index}]")
    else:
        yield {"path": path, "type": infer_json_schema(safe).get("type")}


def infer_json_schema(value: Any) -> dict[str, Any]:
    safe = json_safe(value)
    if safe is None:
        return {"type": "null"}
    if isinstance(safe, bool):
        return {"type": "boolean"}
    if isinstance(safe, int):
        return {"type": "integer"}
    if isinstance(safe, float):
        return {"type": "number"}
    if isinstance(safe, str):
        return {"type": "string"}
    if isinstance(safe, list):
        return {"type": "array"}
    return {"type": "object"}


@lru_cache(maxsize=None)
def _schema_validator(serialized_schema: str) -> Draft202012Validator:
    schema = json.loads(serialized_schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=262_144)
def _schema_accepts_serialized(
    serialized_schema: str, serialized_value: str
) -> bool:
    """Cache exact validation for repeated field/value compatibility checks."""

    try:
        return _schema_validator(serialized_schema).is_valid(
            json.loads(serialized_value)
        )
    except SchemaError:
        return False


def schema_accepts(schema: Mapping[str, Any], value: Any) -> bool:
    return _schema_accepts_serialized(
        canonical_json(schema), canonical_json(json_safe(value))
    )


@dataclass(frozen=True)
class ToolParameter:
    name: str
    required: bool
    schema: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "schema": json_safe(self.schema),
        }


@dataclass(frozen=True)
class ToolDefinition:
    tool: str
    parameters: tuple[ToolParameter, ...]
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class StateSnapshot:
    digest: str
    components: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"digest": self.digest, "components": json_safe(self.components)}


@dataclass(frozen=True)
class TransitionDelta:
    before_digest: str
    after_digest: str
    changes: Any

    @property
    def changed(self) -> bool:
        return self.before_digest != self.after_digest or bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "changes": json_safe(self.changes),
            "changed": self.changed,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "TransitionDelta":
        return cls(
            before_digest=str(row["before_digest"]),
            after_digest=str(row["after_digest"]),
            changes=row.get("changes"),
        )


@dataclass(frozen=True)
class InvocationResult:
    status: str
    response: Any
    machine_status: Mapping[str, Any]
    transition: TransitionDelta

    def __post_init__(self) -> None:
        if self.status not in INVOCATION_STATUSES:
            raise ValueError(f"unsupported invocation status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "response": json_safe(self.response),
            "machine_status": json_safe(self.machine_status),
            "transition": self.transition.to_dict(),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "InvocationResult":
        return cls(
            status=str(row["status"]),
            response=row.get("response"),
            machine_status=dict(row.get("machine_status") or {}),
            transition=TransitionDelta.from_dict(row["transition"]),
        )


def normalized_invocation(result: InvocationResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "output": json_safe(result.response),
        "machine": json_safe(result.machine_status),
        "state_changed": result.transition.changed,
        "changes": json_safe(result.transition.changes),
    }


@lru_cache(maxsize=4096)
def _outcome_validators(
    success_schema: str, failure_schema: str
) -> tuple[Draft202012Validator, Draft202012Validator]:
    return (
        Draft202012Validator(json.loads(success_schema)),
        Draft202012Validator(json.loads(failure_schema)),
    )


def classify_outcome_schemas(
    success_schema: Mapping[str, Any],
    failure_schema: Mapping[str, Any],
    result: InvocationResult,
) -> str:
    record = normalized_invocation(result)
    success_validator, failure_validator = _outcome_validators(
        canonical_json(success_schema),
        canonical_json(failure_schema),
    )
    success = success_validator.is_valid(record)
    failure = failure_validator.is_valid(record)
    if success and failure:
        return "conflict"
    if success:
        return "success"
    if failure:
        return "failure"
    # A transport-level rejection is a formal negative outcome even when a
    # contract's response-shape schema is narrower than the framework's own
    # validation/authentication error payload.  Runtime exceptions remain
    # unknown: infrastructure failure is not evidence about tool semantics.
    if result.status == "rejected":
        return "failure"
    return "unknown"


def classify_outcome(contract: ToolCallContract, result: InvocationResult) -> str:
    return classify_outcome_schemas(
        contract.success_schema, contract.failure_schema, result
    )


def effective_invocation_outcome(
    result: InvocationResult, schema_outcome: str
) -> str:
    """Allow a transport success through an incomplete outcome contract.

    This is deliberately narrower than trusting transport status in general:
    a schema conflict or a schema-classified failure remains formal evidence,
    and rejected/error transports retain their existing outcomes.  Individual
    exports are still selected and JSON-Schema validated independently.
    """

    if result.status == "success" and schema_outcome == "unknown":
        return "success"
    return schema_outcome


def _explicit_value_contradiction(
    schema: Mapping[str, Any], value: Any
) -> bool:
    """Whether a present value directly contradicts a declared const/enum."""

    if "const" in schema and canonical_json(value) != canonical_json(schema["const"]):
        return True
    if isinstance(schema.get("enum"), list) and all(
        canonical_json(value) != canonical_json(candidate)
        for candidate in schema["enum"]
    ):
        return True
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return any(
                name in value
                and isinstance(child, Mapping)
                and _explicit_value_contradiction(child, value[name])
                for name, child in properties.items()
            )
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return any(
            _explicit_value_contradiction(schema["items"], child)
            for child in value
        )
    return False


def structural_feedback(
    *,
    example_expected: str,
    result: InvocationResult,
    observed_outcome: str,
) -> dict[str, Any]:
    def shape(value: Any, depth: int = 0) -> Any:
        if depth >= 4:
            return infer_json_schema(value).get("type")
        safe = json_safe(value)
        if isinstance(safe, Mapping):
            return {key: shape(child, depth + 1) for key, child in safe.items()}
        if isinstance(safe, list):
            variants: dict[str, Any] = {}
            for child in safe[:16]:
                child_shape = shape(child, depth + 1)
                variants.setdefault(canonical_json(child_shape), child_shape)
            return {"type": "array", "items": list(variants.values())[:4]}
        return infer_json_schema(safe).get("type")

    return {
        "expected": example_expected,
        "classified": observed_outcome,
        "transport_status": result.status,
        # Contract JIT is allowed to inspect the complete return from this
        # same tool.  Only the separate field-embedding client is restricted
        # to value-free contract descriptors.
        "invocation_result": normalized_invocation(result),
        "output_shape": shape(result.response),
        "machine_shape": shape(result.machine_status),
        "state_changed": result.transition.changed,
        "change_shape": shape(result.transition.changes),
    }


class BranchingRuntime(Protocol):
    runtime_id: str

    def save_state(self, state_id: str) -> str: ...

    def load_state(self, state_handle: str) -> None: ...

    def snapshot(self) -> StateSnapshot: ...

    def invoke(self, tool: str, args: Mapping[str, Any]) -> InvocationResult: ...

    def source_metadata(self) -> Mapping[str, Any]: ...


class StorageUsage(Protocol):
    """Return current builder-owned storage without a full-tree hot-path scan."""

    def __call__(self) -> int: ...


@dataclass(frozen=True)
class Observation:
    id: str
    type_id: str
    field_id: str
    value: Any
    source_tool: str
    source_call: str
    source_revision: int
    location: str
    export_name: str
    path: str
    # The contract-declared selector is stable public structure. ``path`` is
    # the concrete runtime match and may contain indexes or data-derived map
    # keys, so semantic ranking must never treat it as field metadata.
    field_path: str | None = None

    @classmethod
    def create(
        cls,
        value: Any,
        *,
        source_tool: str,
        source_call: str,
        source_revision: int,
        location: str,
        export_name: str,
        path: str,
        value_schema: Mapping[str, Any],
        field_path: str | None = None,
        type_path: str | None = None,
    ) -> "Observation":
        safe = json_safe(value)
        field_id = output_field_id(
            tool=source_tool,
            location=location,
            export_name=export_name,
            field_path=field_path or path,
            value_schema=value_schema,
        )
        type_payload = {
            "field_id": field_id,
        }
        if type_path is not None:
            type_payload["match_path"] = type_path
        type_id = stable_id("observation_type", type_payload)
        return cls(
            id=stable_id("observation", {
                "type_id": type_id,
                "value": safe,
                "path": path,
            }),
            type_id=type_id,
            field_id=field_id,
            value=safe,
            source_tool=source_tool,
            source_call=source_call,
            source_revision=source_revision,
            location=location,
            export_name=export_name,
            path=path,
            field_path=field_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type_id": self.type_id,
            "field_id": self.field_id,
            "value": json_safe(self.value),
            "source_tool": self.source_tool,
            "source_call": self.source_call,
            "source_revision": self.source_revision,
            "location": self.location,
            "export_name": self.export_name,
            "path": self.path,
            "field_path": self.field_path,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "Observation":
        return cls(
            id=str(row["id"]),
            type_id=str(row.get("type_id") or stable_id(
                "legacy_observation_type",
                {
                    "source_tool": row.get("source_tool"),
                    "source_revision": row.get("source_revision"),
                    "location": row.get("location"),
                    "export_name": row.get("export_name"),
                },
            )),
            field_id=str(row.get("field_id") or stable_id(
                "legacy_output_field",
                {
                    "source_tool": row.get("source_tool"),
                    "source_revision": row.get("source_revision"),
                    "location": row.get("location"),
                    "export_name": row.get("export_name"),
                    "path": row.get("path"),
                },
            )),
            value=row.get("value"),
            source_tool=str(row["source_tool"]),
            source_call=str(row["source_call"]),
            source_revision=int(row["source_revision"]),
            location=str(row["location"]),
            export_name=str(row["export_name"]),
            path=str(row["path"]),
            # Rows without a declared selector are matched by name only
            # rather than by reinterpreting a concrete path.
            field_path=(
                str(row["field_path"])
                if row.get("field_path") is not None else None
            ),
        )


@dataclass
class SearchState:
    id: str
    handle: str
    digest: str
    depth: int
    producer_call: str | None = None
    producer_before_handle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "handle": self.handle,
            "digest": self.digest,
            "depth": self.depth,
            "producer_call": self.producer_call,
            "producer_before_handle": self.producer_before_handle,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SearchState":
        return cls(
            id=str(row["id"]),
            handle=str(row["handle"]),
            digest=str(row["digest"]),
            depth=int(row["depth"]),
            producer_call=(
                str(row["producer_call"]) if row.get("producer_call") else None
            ),
            producer_before_handle=(
                str(row["producer_before_handle"])
                if row.get("producer_before_handle")
                else None
            ),
        )


@dataclass
class SearchTask:
    id: str
    state_id: str
    tool: str
    revision: int
    template_index: int
    observation_call: str | None = None
    provider_search: bool = False
    anchor_search: bool = False
    next_combination: int = 0
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SearchTask":
        return cls(
            id=str(row["id"]),
            state_id=str(row["state_id"]),
            tool=str(row["tool"]),
            revision=int(row["revision"]),
            template_index=int(row["template_index"]),
            observation_call=(
                str(row["observation_call"])
                if row.get("observation_call") else None
            ),
            provider_search=bool(row.get("provider_search", False)),
            anchor_search=bool(row.get("anchor_search", False)),
            next_combination=int(row.get("next_combination", 0)),
            completed=bool(row.get("completed", False)),
        )


@dataclass(frozen=True)
class SearchBudget:
    max_calls: int = 20_000
    max_states: int = 1_000
    max_depth: int = 12
    max_seconds: float | None = None
    max_external_model_calls: int | None = None
    max_contract_repair_calls: int | None = None
    max_retries: int | None = None
    max_storage_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_states", "max_depth"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "max_seconds",
            "max_external_model_calls",
            "max_contract_repair_calls",
            "max_retries",
            "max_storage_bytes",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class BuildConfig:
    budget: SearchBudget = field(default_factory=SearchBudget)
    trial_repetitions: int = 2
    max_observation_bytes: int = 65_536
    serial_state_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.trial_repetitions < 2:
            raise ValueError("trial_repetitions must be at least two")
        if self.max_observation_bytes < 1:
            raise ValueError("max_observation_bytes must be positive")
        if self.serial_state_batch_size < 1:
            raise ValueError("serial_state_batch_size must be positive")


class SQLiteBuildStore:
    def __init__(
        self,
        path: str | Path,
        *,
        persist_tasks: bool = True,
        persist_calls: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for stale in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            stale.unlink(missing_ok=True)
        self.db = sqlite3.connect(self.path)
        self._batch_depth = 0
        self.persist_tasks = bool(persist_tasks)
        self.persist_calls = bool(persist_calls)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS states (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                type_id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                binding_key TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS input_trials (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS state_trials (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)
        self._commit()
        # Calls are live coordinator state. SQLite records them for final
        # evidence export, but candidate fill and stable reduction must not
        # round-trip through SQLite to reuse an already executed binding.
        call_rows = (
            list(self.db.execute("SELECT id,binding_key,payload FROM calls"))
            if self.persist_calls else []
        )
        self._calls_by_binding: dict[str, dict[str, Any]] = {}
        self._calls_by_id: dict[str, dict[str, Any]] = {}
        self._call_binding_by_id: dict[str, str] = {}
        for call_id, binding_key, serialized in call_rows:
            payload = dict(json.loads(str(serialized)))
            self._calls_by_binding[str(binding_key)] = payload
            self._calls_by_id[str(call_id)] = payload
            self._call_binding_by_id[str(call_id)] = str(binding_key)
        self._binding_keys = set(self._calls_by_binding)
        self._call_ids = set(self._calls_by_id)
        self._batched_call_rows: list[tuple[str, str, str]] = []
        self._batched_calls_by_binding: dict[str, dict[str, Any]] = {}
        self._batched_calls_by_id: dict[str, dict[str, Any]] = {}
        self._tasks_by_id: dict[str, SearchTask] = {
            str(task_id): SearchTask.from_dict(json.loads(str(payload)))
            for task_id, payload in self.db.execute("SELECT id,payload FROM tasks")
        } if not self.persist_tasks else {}
        self._input_trials_by_value: dict[str, dict[str, Any]] = {}
        self._batched_input_trial_values_added: set[str] = set()
        for (payload,) in self.db.execute("SELECT payload FROM input_trials"):
            row = dict(json.loads(str(payload)))
            if row.get("value_trial_id") is not None:
                self._input_trials_by_value.setdefault(
                    str(row["value_trial_id"]), row
                )
    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def storage_bytes(self) -> int:
        """Return the on-disk SQLite footprint using only three file stats."""

        total = 0
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total

    def _commit(self) -> None:
        if self._batch_depth == 0:
            self.db.commit()

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Coalesce nested coordinator mutations into one SQLite commit."""

        outermost = self._batch_depth == 0
        if outermost:
            self.db.execute("BEGIN IMMEDIATE")
            self._batched_call_rows = []
            self._batched_calls_by_binding = {}
            self._batched_calls_by_id = {}
            self._batched_input_trial_values_added = set()
        self._batch_depth += 1
        try:
            yield
        except BaseException:
            self._batch_depth -= 1
            if outermost:
                self.db.rollback()
                self._binding_keys.difference_update(
                    self._batched_calls_by_binding
                )
                self._call_ids.difference_update(self._batched_calls_by_id)
                for binding_key, payload in self._batched_calls_by_binding.items():
                    call_id = str(payload["id"])
                    self._calls_by_binding.pop(binding_key, None)
                    self._calls_by_id.pop(call_id, None)
                    self._call_binding_by_id.pop(call_id, None)
                for value_trial_id in self._batched_input_trial_values_added:
                    self._input_trials_by_value.pop(value_trial_id, None)
                self._batched_call_rows = []
                self._batched_calls_by_binding = {}
                self._batched_calls_by_id = {}
                self._batched_input_trial_values_added = set()
            raise
        else:
            self._batch_depth -= 1
            if outermost:
                try:
                    if self.persist_calls and self._batched_call_rows:
                        self.db.executemany(
                            "INSERT INTO calls(id,binding_key,payload) VALUES(?,?,?)",
                            self._batched_call_rows,
                        )
                    self.db.commit()
                except BaseException:
                    self.db.rollback()
                    self._binding_keys.difference_update(
                        self._batched_calls_by_binding
                    )
                    self._call_ids.difference_update(self._batched_calls_by_id)
                    for value_trial_id in self._batched_input_trial_values_added:
                        self._input_trials_by_value.pop(value_trial_id, None)
                    raise
                finally:
                    self._batched_call_rows = []
                    self._batched_calls_by_binding = {}
                    self._batched_calls_by_id = {}
                    self._batched_input_trial_values_added = set()

    def get_meta(self, key: str) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            (key, canonical_json(value)),
        )
        self._commit()

    def set_meta_many(self, values: Mapping[str, Any]) -> None:
        with self.batch():
            self.db.executemany(
                "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                [
                    (str(key), canonical_json(value))
                    for key, value in values.items()
                ],
            )

    def _put(self, table: str, row_id: str, payload: Mapping[str, Any]) -> bool:
        cursor = self.db.execute(
            f"INSERT OR IGNORE INTO {table}(id,payload) VALUES(?,?)",
            (row_id, canonical_json(payload)),
        )
        self._commit()
        return cursor.rowcount > 0

    def _replace(self, table: str, row_id: str, payload: Mapping[str, Any]) -> None:
        self.db.execute(
            f"INSERT OR REPLACE INTO {table}(id,payload) VALUES(?,?)",
            (row_id, canonical_json(payload)),
        )
        self._commit()

    def _rows(self, table: str) -> list[dict[str, Any]]:
        return list(self._iter_rows(table))

    def _iter_rows(
        self,
        table: str,
        *,
        where: str = "",
        parameters: Sequence[Any] = (),
    ) -> Iterator[dict[str, Any]]:
        clause = f" WHERE {where}" if where else ""
        for (payload,) in self.db.execute(
            f"SELECT payload FROM {table}{clause} ORDER BY rowid",
            tuple(parameters),
        ):
            yield json.loads(payload)

    def count(self, table: str) -> int:
        if table == "calls" and not self.persist_calls:
            return len(self._calls_by_id)
        return int(self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def has_row(self, table: str, row_id: str) -> bool:
        return self.db.execute(
            f"SELECT 1 FROM {table} WHERE id=?", (row_id,)
        ).fetchone() is not None

    def put_state(self, state: SearchState) -> bool:
        return self._put("states", state.id, state.to_dict())

    def update_state(self, state: SearchState) -> None:
        self._replace("states", state.id, state.to_dict())

    def states(self) -> list[SearchState]:
        return [SearchState.from_dict(row) for row in self._rows("states")]

    def put_observations(
        self, observations: Iterable[Observation]
    ) -> set[str]:
        """Persist the first value admitted for each observation field type."""

        inserted: set[str] = set()
        with self.batch():
            for observation in observations:
                cursor = self.db.execute(
                    """
                    INSERT OR IGNORE INTO observations(id,type_id,payload)
                    VALUES(?,?,?)
                    """,
                    (
                        observation.id,
                        observation.type_id,
                        canonical_json(observation.to_dict()),
                    ),
                )
                if cursor.rowcount > 0:
                    inserted.add(observation.type_id)
        return inserted

    def observations(self) -> list[Observation]:
        return [
            Observation.from_dict(row)
            for row in self._rows("observations")
        ]

    def put_task(self, task: SearchTask) -> bool:
        if not self.persist_tasks:
            if task.id in self._tasks_by_id:
                return False
            self._tasks_by_id[task.id] = task
            return True
        return self._put("tasks", task.id, task.to_dict())

    def put_tasks(self, tasks: Iterable[SearchTask]) -> set[str]:
        if not self.persist_tasks:
            inserted: set[str] = set()
            for task in tasks:
                if task.id in self._tasks_by_id:
                    continue
                self._tasks_by_id[task.id] = task
                inserted.add(task.id)
            return inserted
        inserted: set[str] = set()
        with self.batch():
            for task in tasks:
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO tasks(id,payload) VALUES(?,?)",
                    (task.id, canonical_json(task.to_dict())),
                )
                if cursor.rowcount > 0:
                    inserted.add(task.id)
        return inserted

    def update_task(self, task: SearchTask) -> None:
        if not self.persist_tasks:
            self._tasks_by_id[task.id] = task
            return
        self._replace("tasks", task.id, task.to_dict())

    def tasks(self) -> list[SearchTask]:
        if not self.persist_tasks:
            return list(self._tasks_by_id.values())
        return [SearchTask.from_dict(row) for row in self._rows("tasks")]

    def task(self, task_id: str) -> SearchTask | None:
        if not self.persist_tasks:
            return self._tasks_by_id.get(task_id)
        row = self.db.execute(
            "SELECT payload FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        return SearchTask.from_dict(json.loads(row[0])) if row else None

    def pending_task_state_ids(self) -> list[str]:
        if not self.persist_tasks:
            return list({
                task.state_id
                for task in self._tasks_by_id.values()
                if not task.completed
            })
        return [
            str(row[0])
            for row in self.db.execute("""
                SELECT DISTINCT json_extract(payload, '$.state_id')
                FROM tasks
                WHERE json_extract(payload, '$.completed') = 0
            """)
        ]

    def pending_tasks_for_state(self, state_id: str) -> Iterator[SearchTask]:
        if not self.persist_tasks:
            yield from (
                task
                for task in self._tasks_by_id.values()
                if not task.completed and task.state_id == state_id
            )
            return
        for row in self._iter_rows(
            "tasks",
            where=(
                "json_extract(payload, '$.completed') = 0 AND "
                "json_extract(payload, '$.state_id') = ?"
            ),
            parameters=(state_id,),
        ):
            yield SearchTask.from_dict(row)

    def task_revisions(self) -> set[tuple[str, int]]:
        if not self.persist_tasks:
            return {
                (task.tool, task.revision)
                for task in self._tasks_by_id.values()
            }
        return {
            (str(tool), int(revision))
            for tool, revision in self.db.execute("""
                SELECT DISTINCT
                    json_extract(payload, '$.tool'),
                    json_extract(payload, '$.revision')
                FROM tasks
            """)
        }

    def put_call(self, call: Mapping[str, Any], binding_key: str) -> bool:
        call_id = str(call["id"])
        if self._batch_depth > 0:
            if call_id in self._call_ids:
                return False
            stored_binding_key = (
                binding_key
                if binding_key not in self._binding_keys
                else f"{binding_key}:{call_id}"
            )
            payload = dict(call)
            if self.persist_calls:
                self._batched_call_rows.append((
                    call_id,
                    stored_binding_key,
                    canonical_json(payload),
                ))
            self._binding_keys.add(stored_binding_key)
            self._call_ids.add(call_id)
            self._calls_by_binding[stored_binding_key] = payload
            self._calls_by_id[call_id] = payload
            self._call_binding_by_id[call_id] = stored_binding_key
            self._batched_calls_by_binding[stored_binding_key] = payload
            self._batched_calls_by_id[call_id] = payload
            return True
        if not self.persist_calls:
            if call_id in self._call_ids:
                return False
            stored_binding_key = (
                binding_key
                if binding_key not in self._binding_keys
                else f"{binding_key}:{call_id}"
            )
            payload = dict(call)
            self._binding_keys.add(stored_binding_key)
            self._call_ids.add(call_id)
            self._calls_by_binding[stored_binding_key] = payload
            self._calls_by_id[call_id] = payload
            self._call_binding_by_id[call_id] = stored_binding_key
            return True
        stored_binding_key = binding_key
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO calls(id,binding_key,payload) VALUES(?,?,?)",
            (call_id, binding_key, canonical_json(call)),
        )
        if cursor.rowcount == 0 and self.db.execute(
            "SELECT 1 FROM calls WHERE id=?", (call_id,)
        ).fetchone() is None:
            stored_binding_key = f"{binding_key}:{call_id}"
            cursor = self.db.execute(
                "INSERT INTO calls(id,binding_key,payload) VALUES(?,?,?)",
                (
                    call_id,
                    stored_binding_key,
                    canonical_json(call),
                ),
            )
        if cursor.rowcount > 0:
            self._binding_keys.add(stored_binding_key)
            self._call_ids.add(call_id)
            payload = dict(call)
            self._calls_by_binding[stored_binding_key] = payload
            self._calls_by_id[call_id] = payload
            self._call_binding_by_id[call_id] = stored_binding_key
        self._commit()
        return cursor.rowcount > 0

    def update_call(self, call: Mapping[str, Any]) -> None:
        """Replace one already-recorded call while preserving its binding."""

        call_id = str(call["id"])
        payload = dict(call)
        if call_id in self._batched_calls_by_id:
            self._batched_calls_by_id[call_id] = payload
            for binding_key, pending in tuple(
                self._batched_calls_by_binding.items()
            ):
                if str(pending.get("id")) == call_id:
                    self._batched_calls_by_binding[binding_key] = payload
                    self._calls_by_binding[binding_key] = payload
            self._calls_by_id[call_id] = payload
            self._batched_call_rows = [
                (
                    row_id,
                    binding_key,
                    canonical_json(payload) if row_id == call_id else serialized,
                )
                for row_id, binding_key, serialized in self._batched_call_rows
            ]
            return
        if not self.persist_calls:
            if call_id not in self._calls_by_id:
                raise KeyError(f"unknown call: {call_id}")
            self._calls_by_id[call_id] = payload
            binding_key = self._call_binding_by_id[call_id]
            self._calls_by_binding[binding_key] = payload
            return
        cursor = self.db.execute(
            "UPDATE calls SET payload=? WHERE id=?",
            (canonical_json(payload), call_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown call: {call_id}")
        self._calls_by_id[call_id] = payload
        binding_key = self._call_binding_by_id[call_id]
        self._calls_by_binding[binding_key] = payload
        self._commit()

    def calls(self) -> list[dict[str, Any]]:
        if not self.persist_calls:
            return [dict(call) for call in self._calls_by_id.values()]
        return self._rows("calls")

    def iter_calls(self) -> Iterator[dict[str, Any]]:
        if not self.persist_calls:
            yield from (dict(call) for call in self._calls_by_id.values())
            return
        yield from self._iter_rows("calls")

    def call(self, call_id: str) -> dict[str, Any] | None:
        call = self._calls_by_id.get(call_id)
        return dict(call) if call is not None else None

    def call_by_binding(self, binding_key: str) -> dict[str, Any] | None:
        call = self._calls_by_binding.get(binding_key)
        return dict(call) if call is not None else None

    def seen_binding(self, binding_key: str) -> bool:
        return binding_key in self._binding_keys

    def put_input_trial(self, row: Mapping[str, Any]) -> bool:
        inserted = self._put("input_trials", str(row["id"]), row)
        if inserted and row.get("value_trial_id") is not None:
            value_trial_id = str(row["value_trial_id"])
            if value_trial_id not in self._input_trials_by_value:
                self._input_trials_by_value[value_trial_id] = dict(row)
                if self._batch_depth > 0:
                    self._batched_input_trial_values_added.add(value_trial_id)
        return inserted

    def input_trials(self) -> list[dict[str, Any]]:
        return self._rows("input_trials")

    def input_trial_for_value(self, value_trial_id: str) -> dict[str, Any] | None:
        row = self._input_trials_by_value.get(value_trial_id)
        return dict(row) if row is not None else None

    def put_state_trial(self, row: Mapping[str, Any]) -> bool:
        return self._put("state_trials", str(row["id"]), row)

    def state_trials(self) -> list[dict[str, Any]]:
        return self._rows("state_trials")

_OMIT = object()


@lru_cache(maxsize=262_144)
def _rank_sum_cumulative_count(
    sizes: tuple[int, ...], total: int
) -> int:
    """Count bounded coordinates whose rank sum is at most ``total``."""

    if total < 0:
        return 0
    if not sizes:
        return 1
    maximum = sum(size - 1 for size in sizes)
    if total >= maximum:
        return math.prod(sizes)
    if all(size > total for size in sizes):
        return math.comb(total + len(sizes), len(sizes))
    first, rest = sizes[0], sizes[1:]
    return sum(
        _rank_sum_cumulative_count(rest, total - value)
        for value in range(min(first - 1, total) + 1)
    )


@lru_cache(maxsize=262_144)
def _rank_sum_nonpreferred_cumulative_count(
    sizes: tuple[int, ...],
    preferred: tuple[frozenset[int], ...],
    total: int,
) -> int:
    """Count rank-bounded coordinates containing no preferred choice."""

    if total < 0:
        return 0
    if not sizes:
        return 1
    if not any(preferred):
        return _rank_sum_cumulative_count(sizes, total)
    maximum = sum(size - 1 for size in sizes)
    if total >= maximum:
        return math.prod(
            size - len(choices)
            for size, choices in zip(sizes, preferred, strict=True)
        )
    first, rest = sizes[0], sizes[1:]
    blocked, remaining = preferred[0], preferred[1:]
    return sum(
        _rank_sum_nonpreferred_cumulative_count(
            rest, remaining, total - value
        )
        for value in range(min(first - 1, total) + 1)
        if value not in blocked
    )


def _rank_sum_exact_count(sizes: tuple[int, ...], total: int) -> int:
    return (
        _rank_sum_cumulative_count(sizes, total)
        - _rank_sum_cumulative_count(sizes, total - 1)
    )


def _rank_sum_filtered_exact_count(
    sizes: tuple[int, ...],
    preferred: tuple[frozenset[int], ...],
    total: int,
) -> int:
    all_rows = _rank_sum_exact_count(sizes, total)
    nonpreferred = (
        _rank_sum_nonpreferred_cumulative_count(sizes, preferred, total)
        - _rank_sum_nonpreferred_cumulative_count(
            sizes, preferred, total - 1
        )
    )
    return all_rows - nonpreferred


def _rank_sum_shell(
    sizes: tuple[int, ...],
    index: int,
    preferred: tuple[frozenset[int], ...] | None = None,
) -> tuple[int, int]:
    """Return ``(sum, offset)`` for an index in graded lexicographic order."""

    maximum = sum(size - 1 for size in sizes)

    def cumulative(total: int) -> int:
        count = _rank_sum_cumulative_count(sizes, total)
        if preferred is not None:
            count -= _rank_sum_nonpreferred_cumulative_count(
                sizes, preferred, total
            )
        return count

    if index < cumulative(0):
        return 0, index
    low = 1
    high = 1
    while high < maximum and index >= cumulative(high):
        low = high + 1
        high = min(maximum, high * 2 + 1)
    high = max(low, high)
    while low < high:
        middle = (low + high) // 2
        if index < cumulative(middle):
            high = middle
        else:
            low = middle + 1
    shell = low
    return shell, index - cumulative(shell - 1)


def rank_sum_product_index(
    sizes: Sequence[int], index: int
) -> tuple[int, ...] | None:
    """Unrank a product by increasing sum of per-field candidate ranks.

    Rank zero is the contract seed (literal or omission) when one exists;
    observation-only fields start with their best semantic match. This order is
    complete but lets several top-ranked replacements precede one distant
    single-field replacement.
    """

    dimensions = tuple(sizes)
    if index < 0 or any(size <= 0 for size in dimensions):
        return None
    if not dimensions:
        return () if index == 0 else None
    if index >= math.prod(dimensions):
        return None
    if index == 0:
        return (0,) * len(dimensions)
    remaining, offset = _rank_sum_shell(dimensions, index)
    coordinates: list[int] = []
    for position, size in enumerate(dimensions):
        suffix = dimensions[position + 1:]
        for value in range(min(size - 1, remaining) + 1):
            count = _rank_sum_exact_count(suffix, remaining - value)
            if offset < count:
                coordinates.append(value)
                remaining -= value
                break
            offset -= count
        else:  # pragma: no cover - guarded by the shell counts above
            raise AssertionError("rank-sum product unranking failed")
    return tuple(coordinates)


def promoted_product_index(
    sizes: Sequence[int],
    index: int,
    promoted: Sequence[int | None],
) -> tuple[int, ...] | None:
    """Move one existing product candidate to the front without pruning."""

    candidate = rank_sum_product_index(sizes, index)
    if candidate is None:
        return None
    choices = tuple(promoted)
    dimensions = tuple(sizes)
    if len(choices) != len(dimensions):
        raise ValueError("promoted indexes and sizes differ")
    if any(
        choice is not None and (choice < 0 or choice >= size)
        for choice, size in zip(choices, dimensions, strict=True)
    ):
        raise ValueError("promoted coordinate is outside its dimension")
    if not any(choice is not None for choice in choices):
        return candidate
    head = tuple(
        choice if choice is not None else 0 for choice in choices
    )
    baseline = (0,) * len(dimensions)
    if head == baseline:
        return candidate
    if index == 0:
        return head
    if candidate == head:
        return baseline
    return candidate


def rank_sum_filtered_product_index(
    sizes: Sequence[int],
    preferred: Sequence[frozenset[int]],
    index: int,
) -> tuple[int, ...] | None:
    """Unrank the source-triggered subsequence of rank-sum product order."""

    dimensions = tuple(sizes)
    preferred_rows = tuple(preferred)
    if len(dimensions) != len(preferred_rows):
        raise ValueError("sizes and preferred dimensions differ")
    if index < 0 or any(size <= 0 for size in dimensions) or not dimensions:
        return None
    if any(
        choice < 0 or choice >= size
        for size, choices in zip(dimensions, preferred_rows, strict=True)
        for choice in choices
    ):
        raise ValueError("preferred coordinate is outside its dimension")
    if not any(preferred_rows):
        return None
    total = math.prod(dimensions) - math.prod(
        size - len(choices)
        for size, choices in zip(dimensions, preferred_rows, strict=True)
    )
    if index >= total:
        return None
    remaining, offset = _rank_sum_shell(
        dimensions, index, preferred_rows
    )
    coordinates: list[int] = []
    hit_preferred = False
    for position, size in enumerate(dimensions):
        suffix_sizes = dimensions[position + 1:]
        suffix_preferred = preferred_rows[position + 1:]
        for value in range(min(size - 1, remaining) + 1):
            next_hit = hit_preferred or value in preferred_rows[position]
            if next_hit:
                count = _rank_sum_exact_count(
                    suffix_sizes, remaining - value
                )
            else:
                count = _rank_sum_filtered_exact_count(
                    suffix_sizes, suffix_preferred, remaining - value
                )
            if offset < count:
                coordinates.append(value)
                remaining -= value
                hit_preferred = next_hit
                break
            offset -= count
        else:  # pragma: no cover - guarded by the shell counts above
            raise AssertionError("filtered rank-sum product unranking failed")
    if not hit_preferred:  # pragma: no cover - guarded by filtered counts
        raise AssertionError("filtered rank-sum product omitted its trigger")
    return tuple(coordinates)


def source_coherent_product_index(
    sizes: Sequence[int],
    preferred: Sequence[frozenset[int]],
    index: int,
    *,
    preferred_scores: Sequence[float | None],
) -> tuple[int, ...] | None:
    """Try two complementary dependency bindings, then keep exhaustive order.

    The first head lets one source coherently satisfy every compatible field.
    The second uses that source only for its strongest target field; coordinate
    zero gives other observation fields their globally best match while keeping
    contract literals and omissions.  Both heads swap with their old positions,
    so no candidate is added or removed before the unchanged exhaustive suffix.
    """

    if index < 0:
        return None
    candidate = rank_sum_filtered_product_index(sizes, preferred, index)
    if candidate is None:
        return None
    if len(preferred_scores) != len(tuple(sizes)):
        raise ValueError("preferred_scores and sizes differ")
    available = [
        position for position, choices in enumerate(preferred) if choices
    ]
    if not available:
        return candidate
    selected = available[0]
    selected_score = preferred_scores[selected]
    for position in available[1:]:
        score = preferred_scores[position]
        if score is not None and (
            selected_score is None or score > selected_score
        ):
            selected = position
            selected_score = score
    source_coherent = tuple(
        min(choices) if choices else 0 for choices in preferred
    )
    fieldwise = tuple(
        min(choices) if position == selected else 0
        for position, choices in enumerate(preferred)
    )
    heads = tuple(dict.fromkeys((source_coherent, fieldwise)))
    displaced: list[tuple[int, ...]] = []
    for position, head in enumerate(heads):
        row = rank_sum_filtered_product_index(sizes, preferred, position)
        if row is None:  # pragma: no cover - every head is in the sequence
            raise AssertionError("dependency head is outside the product")
        for previous, previous_head in enumerate(heads[:position]):
            if row == previous_head:
                row = displaced[previous]
        displaced.append(row)
        if index == position:
            candidate = head
        elif candidate == head:
            candidate = row
    return candidate


@dataclass(frozen=True)
class ArgumentChoice:
    value: Any
    observations: tuple[Observation, ...] = ()
    omitted: bool = False


@dataclass(frozen=True)
class _ObservationChoiceGroup:
    """One equal-semantic-score block shared by every source-call view."""

    choices: tuple[ArgumentChoice, ...]
    value_offsets: Mapping[str, int]


@dataclass(frozen=True)
class _ObservationChoiceIndex:
    groups: tuple[_ObservationChoiceGroup, ...]
    value_locations: Mapping[str, tuple[int, int, ArgumentChoice]]
    group_starts: tuple[int, ...]


class _ArgumentChoiceDomain(Sequence[ArgumentChoice]):
    """A zero-copy source-preferred view over a shared observation domain.

    Event tasks give their triggering source call precedence only within an
    equal semantic-score block.  Materializing that permutation once per
    source made bulk orchestration quadratic in the number of observations.
    This view computes the handful of selected indexes lazily while all tasks
    share the choices, provenance, and semantic groups.
    """

    def __init__(
        self,
        prefix: tuple[ArgumentChoice, ...],
        observation_index: _ObservationChoiceIndex,
        preferred_source: str | None,
        preferred_entries: tuple[tuple[int, ArgumentChoice], ...] = (),
        preferred_offsets: Mapping[int, tuple[int, ...]] | None = None,
        preferred_score: float | None = None,
    ) -> None:
        self.prefix = prefix
        self.groups = observation_index.groups
        self._observation_index = observation_index
        self.preferred_source = preferred_source
        self._preferred_entries = preferred_entries
        self._preferred_offsets = preferred_offsets or {}
        self.preferred_score = preferred_score
        self._preferred_choice_by_index = {
            len(prefix) + index: choice
            for index, choice in self._preferred_entries
        }
        self._group_starts = tuple(
            len(prefix) + start for start in observation_index.group_starts
        )
        self._length = len(prefix) + sum(
            len(group.choices) for group in self.groups
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> ArgumentChoice | tuple[ArgumentChoice, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < len(self.prefix):
            return self.prefix[index]
        preferred_choice = self._preferred_choice_by_index.get(index)
        if preferred_choice is not None:
            return preferred_choice
        for group_index, (start, group) in enumerate(zip(
            self._group_starts, self.groups, strict=True
        )):
            end = start + len(group.choices)
            if index >= end:
                continue
            local = index - start
            preferred = self._preferred_offsets.get(group_index, ())
            if local < len(preferred):
                return group.choices[preferred[local]]
            remaining = local - len(preferred)
            for offset, choice in enumerate(group.choices):
                if offset in preferred:
                    continue
                if remaining == 0:
                    return choice
                remaining -= 1
            raise IndexError(index)  # pragma: no cover - guarded by length
        raise IndexError(index)  # pragma: no cover - guarded by length

    def preferred_indexes(self, source_call: str) -> frozenset[int]:
        """Return source indexes without enumerating the full domain."""

        if source_call != self.preferred_source:
            return frozenset(
                index
                for index, choice in enumerate(self)
                if any(
                    observation.source_call == source_call
                    for observation in choice.observations
                )
            )
        prefix = len(self.prefix)
        return frozenset(prefix + index for index, _ in self._preferred_entries)

    def observation_index_for_value(self, value: Any) -> int | None:
        """Locate an existing observation value in this source-preferred view."""

        location = self._observation_index.value_locations.get(
            canonical_json(json_safe(value))
        )
        if location is None:
            return None
        group_index, original_offset, _ = location
        preferred = self._preferred_offsets.get(group_index, ())
        if not preferred:
            local_offset = original_offset
        elif original_offset in preferred:
            local_offset = preferred.index(original_offset)
        else:
            local_offset = len(preferred) + sum(
                offset not in preferred for offset in range(original_offset)
            )
        return self._group_starts[group_index] + local_offset


def binding_observations(value: Any) -> tuple[Observation, ...]:
    """Decode evidence bindings in either list or singleton form."""

    if isinstance(value, Mapping):
        return (Observation.from_dict(value),)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(
            Observation.from_dict(row) for row in value if isinstance(row, Mapping)
        )
    return ()


def serialized_bindings(
    bindings: Mapping[str, Sequence[Observation]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(parameter): [observation.to_dict() for observation in observations]
        for parameter, observations in bindings.items()
        if observations
    }


class FieldPriorityError(RuntimeError):
    """A semantic field index could not provide a reproducible ordering."""


class FieldPriorityIndex(Protocol):
    """Optional semantic ordering layer for otherwise unchanged candidates."""

    def ensure_contract(
        self, definition: ToolDefinition, contract: ToolCallContract
    ) -> int: ...

    def ensure_manifest(
        self, contracts: Mapping[str, ToolCallContract]
    ) -> int: ...

    def score(
        self,
        observation: Observation,
        *,
        target_tool: str,
        target_parameter: ToolParameter,
    ) -> float | None: ...

    def build_metadata(self) -> Mapping[str, Any]: ...


class DependencyGraphBuilder:
    include_self_argument_edges = False

    def __init__(
        self,
        runtime: BranchingRuntime,
        definitions: Sequence[ToolDefinition],
        contracts: OnlineContractSet,
        *,
        store: SQLiteBuildStore,
        config: BuildConfig | None = None,
        field_priority: FieldPriorityIndex | None = None,
        scheduling_priority: FieldPriorityIndex | None = None,
        storage_usage: StorageUsage | None = None,
    ) -> None:
        self.runtime = runtime
        # Serial discovery often evaluates many candidates against the same
        # immutable state.  Keep that state loaded after an unchanged call;
        # a mutation or exception invalidates the cache so the next call
        # reloads the immutable handle.  This is the same safety rule used by
        # the persistent bulk workers.
        self._loaded_runtime_state_handle: str | None = None
        self._loaded_runtime_state_pristine = False
        self.definitions = {definition.tool: definition for definition in definitions}
        self.contracts = contracts
        self.store = store
        self.config = config or BuildConfig()
        self._storage_usage = storage_usage
        self._last_storage_bytes = 0
        self._last_storage_check = system_monotonic()
        select_template_policy(
            self.contracts.manifest,
            MINIMAL_SUCCESS_TEMPLATE_POLICY,
        )
        self.minimal_success_templates_added = 0
        for tool, contract in sorted(self.contracts.manifest.contracts.items()):
            definition = self.definitions.get(tool)
            if definition is None:
                continue
            normalized = with_minimal_success_baseline(
                contract,
                parameter_names=[row.name for row in definition.parameters],
                required_parameters={
                    row.name for row in definition.parameters if row.required
                },
            )
            if normalized != contract:
                self.contracts.manifest.contracts[tool] = normalized
                self.minimal_success_templates_added += 1
        # Argument binding and global task ordering may use different public
        # field descriptors. By default they share one index.
        self.field_priority = field_priority
        self.scheduling_priority = (
            scheduling_priority
            if scheduling_priority is not None
            else field_priority
        )
        self.embedding_fields_added = 0
        self.scheduling_embedding_fields_added = 0
        self.blocked_contract_revisions: set[tuple[str, int]] = set()
        self.scheduler_policy = (
            "template_index_base_first_uniform_task_kind_dovetail_semantic_v4"
            if self.scheduling_priority is not None
            else "template_index_base_first_uniform_task_kind_dovetail_v4"
        )
        self.started = system_monotonic()
        self.calls_used = 0
        self.stop_reason = "completed"
        self.budget_violations: set[str] = set()
        self._active_budget_violations: set[str] = set()
        self.initial_llm_calls = contracts.total_llm_calls()
        self.contract_repair_requests = 0
        self.contract_repairs_accepted = 0
        self.contract_repairs_rejected = 0
        self.contract_repairs_budget_rejected = 0
        store.set_meta("initial_llm_calls", self.initial_llm_calls)
        store.set_meta("runtime_id", runtime.runtime_id)
        store.set_meta("graph_version", GRAPH_VERSION)
        store.set_meta(
            "serial_state_batch_size", self.config.serial_state_batch_size
        )
        requested_order = (
            "semantic" if self.scheduling_priority is not None else "legacy"
        )
        store.set_meta("propagation_order", requested_order)
        if self.field_priority is not None:
            try:
                added_now = self.field_priority.ensure_manifest(
                    self.contracts.manifest.contracts
                )
                self.embedding_fields_added += added_now
            except FieldPriorityError as exc:
                self.stop_reason = "field_embedding_error"
                store.set_meta("field_embedding_error", str(exc))
                raise
            store.set_meta("field_embedding_error", None)
        if (
            self.scheduling_priority is not None
            and self.scheduling_priority is not self.field_priority
        ):
            try:
                self.scheduling_embedding_fields_added += (
                    self.scheduling_priority.ensure_manifest(
                        self.contracts.manifest.contracts
                    )
                )
            except FieldPriorityError as exc:
                self.stop_reason = "field_embedding_error"
                store.set_meta("field_embedding_error", str(exc))
                raise
        self.states = {state.id: state for state in store.states()}
        self._live_state_handles = {
            state.handle for state in self.states.values()
        }
        self._state_producer_tools: dict[str, str] = {}
        self.observations_by_type = {
            observation.type_id: observation
            for observation in store.observations()
        }
        self.observations_by_source_call: dict[str, tuple[Observation, ...]] = {}
        self.observations_by_source_value: dict[
            str, dict[str, tuple[Observation, ...]]
        ] = {}
        self.observations_by_value: dict[str, tuple[Observation, ...]] = {}
        self._observation_task_blueprints: dict[
            str, tuple[tuple[str, int, int, bool], ...]
        ] = {}
        self._semantic_task_score_cache: dict[
            tuple[str, str, int, int, bool], float | None
        ] = {}
        self._observation_call_input_states: dict[str, str] = {}
        self._observation_call_output_states: dict[str, str] = {}
        self._observation_call_carried_states: dict[str, str] = {}
        self._call_bound_values: dict[str, dict[str, Any]] = {}
        self._argument_choices_cache: dict[
            tuple[Any, ...], Sequence[ArgumentChoice]
        ] = {}
        self._observation_choices_cache: dict[
            tuple[str, str, str], _ObservationChoiceIndex
        ] = {}
        self._parameter_observation_scores_cache: dict[
            tuple[str, str, str],
            tuple[tuple[int, float | None, Observation], ...],
        ] = {}
        # Field priority is descriptor-only: the score for one declared output
        # field and one public input field does not depend on runtime values or
        # world state.  Observation-domain caches below may need rebuilding as
        # new values arrive, but these pair scores remain valid.
        self._field_priority_score_cache: dict[
            tuple[str, str, str], float | None
        ] = {}
        self._scheduling_priority_score_cache: dict[
            tuple[str, str, str], float | None
        ] = {}
        self._parameter_schema_json_cache: dict[tuple[str, str], str] = {}
        self._specification_key_cache: dict[
            int, tuple[Any, tuple[str, str]]
        ] = {}
        self._schema_compatible_observations_cache: dict[
            str, tuple[tuple[int, Observation], ...]
        ] = {}
        self._observation_value_json: dict[str, str] = {}
        self._observation_source_index_size = -1
        self._rebuild_observation_source_index()
        self.admitted_observation_types = set(self.observations_by_type)
        self.observation_types_dropped = int(
            store.get_meta("observation_types_dropped") or 0
        )
        self.duplicate_template_calls_reused = int(
            store.get_meta("duplicate_template_calls_reused") or 0
        )
        self.pending_base_by_state: dict[str, deque[str]] = defaultdict(deque)
        self.pending_event_by_state: dict[str, deque[str]] = defaultdict(deque)
        self.pending_continuation_by_state: dict[str, deque[str]] = defaultdict(deque)
        self.pending_provider_by_state: dict[str, deque[str]] = defaultdict(deque)
        self.active_states: deque[str] = deque()
        self.active_state_ids: set[str] = set()
        self.scheduled_task_ids: set[str] = set()
        self.schedule_phase: dict[str, int] = defaultdict(int)
        self.literal_baselines: set[tuple[str, str, int, int]] = set()
        self._first_success_call_by_tool: dict[str, str] = {}
        self._first_success_binding_by_tool: dict[str, str] = {}
        self._anchor_success_call_by_tool: dict[str, str] = {}
        self._provider_anchor_source_call_by_tool: dict[str, str] = {}
        self.observation_tasks_pruned_after_success = 0
        for source_call in self._observation_source_calls():
            observations = self._observations_for_source_call(source_call)
            if (
                observations
                and any(row.location != "arguments" for row in observations)
            ):
                self._provider_anchor_source_call_by_tool.setdefault(
                    observations[0].source_tool,
                    source_call,
                )
        source_calls = set(self.observations_by_source_call)
        for state in self.states.values():
            if state.producer_call in source_calls:
                self._observation_call_output_states[
                    str(state.producer_call)
                ] = state.id
        for call in store.iter_calls():
            self._record_literal_baseline(call)
            self._remember_call_bound_values(call)
            call_id = str(call.get("id") or "")
            saved_handle = str(call.get("saved_state_handle") or "")
            child_id = (
                stable_id("state", {"handle": saved_handle})
                if saved_handle else None
            )
            if call_id in source_calls:
                input_state_id = str(call.get("state_id") or "")
                if input_state_id in self.states:
                    self._observation_call_input_states[call_id] = input_state_id
                if child_id in self.states:
                    self._observation_call_output_states[call_id] = child_id
            if child_id in self.states:
                child = self.states[child_id]
                producer_tool = str(call.get("tool") or "")
                if producer_tool:
                    self._state_producer_tools[child_id] = producer_tool
                self._carry_bound_observations_to_state(call, child_id)
        for state_id in sorted(
            store.pending_task_state_ids(),
            key=lambda value: stable_id("state_schedule", value),
        ):
            self._schedule_tasks(store.pending_tasks_for_state(state_id))
        if not self.states:
            snapshot = runtime.snapshot()
            handle = runtime.save_state("dependency_root")
            root = SearchState(
                id=stable_id("state", {"handle": handle}),
                handle=handle,
                digest=snapshot.digest,
                depth=0,
            )
            store.put_state(root)
            self.states[root.id] = root
            self._live_state_handles.add(root.handle)
            self._enqueue_state(root)
        else:
            task_revisions = store.task_revisions()
            for tool, contract in self.contracts.manifest.contracts.items():
                if (tool, contract.revision) not in task_revisions:
                    self._enqueue_tool_revision(tool, contract.revision)

    def _elapsed(self) -> float:
        return system_monotonic() - self.started

    def _record_discovered_edge(
        self,
        kind: str,
        row: Mapping[str, Any],
    ) -> None:
        """Optional live log hook; the graph store remains authoritative."""

    def _record_first_tool_success(self, call: Mapping[str, Any]) -> None:
        """Optional live log hook for the first success of each tool."""

    def _observation_task_is_pruned(self, task: SearchTask) -> bool:
        if task.observation_call is None:
            return False
        winning_binding = self._first_success_binding_by_tool.get(task.tool)
        if winning_binding is None:
            return task.anchor_search
        if task.anchor_search:
            return False
        return self._incomplete_binding(task.id) != winning_binding

    def _provider_anchor_task(
        self,
        observation_call: str,
        target_tool: str,
    ) -> SearchTask | None:
        observations = self._observations_for_source_call(observation_call)
        if not observations or observations[0].source_tool == target_tool:
            return None
        winning_call_id = self._anchor_success_call_by_tool.get(target_tool)
        winning_call = (
            self.store.call(winning_call_id) if winning_call_id is not None else None
        )
        contract = self.contracts.get(target_tool)
        definition = self.definitions.get(target_tool)
        if winning_call is None or contract is None or definition is None:
            return None
        template_index = self._matching_selected_template_index(
            contract, winning_call
        )
        if template_index is None:
            return None
        state_id = (
            self._preferred_observation_state_id(observation_call)
            or str(winning_call.get("state_id") or "")
        )
        state = self.states.get(state_id)
        if state is None:
            return None
        if not any(
            schema_accepts(parameter.schema, observation.value)
            for parameter in definition.parameters
            for observation in observations
        ):
            return None
        return SearchTask(
            id=stable_id("provider_anchor_task", [
                observation_call,
                target_tool,
                winning_call_id,
                contract.revision,
                template_index,
                state_id,
            ]),
            state_id=state_id,
            tool=target_tool,
            revision=contract.revision,
            template_index=template_index,
            observation_call=observation_call,
            anchor_search=True,
        )

    def _enqueue_provider_anchor_tasks(
        self,
        *,
        observation_call: str | None = None,
        target_tool: str | None = None,
    ) -> None:
        if observation_call is not None:
            observations = self._observations_for_source_call(observation_call)
            if not observations or not any(
                row.location != "arguments" for row in observations
            ):
                return
            source_tool = observations[0].source_tool
            selected_call = self._provider_anchor_source_call_by_tool.setdefault(
                source_tool,
                observation_call,
            )
            if selected_call != observation_call:
                return
            source_calls = (observation_call,)
        else:
            source_calls = tuple(
                self._provider_anchor_source_call_by_tool[tool]
                for tool in sorted(self._provider_anchor_source_call_by_tool)
            )
        target_tools = (
            (target_tool,)
            if target_tool is not None
            else tuple(self._first_success_call_by_tool)
        )
        created = [
            task
            for source_call in sorted(source_calls)
            for tool in sorted(target_tools)
            if (task := self._provider_anchor_task(source_call, tool)) is not None
        ]
        inserted = self.store.put_tasks(created)
        self._schedule_tasks(task for task in created if task.id in inserted)

    def _state_producer_tool(self, state: SearchState) -> str | None:
        producer_tool = self._state_producer_tools.get(state.id)
        if producer_tool is not None or state.producer_call is None:
            return producer_tool
        call = self.store.call(state.producer_call)
        if call is None:
            return None
        producer_tool = str(call.get("tool") or "") or None
        if producer_tool is not None:
            self._state_producer_tools[state.id] = producer_tool
        return producer_tool

    def _storage_bytes(self) -> int:
        """Measure serial builder storage through lightweight ownership APIs.

        The coordinator SQLite footprint is three direct file stats.  A
        runtime can expose an incrementally maintained ``storage_usage_bytes``
        value for its immutable state/checkpoint root.  Callers with other
        builder-owned working artifacts can inject an additional
        ``storage_usage`` callback instead.  SQLite is always charged by the
        core.  The callback is deliberately a lightweight protocol: the
        invocation hot path must not recursively walk a whole run tree.
        """

        measured = self.store.storage_bytes()
        if self._storage_usage is not None:
            measured += int(self._storage_usage())
        else:
            runtime_usage = getattr(self.runtime, "storage_usage_bytes", None)
            if runtime_usage is not None:
                measured += int(runtime_usage())
        if measured < 0:
            raise ValueError("storage usage must be non-negative")
        return measured

    def _refresh_storage_bytes(self, *, force: bool = False) -> int:
        """Refresh serial storage usage; parallel builders override cadence."""

        del force
        self._last_storage_bytes = self._storage_bytes()
        self._last_storage_check = system_monotonic()
        return self._last_storage_bytes

    def _storage_budget_available(self, *, force: bool = False) -> bool:
        limit = self.config.budget.max_storage_bytes
        if limit is None:
            if force:
                self._refresh_storage_bytes(force=True)
            return True
        if self._refresh_storage_bytes(force=force) > limit:
            self._record_budget_violation("max_storage_bytes")
            return False
        return True

    def _record_budget_violation(self, reason: str) -> None:
        """Record every exceeded hard budget without replacing its root cause."""

        self.budget_violations.add(str(reason))
        self._active_budget_violations.add(str(reason))
        if self.stop_reason == "completed":
            self.stop_reason = str(reason)

    def _rebuild_observation_source_index(self) -> None:
        """Index admitted observations by the call that produced them.

        Event scheduling and semantic scoring repeatedly query this grouping.
        Re-scanning the global observation pool once per candidate makes queue
        expansion quadratic as the full graph grows.  The size guard also
        keeps tests and adapters that extend ``observations_by_type`` directly
        compatible with the indexed production path.
        """

        grouped: dict[str, list[Observation]] = defaultdict(list)
        serialized_values: dict[str, str] = {}
        for observation in self.observations_by_type.values():
            grouped[observation.source_call].append(observation)
            serialized_values[observation.type_id] = canonical_json(
                json_safe(observation.value)
            )
        self.observations_by_source_call = {
            source_call: tuple(observations)
            for source_call, observations in grouped.items()
        }
        self.observations_by_source_value = {}
        by_value_global: dict[str, tuple[Observation, ...]] = {}
        for source_call, observations in grouped.items():
            by_value: dict[str, tuple[Observation, ...]] = {}
            for observation in observations:
                value_key = serialized_values[observation.type_id]
                by_value[value_key] = (
                    *by_value.get(value_key, ()),
                    observation,
                )
                by_value_global[value_key] = (
                    *by_value_global.get(value_key, ()),
                    observation,
                )
            self.observations_by_source_value[source_call] = by_value
        self.observations_by_value = by_value_global
        self._observation_value_json = serialized_values
        self._observation_task_blueprints.clear()
        self._semantic_task_score_cache.clear()
        self._argument_choices_cache.clear()
        self._observation_choices_cache.clear()
        self._parameter_observation_scores_cache.clear()
        self._schema_compatible_observations_cache.clear()
        self._field_priority_score_cache.clear()
        self._scheduling_priority_score_cache.clear()
        self._observation_source_index_size = len(self.observations_by_type)

    def _index_new_observations(
        self, observations: Sequence[Observation]
    ) -> None:
        serialized_values = {
            observation.type_id: canonical_json(json_safe(observation.value))
            for observation in observations
        }
        grouped: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            grouped[observation.source_call].append(observation)
        for source_call, rows in grouped.items():
            self.observations_by_source_call[source_call] = (
                *self.observations_by_source_call.get(source_call, ()),
                *rows,
            )
            self._observation_task_blueprints.pop(source_call, None)
            for key in tuple(self._semantic_task_score_cache):
                if key[0] == source_call:
                    self._semantic_task_score_cache.pop(key, None)
            by_value = dict(self.observations_by_source_value.get(
                source_call, {}
            ))
            for row in rows:
                value_key = serialized_values[row.type_id]
                by_value[value_key] = (*by_value.get(value_key, ()), row)
                self.observations_by_value[value_key] = (
                    *self.observations_by_value.get(value_key, ()),
                    row,
                )
            self.observations_by_source_value[source_call] = by_value
        self._observation_value_json.update(serialized_values)
        self._argument_choices_cache.clear()
        self._observation_choices_cache.clear()
        self._parameter_observation_scores_cache.clear()
        self._schema_compatible_observations_cache.clear()
        self._observation_source_index_size = len(self.observations_by_type)

    def _field_priority_score(
        self,
        observation: Observation,
        *,
        target_tool: str,
        target_parameter: ToolParameter,
    ) -> float | None:
        """Return one immutable declared-field score without rescoring values."""

        if self.field_priority is None:
            return None
        key = (observation.field_id, target_tool, target_parameter.name)
        if key not in self._field_priority_score_cache:
            self._field_priority_score_cache[key] = self.field_priority.score(
                observation,
                target_tool=target_tool,
                target_parameter=target_parameter,
            )
        return self._field_priority_score_cache[key]

    def _scheduling_priority_score(
        self,
        observation: Observation,
        *,
        target_tool: str,
        target_parameter: ToolParameter,
    ) -> float | None:
        """Return one declared-field score used only for task ordering."""

        if self.scheduling_priority is None:
            return None
        key = (observation.field_id, target_tool, target_parameter.name)
        if key not in self._scheduling_priority_score_cache:
            self._scheduling_priority_score_cache[key] = (
                self.scheduling_priority.score(
                    observation,
                    target_tool=target_tool,
                    target_parameter=target_parameter,
                )
            )
        return self._scheduling_priority_score_cache[key]

    def _preferred_observation_view(
        self,
        observation_index: _ObservationChoiceIndex,
        *,
        source_call: str | None,
        target_tool: str,
        target_parameter: ToolParameter,
    ) -> tuple[
        tuple[tuple[int, ArgumentChoice], ...],
        Mapping[int, tuple[int, ...]],
        float | None,
    ]:
        """Build one triggering source's short view only when it is selected."""

        if source_call is None:
            return (), {}, None
        source_observations = self._observations_for_source_call(source_call)
        if not source_observations or source_observations[0].source_tool == target_tool:
            return (), {}, None
        by_group: dict[int, list[tuple[int, int, ArgumentChoice]]] = defaultdict(list)
        best_source_score: float | None = None
        for order, (value_key, source_rows) in enumerate(
            self.observations_by_source_value.get(source_call, {}).items()
        ):
            location = observation_index.value_locations.get(value_key)
            if location is None:
                continue
            group_index, offset, global_choice = location
            preferred_score: float | None = None
            preferred: dict[str, Observation] = {}
            for observation in source_rows:
                score = self._field_priority_score(
                    observation,
                    target_tool=target_tool,
                    target_parameter=target_parameter,
                )
                if not preferred or (
                    score is not None
                    and (preferred_score is None or score > preferred_score)
                ):
                    preferred_score = score
                    preferred = {observation.type_id: observation}
                elif score == preferred_score:
                    preferred.setdefault(observation.type_id, observation)
            if preferred_score is not None and (
                best_source_score is None
                or preferred_score > best_source_score
            ):
                best_source_score = preferred_score
            choice = ArgumentChoice(
                value=global_choice.value,
                observations=tuple({
                    observation.type_id: observation
                    for observation in (
                        *global_choice.observations,
                        *preferred.values(),
                    )
                }.values()),
            )
            by_group[group_index].append((order, offset, choice))

        entries: list[tuple[int, ArgumentChoice]] = []
        offsets: dict[int, tuple[int, ...]] = {}
        for group_index in sorted(by_group):
            rows = sorted(by_group[group_index])
            offsets[group_index] = tuple(row[1] for row in rows)
            entries.extend(
                (observation_index.group_starts[group_index] + rank, row[2])
                for rank, row in enumerate(rows)
            )
        return tuple(entries), offsets, best_source_score

    def _observations_for_source_call(
        self, source_call: str | None
    ) -> tuple[Observation, ...]:
        if self._observation_source_index_size != len(self.observations_by_type):
            self._rebuild_observation_source_index()
        if source_call is None:
            return ()
        return self.observations_by_source_call.get(source_call, ())

    def _parameter_schema_json(
        self, target_tool: str, parameter: ToolParameter
    ) -> str:
        key = (target_tool, parameter.name)
        serialized = self._parameter_schema_json_cache.get(key)
        if serialized is None:
            serialized = canonical_json(parameter.schema)
            self._parameter_schema_json_cache[key] = serialized
        return serialized

    def _specification_key(self, specification: Any) -> tuple[str, str]:
        identity = id(specification)
        cached = self._specification_key_cache.get(identity)
        if cached is not None and cached[0] is specification:
            return cached[1]
        key = (
            str(specification.kind),
            canonical_json(json_safe(specification.value)),
        )
        # Retaining the tiny immutable specification also prevents Python from
        # reusing its identity for a later contract revision.
        self._specification_key_cache[identity] = (specification, key)
        return key

    def _serialized_observation_value(self, observation: Observation) -> str:
        serialized = self._observation_value_json.get(observation.type_id)
        if serialized is None:
            serialized = canonical_json(json_safe(observation.value))
            self._observation_value_json[observation.type_id] = serialized
        return serialized

    def _observation_source_calls(self) -> tuple[str, ...]:
        if self._observation_source_index_size != len(self.observations_by_type):
            self._rebuild_observation_source_index()
        return tuple(self.observations_by_source_call)

    def _observation_blueprints(
        self, observation_call: str
    ) -> tuple[tuple[str, int, int, bool], ...]:
        """Return state-independent target templates for one source call."""

        cached = self._observation_task_blueprints.get(observation_call)
        if cached is not None:
            return cached
        observations = self._observations_for_source_call(observation_call)
        if not observations:
            return ()
        source_tool = observations[0].source_tool
        created: list[tuple[str, int, int, bool]] = []
        for tool, contract in self.contracts.manifest.contracts.items():
            if tool == source_tool:
                continue
            if (tool, contract.revision) in self.blocked_contract_revisions:
                continue
            definition = self.definitions[tool]
            if not any(
                schema_accepts(parameter.schema, observation.value)
                for parameter in definition.parameters
                for observation in observations
            ):
                continue
            for index, template in enumerate(contract.templates):
                if template.expected == "success":
                    created.append((
                        tool,
                        contract.revision,
                        index,
                        not template.has_observations(),
                    ))
        result = tuple(created)
        self._observation_task_blueprints[observation_call] = result
        return result

    def _repair_contract(
        self, tool: str, feedback: Mapping[str, Any]
    ) -> ToolCallContract | None:
        previous = self.contracts.get(tool)
        if previous is None:
            return None
        repair_calls_used = max(
            0, self.contracts.total_llm_calls() - self.initial_llm_calls
        )
        repair_limit = self.config.budget.max_contract_repair_calls
        if repair_limit is not None and repair_calls_used >= repair_limit:
            self.contract_repairs_budget_rejected += 1
            self._persist_contract_repair_meta()
            return None
        self.contract_repair_requests += 1

        candidate_set, repaired = self._generate_contract_repair_candidate(
            tool, feedback
        )
        return self._publish_contract_repair_candidate(
            tool=tool,
            previous=previous,
            candidate_set=candidate_set,
            repaired=repaired,
        )

    def _generate_contract_repair_candidate(
        self,
        tool: str,
        feedback: Mapping[str, Any],
        *,
        max_total_llm_calls: int | None = None,
    ) -> tuple[OnlineContractSet, ToolCallContract | None]:
        """Generate one repair without publishing coordinator-visible state.

        Parallel schedulers may call this method concurrently for different
        tools.  Each call owns a private manifest and feedback ledger; the
        coordinator later publishes candidates in deterministic tool order.
        """

        candidate_set = OnlineContractSet(
            manifest=self.contracts.manifest.model_copy(deep=True),
            metadata=self.contracts.metadata,
            generator=self.contracts.generator,
            count_all_history=self.contracts.count_all_history,
            max_total_llm_calls=(
                self.contracts.max_total_llm_calls
                if max_total_llm_calls is None
                else max_total_llm_calls
            ),
            _feedback_seen=deepcopy(self.contracts._feedback_seen),
        )
        custom_repair = self.contracts.__dict__.get("repair")
        if custom_repair is None:
            repaired = candidate_set.repair(tool, feedback)
        else:
            # Test/adaptor implementations sometimes install an instance-level
            # repair function. Run it transactionally too, then immediately
            # restore the active manifest before embedding the candidate.
            active_snapshot = self.contracts.manifest.model_copy(deep=True)
            repaired = custom_repair(tool, feedback)
            candidate_set.manifest = self.contracts.manifest.model_copy(deep=True)
            self.contracts.manifest = active_snapshot
        return candidate_set, repaired

    def _publish_contract_repair_candidate(
        self,
        *,
        tool: str,
        previous: ToolCallContract,
        candidate_set: OnlineContractSet,
        repaired: ToolCallContract | None,
    ) -> ToolCallContract | None:
        """Publish one privately generated candidate as the sole writer."""

        if tool in candidate_set._feedback_seen:
            self.contracts._feedback_seen[tool] = set(
                candidate_set._feedback_seen[tool]
            )
        candidate_audit = list(candidate_set.manifest.audit.get(tool, ()))
        candidate_unknown = candidate_set.manifest.unknown.get(tool)
        if (
            repaired is None
            or repaired.revision <= previous.revision
        ):
            self.contracts.manifest.audit[tool] = candidate_audit
            if candidate_unknown is None:
                self.contracts.manifest.unknown.pop(tool, None)
            else:
                self.contracts.manifest.unknown[tool] = candidate_unknown
            self.contract_repairs_rejected += 1
            self._persist_contract_repair_meta()
            return repaired
        if (
            self.field_priority is not None
            or self.scheduling_priority is not None
        ):
            try:
                if self.field_priority is not None:
                    self.embedding_fields_added += (
                        self.field_priority.ensure_contract(
                            self.definitions[tool], repaired
                        )
                    )
                if (
                    self.scheduling_priority is not None
                    and self.scheduling_priority is not self.field_priority
                ):
                    self.scheduling_embedding_fields_added += (
                        self.scheduling_priority.ensure_contract(
                            self.definitions[tool], repaired
                        )
                    )
            except FieldPriorityError as exc:
                self.stop_reason = "field_embedding_error"
                self.store.set_meta("field_embedding_error", str(exc))
                self.blocked_contract_revisions.add((tool, repaired.revision))
                self.contracts.manifest.audit[tool] = candidate_audit
                self.contract_repairs_rejected += 1
                self._persist_contract_repair_meta()
                return self.contracts.get(tool)
        self.contracts.manifest.contracts[tool] = repaired
        self.contracts.manifest.audit[tool] = candidate_audit
        if candidate_unknown is None:
            self.contracts.manifest.unknown.pop(tool, None)
        else:
            self.contracts.manifest.unknown[tool] = candidate_unknown
        self.contract_repairs_accepted += 1
        self._persist_contract_repair_meta()
        return repaired

    def _persist_contract_repair_meta(self) -> None:
        self.store.set_meta_many({
            "initial_llm_calls": self.initial_llm_calls,
            "contract_repair_requests": self.contract_repair_requests,
            "contract_repairs_accepted": self.contract_repairs_accepted,
            "contract_repairs_rejected": self.contract_repairs_rejected,
            "contract_repairs_budget_rejected": (
                self.contract_repairs_budget_rejected
            ),
            "contract_repair_llm_calls": max(
                0,
                self.contracts.total_llm_calls() - self.initial_llm_calls,
            ),
        })

    def _budget_available(self, extra_calls: int = 1) -> bool:
        budget = self.config.budget
        if not self._storage_budget_available():
            return False
        if self.calls_used + extra_calls > budget.max_calls:
            self._record_budget_violation("max_calls")
            return False
        if budget.max_seconds is not None and self._elapsed() >= budget.max_seconds:
            self._record_budget_violation("max_seconds")
            return False
        return True

    def _activate_state(self, state_id: str) -> None:
        if state_id in self.active_state_ids:
            return
        if not any((
            self.pending_base_by_state.get(state_id),
            self.pending_event_by_state.get(state_id),
            self.pending_continuation_by_state.get(state_id),
            self.pending_provider_by_state.get(state_id),
        )):
            return
        self.active_states.append(state_id)
        self.active_state_ids.add(state_id)

    def _schedule_task(self, task: SearchTask, *, front: bool = False) -> None:
        if task.completed or task.id in self.scheduled_task_ids:
            return
        if self._observation_task_is_pruned(task):
            task.completed = True
            self.observation_tasks_pruned_after_success += 1
            self.store.update_task(task)
            return
        if task.provider_search:
            pending = self.pending_provider_by_state[task.state_id]
        elif task.observation_call is None and task.next_combination == 0:
            pending = self.pending_base_by_state[task.state_id]
        elif task.next_combination == 0:
            pending = self.pending_event_by_state[task.state_id]
        else:
            pending = self.pending_continuation_by_state[task.state_id]
        if front:
            pending.appendleft(task.id)
        else:
            pending.append(task.id)
        self.scheduled_task_ids.add(task.id)
        self._activate_state(task.state_id)

    def _pop_task(self, state_id: str) -> str | None:
        base = self.pending_base_by_state[state_id]
        event = self.pending_event_by_state[state_id]
        continuation = self.pending_continuation_by_state[state_id]
        provider = self.pending_provider_by_state[state_id]
        # Establish every contract's literal/omitted baseline before an
        # observation may replace one of those seeds. Otherwise the first
        # successful provider call can use an unrelated observed value (for
        # example a display label as an account credential), and the global
        # first-value-per-export rule then permanently hides the canonical
        # producer value needed by a downstream lifecycle step. The bulk
        # scheduler gets this property from its frozen initial round; the
        # serial scheduler makes it explicit.
        if base:
            return base.popleft()
        phase = self.schedule_phase[state_id]
        self.schedule_phase[state_id] = (phase + 1) % 3
        queues = (event, continuation, provider)
        preferred = queues[phase]
        if preferred:
            return preferred.popleft()
        # Uniform cyclic fallback: task kinds differ only in how they are
        # generated, never in priority.  In particular, successful-literal
        # provider searches cannot be starved by an expanding continuation
        # queue.
        for offset in range(1, len(queues)):
            pending = queues[(phase + offset) % len(queues)]
            if pending:
                return pending.popleft()
        return None

    def _schedule_tasks(self, tasks: Iterable[SearchTask]) -> None:
        def ordering_key(task: SearchTask) -> tuple[int, float, int, str]:
            tie = stable_id(
                "task_schedule",
                [
                    task.state_id,
                    task.tool,
                    task.revision,
                    task.template_index,
                    task.observation_call,
                    task.provider_search,
                ],
            )
            score = self._task_semantic_score(task)
            # Contract template zero is the required-only canonical baseline.
            # Base calls have no semantic score, so make their declared order
            # explicit instead of allowing a stable hash to choose which
            # template publishes the first value for an export field.
            template_order = (
                task.template_index if task.observation_call is None else 0
            )
            if score is None:
                return (1, 0.0, template_order, tie)
            return (0, -score, template_order, tie)

        ordered = sorted(
            tasks,
            key=ordering_key,
        )
        for task in ordered:
            self._schedule_task(task)

    def _preferred_observation_state_id(
        self, observation_call: str
    ) -> str | None:
        """Return the one state where a source call's head should run first.

        A changed producer starts in its child state.  When a later successful
        state transition actually consumes one of that producer's observations,
        the preferred state follows the value into the retained child.  This
        lets a credential produced by one call meet state established by a
        later call without giving every unrelated state clone another semantic
        head.  All mappings are coordinator-owned memory; scoring never queries
        SQLite.
        """

        return self._observation_call_carried_states.get(
            observation_call
        ) or self._observation_call_output_states.get(
            observation_call
        ) or self._observation_call_input_states.get(observation_call)

    @staticmethod
    def _bound_observation_source_calls(
        call: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return producer calls whose concrete values this call consumed."""

        return tuple(sorted({
            observation.source_call
            for value in (call.get("bindings") or {}).values()
            for observation in binding_observations(value)
        }))

    def _remember_call_bound_values(self, call: Mapping[str, Any]) -> None:
        """Keep the concrete observation inputs of one successful source call."""

        if call.get("outcome") != "success":
            return
        args = dict(call.get("args") or {})
        values = {
            str(name): args[str(name)]
            for name in (call.get("bindings") or {})
            if str(name) in args
        }
        if values:
            self._call_bound_values[str(call["id"])] = values

    def _preferred_bound_observation_source_calls(
        self,
        call: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Choose which consumed producers move their preferred-state head.

        A concrete value can retain many producer provenances for exhaustive
        propagation and edge evidence.  Moving every one of their scheduling
        pointers lets an unrelated same-valued field repeatedly steal those
        producers' semantic heads.  For each bound target field, move only the
        provenance with the best existing public field score, retaining exact
        ties.  This changes scheduling only; ``call["bindings"]`` is untouched.
        """

        bindings = call.get("bindings") or {}
        target_tool = str(call.get("tool") or "")
        definition = self.definitions.get(target_tool)
        if self.field_priority is None or definition is None:
            return self._bound_observation_source_calls(call)
        parameters = {
            parameter.name: parameter
            for parameter in definition.parameters
        }
        selected: set[str] = set()
        for parameter_name, value in bindings.items():
            observations = binding_observations(value)
            if not observations:
                continue
            parameter = parameters.get(str(parameter_name))
            if parameter is None:
                selected.update(row.source_call for row in observations)
                continue
            scored = tuple(
                (
                    self._field_priority_score(
                        observation,
                        target_tool=target_tool,
                        target_parameter=parameter,
                    ),
                    observation,
                )
                for observation in observations
            )
            available = tuple(score for score, _ in scored if score is not None)
            if not available:
                selected.update(row.source_call for row in observations)
                continue
            best = max(available)
            selected.update(
                observation.source_call
                for score, observation in scored
                if score == best
            )
        return tuple(sorted(selected))

    def _carry_bound_observations_to_state(
        self,
        call: Mapping[str, Any],
        state_id: str,
    ) -> None:
        """Advance consumed producers along their one coherent state chain."""

        input_state_id = str(call.get("state_id") or "")
        for source_call in self._preferred_bound_observation_source_calls(call):
            if source_call not in self.observations_by_source_call:
                continue
            preferred_state_id = self._preferred_observation_state_id(source_call)
            if (
                preferred_state_id is not None
                and input_state_id != preferred_state_id
            ):
                continue
            self._observation_call_carried_states[source_call] = state_id

    def _task_field_score(self, task: SearchTask) -> float | None:
        """Return the state-local raw field score before scope gating."""

        if self.scheduling_priority is None or task.observation_call is None:
            return None
        state = self.states.get(task.state_id)
        contract = self.contracts.get(task.tool)
        definition = self.definitions.get(task.tool)
        template_index = task.template_index
        compatible: tuple[ToolCallContract, int] | None = None
        if contract is not None and contract.revision != task.revision:
            compatible = self._compatible_incomplete_contract(task)
            if compatible is not None:
                contract, template_index = compatible
        if (
            state is None
            or contract is None
            or definition is None
            or (
                contract.revision != task.revision
                and compatible is None
            )
            or template_index >= len(contract.templates)
        ):
            return None
        template = contract.templates[template_index]
        baseline_key = (
            state.id, task.tool, contract.revision, template_index
        )
        allow_seed_replacements = (
            task.provider_search or baseline_key not in self.literal_baselines
        )
        cache_key = (
            task.observation_call,
            task.tool,
            contract.revision,
            template_index,
            allow_seed_replacements,
        )
        if cache_key in self._semantic_task_score_cache:
            return self._semantic_task_score_cache[cache_key]
        observations = self._observations_for_source_call(task.observation_call)
        best: float | None = None
        for parameter in definition.parameters:
            specification = template.arguments.get(parameter.name)
            accepts_observation = (
                specification is not None
                and specification.kind == "observation"
            ) or (
                (specification is None or specification.kind in {"literal", "omit"})
                and allow_seed_replacements
            )
            if not accepts_observation:
                continue
            schema_json = self._parameter_schema_json(task.tool, parameter)
            for observation in observations:
                if not _schema_accepts_serialized(
                    schema_json,
                    self._serialized_observation_value(observation),
                ):
                    continue
                score = self._scheduling_priority_score(
                    observation,
                    target_tool=task.tool,
                    target_parameter=parameter,
                )
                if score is not None and (best is None or score > best):
                    best = score
        self._semantic_task_score_cache[cache_key] = best
        return best

    def _bound_field_score(
        self,
        task: SearchTask,
        bindings: Mapping[str, Sequence[Observation]],
    ) -> float | None:
        """Score only triggering-source fields bound in one concrete tuple."""

        if self.scheduling_priority is None or task.observation_call is None:
            return None
        definition = self.definitions.get(task.tool)
        if definition is None:
            return None
        parameters = {
            parameter.name: parameter for parameter in definition.parameters
        }
        source_types = {
            row.type_id
            for row in self._observations_for_source_call(task.observation_call)
        }
        best: float | None = None
        for parameter_name, observations in bindings.items():
            parameter = parameters.get(str(parameter_name))
            if parameter is None:
                continue
            for observation in observations:
                if observation.type_id not in source_types:
                    continue
                score = self._scheduling_priority_score(
                    observation,
                    target_tool=task.tool,
                    target_parameter=parameter,
                )
                if score is not None and (best is None or score > best):
                    best = score
        return best

    def _task_tuple_score(self, task: SearchTask) -> float | None:
        """Return the field score of the tuple at the task's current cursor."""

        combination = self._combination(task)
        if combination is None:
            return None
        _, bindings = combination
        return self._bound_field_score(task, bindings)

    def _task_semantic_score(self, task: SearchTask) -> float | None:
        """Score the current triggering-source tuple in its preferred state."""

        incomplete = self._incomplete_binding(task.id) is not None
        if (
            self.scheduling_priority is None
            or task.observation_call is None
        ):
            return None
        if not incomplete:
            preferred_state_id = self._preferred_observation_state_id(
                task.observation_call
            )
            # Rows without a source call keep insertion order; observations
            # recorded during discovery always have a durable call.
            if (
                preferred_state_id is not None
                and task.state_id != preferred_state_id
            ):
                return None
        return self._task_tuple_score(task)

    def _enqueue_state(self, state: SearchState) -> None:
        created: list[SearchTask] = []
        for tool, contract in self.contracts.manifest.contracts.items():
            if (tool, contract.revision) in self.blocked_contract_revisions:
                continue
            for index, template in enumerate(contract.templates):
                if state.depth > 0 and template.expected == "failure":
                    continue
                task = SearchTask(
                    id=stable_id("task", [state.id, tool, contract.revision, index]),
                    state_id=state.id,
                    tool=tool,
                    revision=contract.revision,
                    template_index=index,
                )
                created.append(task)
        inserted = self.store.put_tasks(created)
        created = [task for task in created if task.id in inserted]
        self._schedule_tasks(created)

    def _enqueue_tool_revision(self, tool: str, revision: int) -> None:
        # Target compatibility is contract-revision dependent.
        self._observation_task_blueprints.clear()
        self._semantic_task_score_cache.clear()
        self._argument_choices_cache.clear()
        self._observation_choices_cache.clear()
        self._parameter_observation_scores_cache.clear()
        if (tool, revision) in self.blocked_contract_revisions:
            return
        contract = self.contracts.get(tool)
        if contract is None or contract.revision != revision:
            return
        created: list[SearchTask] = []
        for state in self.states.values():
            for index, template in enumerate(contract.templates):
                if state.depth > 0 and template.expected == "failure":
                    continue
                task = SearchTask(
                    id=stable_id("task", [state.id, tool, revision, index]),
                    state_id=state.id,
                    tool=tool,
                    revision=revision,
                    template_index=index,
                )
                created.append(task)
        inserted = self.store.put_tasks(created)
        created = [task for task in created if task.id in inserted]
        self._schedule_tasks(created)

    def _enqueue_observation_tasks(
        self, state: SearchState, observation_call: str
    ) -> None:
        created: list[SearchTask] = []
        for tool, revision, index, provider_search in (
            self._observation_blueprints(observation_call)
        ):
            task = SearchTask(
                id=stable_id("task", [
                    state.id,
                    tool,
                    revision,
                    index,
                    observation_call,
                ]),
                state_id=state.id,
                tool=tool,
                revision=revision,
                template_index=index,
                observation_call=observation_call,
                provider_search=provider_search,
            )
            created.append(task)
        inserted = self.store.put_tasks(created)
        created = [task for task in created if task.id in inserted]
        self._schedule_tasks(created)

    def _choices(
        self,
        state: SearchState,
        target_tool: str,
        parameter: ToolParameter,
        template: CallTemplate,
        *,
        allow_observations: bool,
        allow_seed_replacements: bool,
        preferred_observation_call: str | None,
    ) -> Sequence[ArgumentChoice]:
        if self._observation_source_index_size != len(self.observations_by_type):
            self._rebuild_observation_source_index()
        specification = template.arguments.get(parameter.name)
        schema_json = self._parameter_schema_json(target_tool, parameter)
        cache_key = (
            target_tool,
            parameter.name,
            schema_json,
            self._specification_key(specification)
            if specification is not None else None,
            bool(allow_observations),
            bool(allow_seed_replacements),
            preferred_observation_call,
        )
        cached = self._argument_choices_cache.get(cache_key)
        if cached is not None:
            return cached
        choices: list[ArgumentChoice] = []
        if specification is None or specification.kind == "omit":
            choices.append(ArgumentChoice(_OMIT, omitted=True))
        elif specification.kind == "literal":
            choices.append(ArgumentChoice(json_safe(specification.value)))
        accepts_observation = (
            specification is not None
            and specification.kind == "observation"
        ) or (
            (specification is None or specification.kind in {"literal", "omit"})
            and allow_seed_replacements
        )
        if allow_observations and accepts_observation:
            observation_cache_key = (
                target_tool,
                parameter.name,
                schema_json,
            )
            observation_index = self._observation_choices_cache.get(
                observation_cache_key
            )
            if observation_index is not None:
                preferred_entries, preferred_offsets, preferred_score = (
                    self._preferred_observation_view(
                        observation_index,
                        source_call=preferred_observation_call,
                        target_tool=target_tool,
                        target_parameter=parameter,
                    )
                )
                result = _ArgumentChoiceDomain(
                    tuple(choices),
                    observation_index,
                    preferred_observation_call,
                    preferred_entries,
                    preferred_offsets,
                    preferred_score,
                )
                self._argument_choices_cache[cache_key] = result
                return result
            score_cache_key = (
                target_tool,
                parameter.name,
                schema_json,
            )
            scored = self._parameter_observation_scores_cache.get(score_cache_key)
            if scored is None:
                compatible = self._schema_compatible_observations_cache.get(
                    schema_json
                )
                if compatible is None:
                    compatible = tuple(
                        (index, observation)
                        for index, observation in enumerate(
                            self.observations_by_type.values()
                        )
                        if _schema_accepts_serialized(
                            schema_json,
                            self._serialized_observation_value(observation),
                        )
                    )
                    self._schema_compatible_observations_cache[
                        schema_json
                    ] = compatible
                scored_rows: list[tuple[int, float | None, Observation]] = []
                for index, observation in compatible:
                    if observation.source_tool == target_tool:
                        continue
                    score = self._field_priority_score(
                        observation,
                        target_tool=target_tool,
                        target_parameter=parameter,
                    )
                    scored_rows.append((index, score, observation))
                if self.field_priority is not None:
                    scored = tuple(sorted(scored_rows, key=lambda row: (
                        row[1] is None,
                        -(row[1] or 0.0),
                        row[0],
                    )))
                else:
                    scored = tuple(scored_rows)
                self._parameter_observation_scores_cache[score_cache_key] = scored
            records: dict[str, dict[str, Any]] = {}
            for original_index, score, observation in scored:
                key = self._serialized_observation_value(observation)
                record = records.get(key)
                if record is None:
                    records[key] = {
                        "score": score,
                        "first_index": original_index,
                        "value": observation.value,
                        "observations": {observation.type_id: observation},
                    }
                    continue
                # One concrete value is invoked once, but every producer of
                # that value remains evidence for the binding. ``scored`` is
                # already ordered best-first, so the first row supplies the
                # value's queue score while later rows only add provenance.
                record["observations"].setdefault(
                    observation.type_id, observation
                )

            grouped_records: list[list[dict[str, Any]]] = []
            for record in records.values():
                if (
                    not grouped_records
                    or grouped_records[-1][0]["score"] != record["score"]
                ):
                    grouped_records.append([])
                grouped_records[-1].append(record)

            built_groups: list[_ObservationChoiceGroup] = []
            group_starts: list[int] = []
            observation_count = 0
            value_locations: dict[
                str, tuple[int, int, ArgumentChoice]
            ] = {}
            for group_records in grouped_records:
                group_starts.append(observation_count)
                group_records.sort(key=lambda row: int(row["first_index"]))
                group_choices = tuple(
                    ArgumentChoice(
                        value=record["value"],
                        observations=tuple(record["observations"].values()),
                    )
                    for record in group_records
                )
                group_index = len(built_groups)
                value_offsets = {
                    self._serialized_observation_value(
                        next(iter(record["observations"].values()))
                    ): offset
                    for offset, record in enumerate(group_records)
                }
                group = _ObservationChoiceGroup(
                    choices=group_choices,
                    value_offsets=value_offsets,
                )
                built_groups.append(group)
                observation_count += len(group_choices)
                for value_key, offset in value_offsets.items():
                    value_locations[value_key] = (
                        group_index,
                        offset,
                        group_choices[offset],
                    )
            observation_index = _ObservationChoiceIndex(
                groups=tuple(built_groups),
                value_locations=value_locations,
                group_starts=tuple(group_starts),
            )
            self._observation_choices_cache[
                observation_cache_key
            ] = observation_index
            preferred_entries, preferred_offsets, preferred_score = (
                self._preferred_observation_view(
                    observation_index,
                    source_call=preferred_observation_call,
                    target_tool=target_tool,
                    target_parameter=parameter,
                )
            )
            result = _ArgumentChoiceDomain(
                tuple(choices),
                observation_index,
                preferred_observation_call,
                preferred_entries,
                preferred_offsets,
                preferred_score,
            )
            self._argument_choices_cache[cache_key] = result
            return result
        # Invocation candidates are values, not provenance identities.  Keep a
        # contract literal as its own Hamming coordinate, then coalesce only the
        # observation suffix so one concrete value is tried once while every
        # producer remains available to evidence projection.
        unique: list[ArgumentChoice] = []
        observation_choice_by_value: dict[str, int] = {}
        for choice in choices:
            if choice.omitted or not choice.observations:
                unique.append(choice)
                continue
            key = canonical_json(json_safe(choice.value))
            existing_index = observation_choice_by_value.get(key)
            if existing_index is None:
                observation_choice_by_value[key] = len(unique)
                unique.append(choice)
                continue
            existing = unique[existing_index]
            unique[existing_index] = ArgumentChoice(
                value=existing.value,
                observations=tuple({
                    observation.type_id: observation
                    for observation in (*existing.observations, *choice.observations)
                }.values()),
            )
        result = tuple(unique)
        self._argument_choices_cache[cache_key] = result
        return result

    def _provider_anchor_combination(
        self, task: SearchTask
    ) -> tuple[dict[str, Any], dict[str, tuple[Observation, ...]]] | None:
        if task.observation_call is None:
            return None
        winning_call_id = self._anchor_success_call_by_tool.get(task.tool)
        winning_call = (
            self.store.call(winning_call_id) if winning_call_id is not None else None
        )
        state = self.states.get(task.state_id)
        contract = self.contracts.get(task.tool)
        definition = self.definitions.get(task.tool)
        if (
            winning_call is None
            or state is None
            or contract is None
            or contract.revision != task.revision
            or definition is None
            or self._matching_selected_template_index(contract, winning_call)
            != task.template_index
        ):
            return None
        observations = self._observations_for_source_call(task.observation_call)
        if not observations:
            return None
        selected_parameter: ToolParameter | None = None
        selected_observation: Observation | None = None
        selected_score: float | None = None
        for parameter in definition.parameters:
            schema_json = self._parameter_schema_json(task.tool, parameter)
            for observation in observations:
                if observation.location == "arguments":
                    continue
                if not _schema_accepts_serialized(
                    schema_json,
                    self._serialized_observation_value(observation),
                ):
                    continue
                score = self._field_priority_score(
                    observation,
                    target_tool=task.tool,
                    target_parameter=parameter,
                )
                if selected_parameter is None or (
                    score is not None
                    and (selected_score is None or score > selected_score)
                ):
                    selected_parameter = parameter
                    selected_observation = observation
                    selected_score = score
        if selected_parameter is None or selected_observation is None:
            return None

        value_key = self._serialized_observation_value(selected_observation)
        selected_bindings = tuple(
            observation
            for observation in self.observations_by_value.get(value_key, ())
            if observation.source_tool != task.tool
        )
        if not any(
            observation.source_call == task.observation_call
            for observation in selected_bindings
        ):
            return None

        args = dict(winning_call.get("args") or {})
        args[selected_parameter.name] = json_safe(selected_observation.value)
        return args, {
            selected_parameter.name: selected_bindings,
        }

    def _combination(
        self, task: SearchTask
    ) -> tuple[dict[str, Any], dict[str, tuple[Observation, ...]]] | None:
        if task.anchor_search:
            return self._provider_anchor_combination(task)
        state = self.states[task.state_id]
        definition = self.definitions[task.tool]
        contract = self.contracts.get(task.tool)
        if contract is None or contract.revision != task.revision:
            return None
        template = contract.templates[task.template_index]
        baseline_key = (
            state.id, task.tool, task.revision, task.template_index
        )
        allow_seed_replacements = (
            task.provider_search or baseline_key not in self.literal_baselines
        )
        domains: list[Sequence[ArgumentChoice]] = []
        for parameter in definition.parameters:
            choices = self._choices(
                state,
                task.tool,
                parameter,
                template,
                allow_observations=template.expected == "success",
                allow_seed_replacements=allow_seed_replacements,
                preferred_observation_call=task.observation_call,
            )
            if not choices:
                return None
            domains.append(choices)
        sizes = tuple(len(domain) for domain in domains)
        if task.observation_call:
            preferred = tuple(
                (
                    domain.preferred_indexes(task.observation_call)
                    if isinstance(domain, _ArgumentChoiceDomain)
                    else frozenset(
                        index for index, choice in enumerate(domain)
                        if any(
                            observation.source_call == task.observation_call
                            for observation in choice.observations
                        )
                    )
                )
                for domain in domains
            )
            preferred_scores = tuple(
                domain.preferred_score
                if isinstance(domain, _ArgumentChoiceDomain)
                else None
                for domain in domains
            )
            coordinates = source_coherent_product_index(
                sizes,
                preferred,
                task.next_combination,
                preferred_scores=preferred_scores,
            )
        else:
            producer_call = state.producer_call
            producer_values = (
                self._call_bound_values.get(producer_call, {})
                if (
                    producer_call is not None
                    and task.tool != self._state_producer_tool(state)
                )
                else {}
            )
            promoted_indexes = tuple(
                (
                    domain.observation_index_for_value(
                        producer_values[parameter.name]
                    )
                    if (
                        parameter.name in producer_values
                        and isinstance(domain, _ArgumentChoiceDomain)
                    )
                    else None
                )
                for parameter, domain in zip(
                    definition.parameters, domains, strict=True
                )
            )
            coordinates = promoted_product_index(
                sizes,
                task.next_combination,
                promoted_indexes,
            )
        if coordinates is None:
            return None
        selected = tuple(
            domain[coordinate]
            for domain, coordinate in zip(domains, coordinates, strict=True)
        )
        args: dict[str, Any] = {}
        bindings: dict[str, tuple[Observation, ...]] = {}
        for parameter, choice in zip(definition.parameters, selected):
            if choice.omitted:
                continue
            args[parameter.name] = json_safe(choice.value)
            if choice.observations:
                bindings[parameter.name] = choice.observations
        return args, bindings

    def _estimate_task_calls(self, task: SearchTask) -> int:
        """Return the shared reservation estimate for one task work item."""

        combination = self._combination(task)
        if combination is None:
            return 1
        args, bindings = combination
        binding_key = stable_id("binding", [
            task.state_id,
            task.tool,
            task.revision,
            task.template_index,
            args,
        ])
        primary = int(self.store.call_by_binding(binding_key) is None)
        trials = len(bindings) * (
            (self.config.trial_repetitions - 1)
            + self.config.trial_repetitions
        )
        state_trials = (
            3 * self.config.trial_repetitions
            if self.states[task.state_id].producer_call is not None
            else 0
        )
        return max(1, primary + trials + state_trials)

    async def _execute_invocation_async(
        self,
        *,
        state: SearchState,
        tool: str,
        args: Mapping[str, Any],
        save_on_change: bool,
        binding_key: str,
        reason: str,
    ) -> tuple[InvocationResult, str | None] | None:
        """Execute one runtime attempt; orchestration subclasses override this."""

        if not self._budget_available():
            return None
        reuse_loaded_state = (
            self._loaded_runtime_state_pristine
            and self._loaded_runtime_state_handle == state.handle
        )
        try:
            if not reuse_loaded_state:
                self.runtime.load_state(state.handle)
            invoke_from_state = getattr(self.runtime, "invoke_from_state", None)
            if invoke_from_state is None:
                result = self.runtime.invoke(tool, args)
            else:
                result = invoke_from_state(
                    tool,
                    args,
                    before_digest=state.digest,
                )
        except Exception:
            # A failed runtime operation may have partially mutated the live
            # world, so no following call may reuse it.
            self._loaded_runtime_state_handle = None
            self._loaded_runtime_state_pristine = False
            raise
        self._loaded_runtime_state_handle = state.handle
        self._loaded_runtime_state_pristine = not result.transition.changed
        self.calls_used += 1
        saved_handle = None
        if save_on_change and result.transition.changed:
            saved_handle = self.runtime.save_state(stable_id(
                "post_call",
                [binding_key, reason, self.calls_used, result.transition.after_digest],
            ))
        # A changed invocation may have materialized a large immutable state.
        # Account for it before any control trials can start.  The serial
        # callback is lightweight/incremental; the parallel override uses its
        # durable-checkpoint cadence and therefore does not walk here.
        self._storage_budget_available()
        return result, saved_handle

    def _invocation_ordinal(self) -> int:
        return self.calls_used

    def _request_contract_repair(
        self,
        tool: str,
        feedback: Mapping[str, Any],
        *,
        kind: str,
        source_call_id: str | None = None,
    ) -> ToolCallContract | None:
        """Publish a repair immediately; parallel orchestration buffers it."""

        del kind, source_call_id
        return self._repair_contract(tool, feedback)

    def _resolve_state_export_contract(
        self,
        *,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> ToolCallContract:
        return self._repair_missing_state_exports(
            call=call,
            result=result,
            contract=contract,
        )

    def _trial_budget_available(self, needed: int) -> bool:
        """Serial trials start only when their full control budget is available."""

        return self._budget_available(needed)

    async def _invoke_async(
        self,
        *,
        state: SearchState,
        tool: str,
        revision: int,
        args: Mapping[str, Any],
        bindings: Mapping[str, Sequence[Observation]],
        expected: str,
        reason: str,
        binding_key: str,
        template_index: int | None = None,
        allow_repair: bool = False,
        save_on_change: bool = False,
    ) -> tuple[dict[str, Any], InvocationResult, str, str | None] | None:
        recorded = self.store.call_by_binding(binding_key)
        if recorded is not None:
            return (
                recorded,
                InvocationResult.from_dict(recorded["result"]),
                str(recorded.get("outcome") or "unknown"),
                str(recorded.get("saved_state_handle") or "") or None,
            )
        executed = await self._execute_invocation_async(
            state=state,
            tool=tool,
            args=args,
            save_on_change=save_on_change,
            binding_key=binding_key,
            reason=reason,
        )
        if executed is None:
            return None
        result, saved_handle = executed
        if getattr(self, "_bulk_run_active", False):
            recorded = self.store.call_by_binding(binding_key)
            if recorded is not None:
                return (
                    recorded,
                    InvocationResult.from_dict(recorded["result"]),
                    str(recorded.get("outcome") or "unknown"),
                    str(recorded.get("saved_state_handle") or "") or None,
                )
        if "max_storage_bytes" in self._active_budget_violations and saved_handle:
            # The one atomic save is the only unavoidable overshoot.  It has
            # not entered the state graph yet, so release that exact handle
            # immediately and retain only the invocation record below.
            self._discard_unreferenced_state_handle(saved_handle)
            saved_handle = None
            self._storage_budget_available()
        contract = self.contracts.get(tool)
        if contract is None:
            if saved_handle:
                self._discard_unreferenced_state_handle(saved_handle)
            return None
        executed_revision = int(revision)
        schema_contract_revision = contract.revision
        template_fingerprint: str | None = None
        if (
            contract.revision == executed_revision
            and template_index is not None
            and 0 <= template_index < len(contract.templates)
        ):
            template_fingerprint = selected_template_fingerprint(
                contract.templates[template_index]
            )
        schema_outcome = self._classify_invocation_outcome(contract, result)
        # A cleanly classified business rejection is search evidence, not a
        # malformed contract.  Repairing on an expected-success rejection can
        # consume the final JIT revision on ordinary state such as a duplicate
        # record and can erase exports learned by an earlier state repair.
        # JIT outcome repair is therefore restricted to observable schema
        # ambiguity/mismatch; literal and state diagnosis have their own
        # explicit repair paths below.
        should_repair = (
            schema_outcome == "conflict"
            or (result.status == "success" and schema_outcome != "success")
        )
        if should_repair and allow_repair and self.stop_reason == "completed":
            previous_revision = contract.revision
            repaired = self._request_contract_repair(
                tool,
                structural_feedback(
                    example_expected=expected,
                    result=result,
                    observed_outcome=schema_outcome,
                ),
                kind="outcome",
            )
            if repaired is not None and repaired.revision > previous_revision:
                schema_contract_revision = repaired.revision
                schema_outcome = classify_outcome(repaired, result)
                self._enqueue_tool_revision(tool, repaired.revision)
        # A present value that contradicts an explicit business discriminator
        # is not yet evidence that the tool succeeded.  In parallel runs the
        # repair is published at the epoch boundary; activating here would let
        # a provisional transport fallback prune the repaired revision first.
        defer_for_discriminator_repair = (
            allow_repair
            and result.status == "success"
            and schema_outcome == "unknown"
            and _explicit_value_contradiction(
                contract.success_schema, normalized_invocation(result)
            )
        )
        outcome = (
            schema_outcome
            if defer_for_discriminator_repair
            else effective_invocation_outcome(result, schema_outcome)
        )
        ordinal = self._invocation_ordinal()
        call_id = stable_id("call", {
            "state": state.id,
            "tool": tool,
            "revision": executed_revision,
            "schema_revision": schema_contract_revision,
            "args": args,
            "bindings": {
                key: [observation.id for observation in observations]
                for key, observations in bindings.items()
            },
            "reason": reason,
            "template_index": template_index,
            "binding": binding_key,
        })
        call = {
            "id": call_id,
            "state_id": state.id,
            "state_digest": state.digest,
            "state_handle": state.handle,
            "tool": tool,
            # This is the revision whose selected template produced `args`.
            # Outcome repair may classify the immutable result with a newer
            # schema, recorded separately without relabelling the invocation.
            "contract_revision": executed_revision,
            "schema_contract_revision": schema_contract_revision,
            "selected_template_fingerprint": template_fingerprint,
            "args": json_safe(args),
            "bindings": serialized_bindings(bindings),
            "template_index": template_index,
            "expected": expected,
            "reason": reason,
            "ordinal": ordinal,
            "schema_outcome": schema_outcome,
            "outcome": outcome,
            "result": result.to_dict(),
            "saved_state_handle": saved_handle,
        }
        if not self.store.put_call(call, binding_key):
            raise RuntimeError(
                "actual invocation collided with an existing call: "
                f"ordinal={ordinal} binding={binding_key} call={call_id} "
                f"tool={tool} reason={reason}"
            )
        self._remember_call_bound_values(call)
        if outcome == "success" and reason == "template_search":
            if tool not in self._first_success_call_by_tool:
                self._first_success_call_by_tool[tool] = call_id
                self._first_success_binding_by_tool[tool] = binding_key
            anchor_call_id = self._anchor_success_call_by_tool.get(tool)
            anchor_call = (
                self.store.call(anchor_call_id)
                if anchor_call_id is not None else None
            )
            if (
                anchor_call is None
                or len(call.get("args") or {})
                > len(anchor_call.get("args") or {})
            ):
                self._anchor_success_call_by_tool[tool] = call_id
                self._enqueue_provider_anchor_tasks(target_tool=tool)
        if outcome == "success":
            self._record_first_tool_success(call)
        # Include the newly committed SQLite row even when the runtime did not
        # change.  This is three file stats plus the runtime's cached counter,
        # not a candidate- or checkpoint-tree traversal.
        self._storage_budget_available()
        if "max_storage_bytes" in self._active_budget_violations and saved_handle:
            # SQLite itself can be the write that crosses the boundary.  Drop
            # the still-unreferenced branch and clear the call's pointer.
            self._discard_unreferenced_state_handle(saved_handle)
            saved_handle = None
            call["saved_state_handle"] = None
            self.store.update_call(call)
            self._storage_budget_available()
        self._record_literal_baseline(call)
        return call, result, outcome, saved_handle

    def _classify_invocation_outcome(
        self, contract: ToolCallContract, result: InvocationResult
    ) -> str:
        return classify_outcome(contract, result)

    def _record_literal_baseline(self, call: Mapping[str, Any]) -> None:
        if (
            call.get("outcome") != "success"
            or call.get("schema_outcome") != "success"
        ):
            return
        tool = str(call.get("tool"))
        contract = self.contracts.get(tool)
        if contract is None:
            return
        template_index = self._matching_selected_template_index(contract, call)
        if template_index is None:
            return
        state_identity = str(call["state_id"])
        args = dict(call.get("args") or {})
        bound = set((call.get("bindings") or {}).keys())
        template = contract.templates[template_index]
        if template.expected != "success":
            return
        literal_names = {
            name for name, row in template.arguments.items()
            if row.kind == "literal"
        }
        if literal_names & bound:
            return
        if any(
            (
                specification.kind == "literal"
                and (
                    name not in args
                    or canonical_json(args[name])
                    != canonical_json(specification.value)
                )
            )
            or (specification.kind == "omit" and name in args)
            for name, specification in template.arguments.items()
        ):
            return
        self.literal_baselines.add((
            state_identity, tool, contract.revision, template_index
        ))

    @staticmethod
    def _matching_selected_template_index(
        contract: ToolCallContract,
        call: Mapping[str, Any],
    ) -> int | None:
        """Resolve a call's selected template in the active contract.

        Missing legacy provenance and a changed repair template are both
        conservative misses: neither may suppress search or reinterpret an
        old invocation under a new argument contract.
        """

        fingerprint = call.get("selected_template_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        preferred = call.get("template_index")
        if preferred is not None:
            index = int(preferred)
            if (
                0 <= index < len(contract.templates)
                and selected_template_fingerprint(contract.templates[index])
                == fingerprint
            ):
                return index
        return next((
            index
            for index, template in enumerate(contract.templates)
            if selected_template_fingerprint(template) == fingerprint
        ), None)

    def _observations(
        self,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> tuple[Observation, ...]:
        rows: list[Observation] = []
        argument_bindings = {
            str(name): binding_observations(value)
            for name, value in (call.get("bindings") or {}).items()
        }
        sources = {
            "arguments": call["args"],
            "response": result.response,
            "state_changes": result.transition.changes,
        }
        for export in contract.exports:
            if export.source != "arguments" and not export.propagate:
                continue
            for path, child in select_export_values(export, sources[export.source]):
                if export.source == "arguments":
                    parameter = path.removeprefix("$.")
                    forwarded = tuple(
                        observation
                        for observation in argument_bindings.get(parameter, ())
                        if canonical_json(observation.value) == canonical_json(child)
                    )
                    if forwarded:
                        rows.extend(forwarded)
                        continue
                if len(canonical_json(child).encode("utf-8")) > self.config.max_observation_bytes:
                    continue
                rows.append(Observation.create(
                    child,
                    source_tool=str(call["tool"]),
                    source_call=str(call["id"]),
                    # The field identity belongs to the contract revision that
                    # declares this export.  JIT state repair may legitimately
                    # apply a newer selector to the successful call that
                    # triggered the repair.
                    source_revision=contract.revision,
                    location=export.source,
                    export_name=export.name,
                    path=path,
                    value_schema=export.value_schema,
                    field_path=export.path,
                    # An explicit export represents one field even when it
                    # matches many array elements.  A recursive anonymous
                    # wildcard is different: its concrete leaf paths are the
                    # only formal distinction between unrelated state fields.
                    # Keep one value globally per such path rather than
                    # collapsing every string/number selected by the wildcard.
                    type_path=(path if export.path == "$..[*]" else None),
                ))
        # An export field is one observation type. The first schema-valid
        # value admitted anywhere in the build wins; later values from that
        # field never enter the global value pool or create event tasks.
        admitted: list[Observation] = []
        local_types: set[str] = set()
        for row in rows:
            if (
                row.type_id in self.admitted_observation_types
                or row.type_id in local_types
            ):
                self.observation_types_dropped += 1
                continue
            local_types.add(row.type_id)
            admitted.append(row)
        return tuple(admitted)

    def _create_state(
        self,
        parent: SearchState,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
        *,
        handle: str,
    ) -> bool:
        candidate_rows = self._observations(call, result, contract)
        inserted_types = self.store.put_observations(candidate_rows)
        new_rows = tuple(
            row for row in candidate_rows if row.type_id in inserted_types
        )
        for row in new_rows:
            self.observations_by_type[row.type_id] = row
        if new_rows:
            self._observation_call_input_states[str(call["id"])] = parent.id
            # Update once after the batch so all subsequent event-task scores
            # and compatibility checks use indexed source data.
            self._index_new_observations(new_rows)
        self.admitted_observation_types.update(inserted_types)

        retain_handle = not result.transition.changed
        if (
            result.transition.changed
            and parent.depth < self.config.budget.max_depth
        ):
            # Observations are global values, not part of runtime-state identity.
            # A runtime digest may omit concrete values, so the saved transition
            # handle is the identity of the branch that workers can reload.
            state_id = stable_id("state", {"handle": handle})
            producer_tool = str(call["tool"])
            if state_id in self.states:
                retain_handle = True
                if new_rows:
                    self._observation_call_output_states[
                        str(call["id"])
                    ] = state_id
                self._carry_bound_observations_to_state(call, state_id)
            elif len(self.states) < self.config.budget.max_states:
                state = SearchState(
                    id=state_id,
                    handle=handle,
                    digest=result.transition.after_digest,
                    depth=parent.depth + 1,
                    producer_call=str(call["id"]),
                    producer_before_handle=parent.handle,
                )
                if self.store.put_state(state):
                    retain_handle = True
                    self.states[state.id] = state
                    self._live_state_handles.add(state.handle)
                    self._state_producer_tools[state.id] = producer_tool
                    if new_rows:
                        self._observation_call_output_states[
                            str(call["id"])
                        ] = state.id
                    self._carry_bound_observations_to_state(call, state.id)
                    self._enqueue_state(state)
                    # Values discovered before this world state existed must
                    # also receive one source-triggered attempt in the new
                    # state. One global value per field type bounds this set.
                    for observation_call in sorted(
                        self._observation_source_calls()
                    ):
                        self._enqueue_observation_tasks(
                            state, observation_call
                        )

        if result.transition.changed and not retain_handle:
            self._discard_unreferenced_state_handle(handle)
        if not new_rows:
            return retain_handle
        # A field type enters the system exactly once. Its value can be passed
        # to a tool in any world state, so trigger one event task per state.
        for state in tuple(self.states.values()):
            self._enqueue_observation_tasks(state, str(call["id"]))
        self._enqueue_provider_anchor_tasks(observation_call=str(call["id"]))
        return retain_handle

    def _discard_unreferenced_state_handle(self, handle: str) -> None:
        # A successful primary call can publish its child before its
        # dependency-control trials finish.  Such a handle is no longer
        # transient even while the owning task remains incomplete.
        if handle in self._live_state_handles:
            return
        discard = getattr(self.runtime, "discard_state", None)
        if discard is not None:
            discard(handle)

    def _discard_incomplete_task_state(self, task_id: str) -> None:
        binding_key = self._incomplete_binding(task_id)
        if binding_key is None:
            return
        call = self.store.call_by_binding(binding_key)
        handle = str((call or {}).get("saved_state_handle") or "")
        if handle and handle not in self._live_state_handles:
            self._discard_unreferenced_state_handle(handle)
        self._clear_incomplete_binding(task_id)

    async def _run_input_trials_async(
        self,
        *,
        state: SearchState,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
        template_index: int,
        bindings: Mapping[str, Sequence[Observation]],
    ) -> bool:
        template = contract.templates[template_index]
        promoted_literals: list[str] = []
        for parameter, observations in bindings.items():
            specification = template.arguments.get(parameter)
            # Omission and value provenance are orthogonal.  An omitted base
            # head still has observation-supplied variants; when one of those
            # variants runs it must receive the same trial/evidence treatment
            # as an explicitly observed argument.
            declared_mode = (
                specification.kind if specification is not None else "omit"
            )
            value_trial_id = stable_id("input_value_trial", [
                state.id,
                call["tool"],
                call["args"],
                parameter,
                call["contract_revision"],
                template_index,
            ])
            missing: list[tuple[str, Observation]] = []
            for observation in observations:
                trial_id = stable_id("input_trial", [
                    value_trial_id,
                    observation.type_id,
                ])
                if not self.store.has_row("input_trials", trial_id):
                    missing.append((trial_id, observation))
            if not missing:
                continue

            prior = self.store.input_trial_for_value(value_trial_id)
            if prior is not None:
                required = bool(prior["required"])
                success_calls = list(map(str, prior["success_calls"]))
                control_calls = list(map(str, prior["control_calls"]))
            else:
                control_args = dict(call["args"])
                if specification is not None and specification.kind == "literal":
                    control_args[parameter] = json_safe(specification.value)
                else:
                    control_args.pop(parameter, None)
                needed = self.config.trial_repetitions - 1
                if canonical_json(control_args) != canonical_json(call["args"]):
                    needed += self.config.trial_repetitions
                if not self._trial_budget_available(needed):
                    return False
                success_calls = [str(call["id"])]
                success_outcomes = ["success"]
                for repeat in range(self.config.trial_repetitions - 1):
                    replay = await self._invoke_async(
                        state=state,
                        tool=str(call["tool"]),
                        revision=int(call["contract_revision"]),
                        args=dict(call["args"]),
                        bindings=bindings,
                        expected="success",
                        reason="input_repeat",
                        binding_key=stable_id(
                            "binding", [value_trial_id, "success", repeat]
                        ),
                    )
                    if replay is None:
                        return False
                    success_calls.append(str(replay[0]["id"]))
                    success_outcomes.append(replay[2])
                control_calls = []
                control_outcomes: list[str] = []
                if canonical_json(control_args) == canonical_json(call["args"]):
                    control_calls = [str(call["id"])] * self.config.trial_repetitions
                    control_outcomes = ["success"] * self.config.trial_repetitions
                else:
                    control_bindings = {
                        key: value for key, value in bindings.items()
                        if key != parameter
                    }
                    for repeat in range(self.config.trial_repetitions):
                        control = await self._invoke_async(
                            state=state,
                            tool=str(call["tool"]),
                            revision=int(call["contract_revision"]),
                            args=control_args,
                            bindings=control_bindings,
                            expected="failure",
                            reason="input_ablation",
                            binding_key=stable_id(
                                "binding", [value_trial_id, "control", repeat]
                            ),
                        )
                        if control is None:
                            return False
                        control_calls.append(str(control[0]["id"]))
                        control_outcomes.append(control[2])
                if not all(outcome == "success" for outcome in success_outcomes):
                    return True
                if not all(
                    outcome in {"success", "failure"}
                    for outcome in control_outcomes
                ):
                    return True
                # `required` asks whether this field's value needs an upstream
                # producer. Omission is orthogonal. A declared literal makes
                # the field independent only when its control call actually
                # succeeds; a proposed but unusable scalar does not count.
                literal_succeeded = (
                    specification is not None
                    and specification.kind == "literal"
                    and any(
                        outcome == "success" for outcome in control_outcomes
                    )
                )
                required = not literal_succeeded
                literal_needs_repair = (
                    specification is not None
                    and specification.kind == "literal"
                    and not literal_succeeded
                )

            for trial_id, observation in missing:
                trial_row = {
                    "id": trial_id,
                    "value_trial_id": value_trial_id,
                    "source_type_id": observation.type_id,
                    "source_field_id": observation.field_id,
                    "source": observation.source_tool,
                    "source_call": observation.source_call,
                    "source_revision": observation.source_revision,
                    "source_location": observation.location,
                    "source_export": observation.export_name,
                    "source_path": observation.path,
                    "target": str(call["tool"]),
                    "target_revision": int(call["contract_revision"]),
                    "target_parameter": parameter,
                    "target_template": template_index,
                    "declared_mode": declared_mode,
                    "required": required,
                    "state_id": state.id,
                    "success_calls": success_calls,
                    "control_calls": control_calls,
                }
                if self.store.put_input_trial(trial_row):
                    self._record_discovered_edge("input", trial_row)
            if prior is None and literal_needs_repair:
                promoted_literals.append(parameter)
        if promoted_literals:
            previous_revision = contract.revision
            repaired = self._request_contract_repair(
                str(call["tool"]),
                {
                    "event": "literal_ablation",
                    "parameters": sorted(set(promoted_literals)),
                    "success_repetitions": self.config.trial_repetitions,
                    "control_repetitions": self.config.trial_repetitions,
                    "conclusion": (
                        "observations_succeeded_and_contract_literals_failed"
                    ),
                },
                kind="literal",
                source_call_id=str(call["id"]),
            )
            if repaired is not None and repaired.revision > previous_revision:
                self._record_literal_baseline(call)
                self._enqueue_tool_revision(str(call["tool"]), repaired.revision)
        return True

    async def _run_state_trial_async(
        self,
        *,
        state: SearchState,
        call: Mapping[str, Any],
        contract: ToolCallContract,
    ) -> bool:
        if state.producer_call is None or state.producer_before_handle is None:
            return True
        producer = self.store.call(state.producer_call)
        if producer is None:
            return True
        if str(producer.get("tool") or "") == str(call.get("tool") or ""):
            return True
        if not bool(producer["result"]["transition"].get("changed")):
            return True
        trial_id = stable_id("state_trial", [
            producer["id"], call["tool"], call["args"], call["contract_revision"]
        ])
        if self.store.has_row("state_trials", trial_id):
            return True
        needed = 3 * self.config.trial_repetitions
        if not self._trial_budget_available(needed):
            return False
        before_outcomes: list[str] = []
        after_outcomes: list[str] = []
        evidence_calls: list[str] = []
        before_state = SearchState(
            id=stable_id("control_state", state.producer_before_handle),
            handle=state.producer_before_handle,
            digest=str(producer["result"]["transition"]["before_digest"]),
            depth=max(0, state.depth - 1),
        )
        for repeat in range(self.config.trial_repetitions):
            source_binding = stable_id(
                "binding", [trial_id, "source", repeat]
            )
            after_binding = stable_id(
                "binding", [trial_id, "after", repeat]
            )
            before = await self._invoke_async(
                state=before_state,
                tool=str(call["tool"]),
                revision=contract.revision,
                args=dict(call["args"]),
                bindings={},
                expected="failure",
                reason="state_before",
                binding_key=stable_id("binding", [trial_id, "before", repeat]),
            )
            if before is None:
                return False
            before_outcomes.append(before[2])
            evidence_calls.append(str(before[0]["id"]))
            if before[2] != "failure":
                return True

            # A parallel epoch may pause after the replayed source and target
            # have both completed but before the state-trial row is reduced.
            # Their exact repeat bindings are the durable progress record; do
            # not recreate an already completed replay handle merely to reach
            # a target result that is already committed.
            completed_source = self.store.call_by_binding(source_binding)
            completed_after = self.store.call_by_binding(after_binding)
            if completed_source is not None and completed_after is not None:
                source_outcome = str(
                    completed_source.get("outcome") or "unknown"
                )
                after_outcome = str(
                    completed_after.get("outcome") or "unknown"
                )
                if source_outcome != "success":
                    return True
                after_outcomes.append(after_outcome)
                evidence_calls.extend([
                    str(completed_source["id"]),
                    str(completed_after["id"]),
                ])
                if after_outcome != "success":
                    return True
                continue

            replay_source = await self._invoke_async(
                state=before_state,
                tool=str(producer["tool"]),
                revision=int(producer["contract_revision"]),
                args=dict(producer["args"]),
                bindings={},
                expected="success",
                reason="state_source_replay",
                binding_key=source_binding,
                save_on_change=True,
            )
            if replay_source is None:
                return False
            if replay_source[2] != "success" or replay_source[3] is None:
                if replay_source[3] is not None:
                    self._discard_unreferenced_state_handle(replay_source[3])
                return True
            replay_handle = replay_source[3]
            try:
                replay_state = SearchState(
                    id=stable_id("replay_state", [trial_id, repeat]),
                    handle=replay_handle,
                    digest=replay_source[1].transition.after_digest,
                    depth=state.depth,
                )
                after = await self._invoke_async(
                    state=replay_state,
                    tool=str(call["tool"]),
                    revision=contract.revision,
                    args=dict(call["args"]),
                    bindings={},
                    expected="success",
                    reason="state_after",
                    binding_key=after_binding,
                )
                if after is None:
                    return False
                after_outcomes.append(after[2])
                evidence_calls.extend([
                    str(replay_source[0]["id"]), str(after[0]["id"])
                ])
                if after[2] != "success":
                    return True
            finally:
                self._discard_unreferenced_state_handle(replay_handle)
        if not all(outcome == "failure" for outcome in before_outcomes):
            return True
        if not all(outcome == "success" for outcome in after_outcomes):
            return True
        change_pointers = sorted(
            pointer for pointer, _ in iter_json_values(
                producer["result"]["transition"].get("changes")
            )
        )
        trial_row = {
            "id": trial_id,
            "source": str(producer["tool"]),
            "source_call": str(producer["id"]),
            "source_revision": int(producer["contract_revision"]),
            "target": str(call["tool"]),
            "target_revision": int(call["contract_revision"]),
            "required": True,
            "state_changes": change_pointers,
            "evidence_calls": evidence_calls,
        }
        if self.store.put_state_trial(trial_row):
            self._record_discovered_edge("state", trial_row)
        return True

    def _repair_missing_state_exports(
        self,
        *,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> ToolCallContract:
        if not result.transition.changed:
            return contract
        existing = [
            row for row in contract.exports if row.source == "state_changes"
        ]
        if existing and self._contract_has_matching_state_exports(result, contract):
            return contract
        repair_feedback = self._state_export_repair_feedback(result, contract)
        previous_revision = contract.revision
        repaired = self._repair_contract(str(call["tool"]), repair_feedback)
        if repaired is not None and repaired.revision > previous_revision:
            self._enqueue_tool_revision(str(call["tool"]), repaired.revision)
            if self.stop_reason == "field_embedding_error":
                return repaired
            if self._contract_has_matching_state_exports(result, repaired):
                return repaired
            retried = self._repair_contract(str(call["tool"]), {
                "event": "state_export_matched_no_values",
                "invocation_result": repair_feedback["invocation_result"],
                "paths": list(
                    itertools.islice(
                        iter_scalar_jsonpaths(result.transition.changes), 256
                    )
                ),
                "conclusion": (
                    "replace_with_explicit_named_exports_at_observed_scalar_paths"
                ),
            })
            if retried is not None and retried.revision > repaired.revision:
                self._enqueue_tool_revision(str(call["tool"]), retried.revision)
                return retried
            return repaired
        return contract

    def _contract_has_matching_state_exports(
        self,
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> bool:
        return any(
            select_export_values(row, result.transition.changes)
            for row in contract.exports
            if row.source == "state_changes"
        )

    def _state_export_repair_feedback(
        self,
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> dict[str, Any]:
        """Build deterministic structural feedback for a missing state export."""

        existing = [
            row for row in contract.exports if row.source == "state_changes"
        ]
        feedback = structural_feedback(
            example_expected="success",
            result=result,
            observed_outcome="success",
        )
        return {
            "event": (
                "state_export_matched_no_values"
                if existing
                else "state_observed_without_export"
            ),
            "state_shape": feedback["change_shape"],
            "invocation_result": feedback["invocation_result"],
            "paths": list(
                itertools.islice(
                    iter_scalar_jsonpaths(result.transition.changes), 256
                )
            ),
            "conclusion": (
                "declare_one_explicit_named_state_changes_export_per_reusable_field"
            ),
        }

    def _incomplete_binding(self, task_id: str) -> str | None:
        del task_id
        return None

    def _compatible_incomplete_contract(
        self, task: SearchTask
    ) -> tuple[ToolCallContract, int] | None:
        """Hook for schedulers finishing calls across contract revisions."""

        del task
        return None

    def _incomplete_call_for_task(
        self, task: SearchTask
    ) -> tuple[str, dict[str, Any]] | None:
        """Return the exact committed call still being postprocessed by a task."""

        binding_key = self._incomplete_binding(task.id)
        if binding_key is None:
            return None
        call = self.store.call_by_binding(binding_key)
        if call is None:
            # A worker can fail after the coordinator reserves a binding but
            # before it returns a call. There is no result or saved handle to
            # continue; release the old exact-binding/state-trial claim
            # before the mutable task combination is selected again.
            self._clear_incomplete_binding(task.id)
            return None
        mismatches = (
            (
                call.get("state_id") is not None
                and str(call["state_id"]) != task.state_id
            )
            or (
                call.get("tool") is not None
                and str(call["tool"]) != task.tool
            )
            or (
                call.get("contract_revision") is not None
                and int(call["contract_revision"]) != task.revision
            )
            or (
                call.get("template_index") is not None
                and int(call["template_index"]) != task.template_index
            )
        )
        complete_identity = all(
            call.get(key) is not None
            for key in (
                "state_id",
                "tool",
                "contract_revision",
                "template_index",
                "args",
            )
        )
        if complete_identity:
            mismatches = mismatches or stable_id("binding", [
                str(call["state_id"]),
                str(call["tool"]),
                int(call["contract_revision"]),
                int(call["template_index"]),
                dict(call["args"]),
            ]) != binding_key
        if mismatches:
            raise RuntimeError(
                f"incomplete binding {binding_key} does not match task {task.id}"
            )
        return binding_key, call

    def _mark_incomplete_binding(self, task_id: str, binding_key: str) -> None:
        del task_id, binding_key

    def _try_mark_incomplete_binding(
        self,
        task: SearchTask,
        binding_key: str,
        args: Mapping[str, Any],
    ) -> bool:
        del args
        self._mark_incomplete_binding(task.id, binding_key)
        return True

    def _clear_incomplete_binding(self, task_id: str) -> None:
        del task_id

    def _task_start_budget_available(self, task: SearchTask) -> bool:
        return self._budget_available(self._estimate_task_calls(task))

    def _prepare_tuple_requeue(self, task: SearchTask) -> None:
        self.store.update_task(task)

    def _yield_scored_continuation(self, task: SearchTask) -> bool:
        self._prepare_tuple_requeue(task)
        self._schedule_task(task)
        return True

    async def _postprocess_call_async(
        self,
        *,
        task: SearchTask,
        state: SearchState,
        call: Mapping[str, Any],
        result: InvocationResult,
        bindings: Mapping[str, Sequence[Observation]],
    ) -> bool:
        active = self.contracts.get(task.tool)
        saved_handle = str(call.get("saved_state_handle") or "")
        if active is None or call.get("outcome") != "success":
            if saved_handle:
                self._discard_unreferenced_state_handle(saved_handle)
            return True

        if (
            task.observation_call is not None
            and not task.anchor_search
            and self._first_success_call_by_tool.get(task.tool)
            != str(call["id"])
        ):
            if saved_handle:
                self._discard_unreferenced_state_handle(saved_handle)
            return True

        if self.stop_reason != "completed":
            if saved_handle:
                self._discard_unreferenced_state_handle(saved_handle)
                self._storage_budget_available()
            return True

        if (
            self.stop_reason == "completed"
            and active.revision == int(call["contract_revision"])
        ):
            active = self._resolve_state_export_contract(
                call=call,
                result=result,
                contract=active,
            )
            self._record_literal_baseline(call)

        selected_template_index = self._matching_selected_template_index(
            active, call
        )
        if selected_template_index is None:
            # The accepted repair changed the invocation template. Its new
            # revision is already enqueued; do not reinterpret this old call's
            # arguments, bindings, transition, or exports under that template.
            if saved_handle:
                self._discard_unreferenced_state_handle(saved_handle)
            return True

        trial_call = {
            **call,
            "contract_revision": active.revision,
            "template_index": selected_template_index,
        }

        # Ordinary successful calls publish outputs and child state before
        # their trials. A provider anchor is itself an input trial around an
        # already published successful call, so it adjudicates only that edge
        # and does not open a second search branch.
        if not task.anchor_search:
            post_handle = call.get("saved_state_handle")
            if result.transition.changed and not post_handle:
                return True
            self._create_state(
                state,
                trial_call,
                result,
                active,
                handle=str(post_handle or state.handle),
            )

        if (
            self.stop_reason == "completed"
            and active.revision == int(trial_call["contract_revision"])
        ):
            complete = await self._run_input_trials_async(
                state=state,
                call=trial_call,
                result=result,
                contract=active,
                template_index=selected_template_index,
                bindings=bindings,
            )
            if not complete:
                return False

        if self.stop_reason == "completed" and not task.anchor_search:
            complete = await self._run_state_trial_async(
                state=state,
                call=trial_call,
                contract=active,
            )
            if not complete:
                return False
        return True

    async def _run_task_async(
        self,
        task: SearchTask,
        *,
        combination_prepared: bool = False,
        prepared_combination: tuple[
            dict[str, Any], dict[str, tuple[Observation, ...]]
        ] | None = None,
    ) -> None:
        state = self.states[task.state_id]
        while True:
            if self._observation_task_is_pruned(task):
                self._discard_incomplete_task_state(task.id)
                task.completed = True
                self.observation_tasks_pruned_after_success += 1
                self.store.update_task(task)
                return
            contract = self.contracts.get(task.tool)
            incomplete = self._incomplete_call_for_task(task)
            compatible_incomplete = (
                contract is not None
                and incomplete is not None
                and self._matching_selected_template_index(
                    contract, incomplete[1]
                ) is not None
            )
            if contract is None or (
                contract.revision != task.revision
                and not compatible_incomplete
            ):
                self._discard_incomplete_task_state(task.id)
                task.completed = True
                self.store.update_task(task)
                return
            if incomplete is not None:
                # Observation domains can grow between contract epochs. Resume
                # the exact call selected before that growth; recomputing the
                # same rank can point at different arguments and orphan its
                # primary/state-trial reservation.
                combination_prepared = False
                prepared_combination = None
                binding_key, existing = incomplete
                args = dict(existing.get("args") or {})
                bindings = {
                    str(parameter): binding_observations(value)
                    for parameter, value in dict(
                        existing.get("bindings") or {}
                    ).items()
                }
            else:
                combination = (
                    prepared_combination
                    if combination_prepared else self._combination(task)
                )
                # A prepared value belongs only to the exact cursor selected
                # for this lease.  If a committed binding advances the cursor,
                # the next combination must be selected normally.
                combination_prepared = False
                prepared_combination = None
                if combination is None:
                    self._discard_incomplete_task_state(task.id)
                    task.completed = True
                    self.store.update_task(task)
                    return
                args, bindings = combination
                binding_key = stable_id("binding", [
                    state.id,
                    task.tool,
                    task.revision,
                    task.template_index,
                    args,
                ])
                existing = self.store.call_by_binding(binding_key)
            if existing is not None:
                result = InvocationResult.from_dict(existing["result"])
                if self._incomplete_binding(task.id) == binding_key:
                    if not await self._postprocess_call_async(
                        task=task,
                        state=state,
                        call=existing,
                        result=result,
                        bindings=bindings,
                    ):
                        self.store.update_task(task)
                        self._schedule_task(task, front=True)
                        return
                    self._clear_incomplete_binding(task.id)
                    if contract.revision != task.revision:
                        # New-revision frontiers own every remaining tuple.
                        # This old task exists only to finish the exact call
                        # that was committed before the template-stable repair.
                        task.completed = True
                        self.store.update_task(task)
                        return
                elif existing.get("outcome") == "success" and bindings:
                    active = self.contracts.get(task.tool)
                    if (
                        active is not None
                        and active.revision == task.revision
                        and not await self._run_input_trials_async(
                            state=state,
                            call=existing,
                            result=result,
                            contract=active,
                            template_index=task.template_index,
                            bindings=bindings,
                        )
                    ):
                        self.store.update_task(task)
                        self._schedule_task(task, front=True)
                        return
                was_semantic = self._task_semantic_score(task) is not None
                task.next_combination += 1
                self.duplicate_template_calls_reused += 1
                if task.anchor_search:
                    task.completed = True
                    self.store.update_task(task)
                    return
                # A scored lease must not skip a committed tuple and dispatch
                # the next tuple with the previous tuple's score. Head-only
                # yields once at 0 -> 1; tuple scoring yields at every scored
                # boundary so the concrete next binding is ranked normally.
                if was_semantic and self._yield_scored_continuation(task):
                    return
                # A reused success already created its output/state branch when
                # it was first invoked.  Only newly available producer
                # provenance needs projection into input-trial evidence.
                if existing.get("outcome") == "success" and bindings:
                    self.store.update_task(task)
                    self._schedule_task(task)
                    return
                continue
            if not self._task_start_budget_available(task):
                self.store.update_task(task)
                self._schedule_task(task, front=True)
                return
            break

        template = contract.templates[task.template_index]
        if not self._try_mark_incomplete_binding(task, binding_key, args):
            return
        invoked = await self._invoke_async(
            state=state,
            tool=task.tool,
            revision=task.revision,
            args=args,
            bindings=bindings,
            expected=template.expected,
            reason=("provider_anchor" if task.anchor_search else "template_search"),
            binding_key=binding_key,
            template_index=task.template_index,
            allow_repair=True,
            save_on_change=not task.anchor_search,
        )
        if invoked is None:
            self.store.update_task(task)
            if self.stop_reason == "completed":
                self._schedule_task(task, front=True)
            return
        call, result, _, _ = invoked
        if not await self._postprocess_call_async(
            task=task,
            state=state,
            call=call,
            result=result,
            bindings=bindings,
        ):
            self.store.update_task(task)
            self._schedule_task(task, front=True)
            return
        self._clear_incomplete_binding(task.id)
        task.next_combination += 1
        if task.anchor_search:
            task.completed = True
            self.store.update_task(task)
            return
        self.store.update_task(task)
        self._schedule_task(task)

    def _invoke(self, **kwargs: Any) -> tuple[dict[str, Any], InvocationResult, str] | None:
        """Compatibility wrapper for serial callers; discovery lives in async core."""

        invoked = asyncio.run(self._invoke_async(**kwargs))
        if invoked is None:
            return None
        call, result, outcome, _ = invoked
        return call, result, outcome

    def _run_input_trials(self, **kwargs: Any) -> bool:
        return asyncio.run(self._run_input_trials_async(**kwargs))

    def _run_state_trial(self, **kwargs: Any) -> None:
        asyncio.run(self._run_state_trial_async(**kwargs))

    def _run_task(self, task: SearchTask) -> None:
        asyncio.run(self._run_task_async(task))

    def run(self) -> dict[str, Any]:
        try:
            while (
                self.active_states
                and self.stop_reason == "completed"
                and self._budget_available(1)
            ):
                state_id = self.active_states.popleft()
                batch_calls = self.calls_used
                batch_tasks = 0
                try:
                    # The call quantum is checked only between atomic tasks;
                    # one task's trials may finish beyond the soft call cap.
                    while (
                        self.stop_reason == "completed"
                        and self._budget_available(1)
                        and batch_tasks < self.config.serial_state_batch_size
                        and self.calls_used - batch_calls
                        < self.config.serial_state_batch_size
                    ):
                        task_id = self._pop_task(state_id)
                        if task_id is None:
                            break
                        batch_tasks += 1
                        self.scheduled_task_ids.discard(task_id)
                        task = self.store.task(task_id)
                        if task is None or task.completed:
                            continue
                        self._run_task(task)
                finally:
                    # Keep the state marked active while its batch executes so
                    # a rescheduled task cannot append a duplicate. Rotate the
                    # state behind already-active peers at the boundary.
                    self.active_state_ids.discard(state_id)
                    self._activate_state(state_id)
            if self.active_states and self.stop_reason == "completed":
                self.stop_reason = "interrupted"
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            raise
        finally:
            self.store.set_meta("calls_used", self.calls_used)
            self.store.set_meta("elapsed_seconds", self._elapsed())
            self.store.set_meta(
                "observation_types_dropped", self.observation_types_dropped
            )
            self.store.set_meta(
                "duplicate_template_calls_reused",
                self.duplicate_template_calls_reused,
            )
            # Capture growth from the final coordinator writes and expose the
            # storage stop explicitly even when it coincides with another
            # terminal budget.  Persisting the observation itself is followed
            # by one final cheap refresh so reported usage includes that row.
            self._storage_budget_available(force=True)
            self.store.set_meta("storage_bytes", self._last_storage_bytes)
            self._storage_budget_available(force=True)
            self.store.set_meta(
                "budget_violations", sorted(self.budget_violations)
            )
        return self.graph_artifact()

    def _active_trial(self, row: Mapping[str, Any]) -> bool:
        source = self.contracts.get(str(row["source"]))
        target = self.contracts.get(str(row["target"]))
        if source is None or target is None:
            return False
        source_field = row.get("source_field_id")
        if source_field is None:
            # State trials establish a transition dependency rather than a
            # selected output field and remain empirical across contract-only
            # revisions.
            return True
        active_fields = {
            output_field_id(
                tool=source.tool,
                location=export.source,
                export_name=export.name,
                field_path=export.path,
                value_schema=export.value_schema,
            )
            for export in source.exports
            if export.source == "arguments" or export.propagate
        }
        return str(source_field) in active_fields

    def _trial_declared_source_path(self, row: Mapping[str, Any]) -> str:
        """Resolve a trial to its contract path, never a runtime mapping key."""

        source = self.contracts.get(str(row["source"]))
        source_field = row.get("source_field_id")
        if source is None or source_field is None:
            return ""
        for export in source.exports:
            field_id = output_field_id(
                tool=source.tool,
                location=export.source,
                export_name=export.name,
                field_path=export.path,
                value_schema=export.value_schema,
            )
            if field_id == str(source_field):
                return export.path
        return ""

    def graph_artifact(self) -> dict[str, Any]:
        # A contract-supplied value counts as independent only after that
        # literal has participated in a successful call without itself being
        # replaced by an upstream binding. Other parameters in the same call
        # may still use observations. A merely proposed but unusable literal
        # must not hide the observation that made the target runnable, while
        # omission remains orthogonal and never enters this set.
        input_trials = self.store.input_trials()
        empirical_literal_call_ids = {
            str(call_id)
            for row in input_trials
            if str(row.get("declared_mode")) == "literal"
            and not bool(row.get("required", True))
            for key in ("success_calls", "control_calls")
            for call_id in row.get(key) or ()
        }
        empirical_literal_schema_outcomes: dict[str, str | None] = {}
        constructible: set[tuple[str, str]] = set()
        for call in self.store.iter_calls():
            call_id = str(call.get("id") or "")
            if call_id in empirical_literal_call_ids:
                empirical_literal_schema_outcomes[call_id] = (
                    str(call["schema_outcome"])
                    if call.get("schema_outcome") is not None
                    else None
                )
            tool = str(call["tool"])
            contract = self.contracts.get(tool)
            if (
                contract is None
                or call.get("reason") != "template_search"
                or call.get("expected") != "success"
                or call.get("outcome") != "success"
                # Effective transport fallback is intentionally insufficient
                # here: caller constructibility is a verified contract fact,
                # so a modern call must have matched the success schema.  Old
                # checkpoints without schema_outcome or selected-template
                # provenance remain conservatively required instead of
                # silently inheriting fallback semantics.
                or call.get("schema_outcome") != "success"
            ):
                continue
            template_index = self._matching_selected_template_index(
                contract, call
            )
            if template_index is None:
                continue
            template = contract.templates[template_index]
            if template.expected != "success":
                continue
            args = dict(call.get("args") or {})
            bindings = dict(call.get("bindings") or {})
            constructible.update(
                (tool, name)
                for name, specification in template.arguments.items()
                if specification.kind == "literal"
                and name not in bindings
                and name in args
                and canonical_json(args[name])
                == canonical_json(json_safe(specification.value))
            )
        for row in input_trials:
            target = self.contracts.get(str(row["target"]))
            argument_constructible = (
                str(row.get("source_location")) == "arguments"
                and fields_share_specific_semantics(
                    str(row["source_export"]),
                    self._trial_declared_source_path(row),
                    str(row["target_parameter"]),
                )
            )
            success_calls = [
                str(call_id) for call_id in row.get("success_calls") or ()
            ]
            control_calls = [
                str(call_id) for call_id in row.get("control_calls") or ()
            ]
            empirical_literal_constructible = (
                str(row.get("declared_mode")) == "literal"
                and not bool(row.get("required", True))
                and bool(success_calls)
                and all(
                    empirical_literal_schema_outcomes.get(call_id) == "success"
                    for call_id in success_calls
                )
                and bool(control_calls)
                and any(
                    empirical_literal_schema_outcomes.get(call_id) == "success"
                    for call_id in control_calls
                )
                and all(
                    empirical_literal_schema_outcomes.get(call_id)
                    in {"success", "failure"}
                    for call_id in control_calls
                )
            )
            if (
                target is not None
                and (
                    # An argument export proves only the same public target
                    # field constructible.  Equal scalars from unrelated
                    # caller fields do not make this field independent.
                    argument_constructible
                    # A completed literal ablation is direct empirical proof
                    # and remains valid across target contract-only revisions;
                    # it does not depend on a historical template index.  Its
                    # success and literal-control calls must still be verified
                    # against the active outcome schema rather than transport
                    # fallback.
                    or empirical_literal_constructible
                )
            ):
                constructible.add((target.tool, str(row["target_parameter"])))

        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in input_trials:
            # Unbound argument exports are caller-constructible seeds and do
            # not establish tool-to-tool provenance. Bound forwarding never
            # reaches this branch: _observations preserves the original
            # response/state Observation for forwarded values.
            if str(row.get("source_location")) == "arguments":
                continue
            if str(row.get("source")) == str(row.get("target")):
                continue
            if not self._active_trial(row):
                continue
            key = (row["source"], row["target"], row["target_parameter"])
            grouped[key].append(row)
        edges: list[dict[str, Any]] = []
        for key, rows in sorted(grouped.items()):
            source, target, parameter = key
            required = (str(target), str(parameter)) not in constructible
            if not required:
                # A successful literal proves that the upstream value is not
                # necessary.  Keep such an optional value-flow edge only when
                # the public export/path and target field name the same
                # entity; equal structural scalars alone are not evidence.
                rows = [
                    row for row in rows
                    if fields_share_specific_semantics(
                        str(row["source_export"]),
                        self._trial_declared_source_path(row),
                        str(parameter),
                    )
                ]
                if not rows:
                    continue
            source_observations = sorted(
                {
                    (
                        str(row["source_location"]),
                        str(row["source_export"]),
                        str(row["source_path"]),
                    )
                    for row in rows
                }
            )
            edges.append({
                "id": stable_id("edge", ["input", *key]),
                "source": source,
                "target": target,
                "kind": "input",
                "dependency_type": "value_flow",
                "required": required,
                "requirement_basis": sorted({
                    "no_successful_contract_literal"
                    for row in rows if required
                }),
                "evidence_argument_modes": sorted({
                    str(row["declared_mode"]) for row in rows
                }),
                "source_observations": [
                    {"location": location, "export": export, "path": path}
                    for location, export, path in source_observations
                ],
                "target_parameter": parameter,
                "input_parameters": [parameter],
                "evidence_ids": sorted(row["id"] for row in rows),
                "success_calls": sorted({
                    call_id for row in rows for call_id in row["success_calls"]
                }),
                "control_calls": sorted({
                    call_id for row in rows for call_id in row["control_calls"]
                }),
            })
        state_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in self.store.state_trials():
            if str(row.get("source")) == str(row.get("target")):
                continue
            if not self._active_trial(row):
                continue
            key = (
                row["source"], row["target"],
                tuple(map(str, row["state_changes"])),
            )
            state_grouped[key].append(row)
        for key, rows in sorted(state_grouped.items()):
            source, target, state_changes = key
            edges.append({
                "id": stable_id("edge", ["state", *key]),
                "source": source,
                "target": target,
                "kind": "state",
                "dependency_type": "state_effect",
                "required": True,
                "requirement_basis": ["state_ablation"],
                "state_changes": list(state_changes),
                "evidence_ids": sorted(str(row["id"]) for row in rows),
                "evidence_calls": sorted({
                    str(call_id)
                    for row in rows
                    for call_id in row["evidence_calls"]
                }),
            })
        return {
            "version": GRAPH_VERSION,
            "nodes": {
                tool: {
                    "tool": tool,
                    "parameters": [
                        parameter.to_dict()
                        for parameter in self.definitions[tool].parameters
                    ],
                    "contract_revision": contract.revision,
                    "templates": len(contract.templates),
                    "exports": [row.name for row in contract.exports],
                    "propagation_exports": [
                        row.name
                        for row in contract.exports
                        if row.source != "arguments" and row.propagate
                    ],
                }
                for tool, contract in sorted(self.contracts.manifest.contracts.items())
            },
            "edges": sorted(edges, key=lambda row: row["id"]),
            "tool_projection": {
                "edges": sorted(edges, key=lambda row: row["id"]),
            },
            "build": {
                "usage": {
                    "calls": self.calls_used,
                    "states": len(self.states),
                    "storage_bytes": self._last_storage_bytes,
                    "budget_violations": sorted(self.budget_violations),
                    "elapsed_seconds": round(self._elapsed(), 3),
                    "llm_calls": sum(
                        self.contracts.llm_calls(tool)
                        for tool in self.contracts.manifest.audit
                    ),
                    "contract_repair_llm_calls": max(
                        0,
                        self.contracts.total_llm_calls()
                        - self.initial_llm_calls,
                    ),
                    "contract_repair_requests": self.contract_repair_requests,
                    "contract_repairs_accepted": self.contract_repairs_accepted,
                    "contract_repairs_rejected": self.contract_repairs_rejected,
                    "contract_repairs_budget_rejected": (
                        self.contract_repairs_budget_rejected
                    ),
                },
                "budget": self.config.budget.__dict__,
                "budget_violations": sorted(self.budget_violations),
                "trial_repetitions": self.config.trial_repetitions,
                "serial_state_batch_size": self.config.serial_state_batch_size,
                "semantic_priority": "current_tuple_max_field_cosine",
                "observation_search_policy": (
                    "richest_success_then_one_output_field_per_provider_tool_input_trial"
                ),
                "observation_tasks_pruned_after_success": (
                    self.observation_tasks_pruned_after_success
                ),
                "stop_reason": self.stop_reason,
                "runtime": json_safe(self.runtime.source_metadata()),
                "candidate_policy": (
                    "source_and_best_field_heads_rank_sum_all_provenance_v6"
                ),
                "coalesced_value_provenance_policy": (
                    "highest_declared_field_score_all_provenances_retained_v2"
                    if self.field_priority is not None
                    else "all_provenances_without_field_priority_v1"
                ),
                "optional_input_evidence_policy": (
                    "specific_export_or_declared_path_token_overlap_v1"
                ),
                "contract_template_policy": {
                    "version": MINIMAL_SUCCESS_TEMPLATE_POLICY,
                    "max_added_success_templates_per_tool": 1,
                    "templates_added_this_run": (
                        self.minimal_success_templates_added
                    ),
                },
                "observation_policy": (
                    "global_first_value_per_explicit_export_or_wildcard_match_path"
                ),
                "deduplication": {
                    "observation_types_admitted": len(
                        self.admitted_observation_types
                    ),
                    "observation_values_dropped": self.observation_types_dropped,
                    "template_calls_reused": self.duplicate_template_calls_reused,
                },
                "scheduler_policy": self.scheduler_policy,
                "propagation_order": (
                    "semantic"
                    if self.scheduling_priority is not None else "legacy"
                ),
                "field_priority": (
                    {
                        **dict(self.field_priority.build_metadata()),
                        "fields_added_this_run": self.embedding_fields_added,
                    }
                    if self.field_priority is not None
                    else None
                ),
                "scheduling_priority": (
                    {"source": "field_priority"}
                    if self.scheduling_priority is self.field_priority
                    else {
                        "source": "separate",
                        **dict(self.scheduling_priority.build_metadata()),
                        "fields_added_this_run": (
                            self.scheduling_embedding_fields_added
                        ),
                    }
                    if self.scheduling_priority is not None
                    else None
                ),
                "imports_task_or_gold_data": False,
            },
        }

    def write_outputs(
        self,
        graph: Mapping[str, Any],
        *,
        graph_path: str | Path,
        evidence_path: str | Path,
    ) -> None:
        graph_output = Path(graph_path)
        graph_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = graph_output.with_suffix(graph_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, graph_output)
        evidence_output = Path(evidence_path)
        evidence_output.parent.mkdir(parents=True, exist_ok=True)
        with evidence_output.open("w", encoding="utf-8") as handle:
            for call in self.store.iter_calls():
                handle.write(canonical_json({
                    "version": EVIDENCE_VERSION,
                    "record_type": "call",
                    **call,
                }) + "\n")
            for kind, rows in (
                ("input_trial", self.store.input_trials()),
                ("state_trial", self.store.state_trials()),
            ):
                for row in rows:
                    handle.write(canonical_json({
                        "version": EVIDENCE_VERSION,
                        "record_type": kind,
                        **row,
                    }) + "\n")
