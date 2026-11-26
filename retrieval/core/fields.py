"""Independent semantic field index for propagation ordering.

Only field descriptors are embedded. Runtime field values never leave the
dependency builder and are used solely by the existing schema compatibility
checks and tool invocations.
"""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Protocol, Sequence

import numpy as np
import requests

from retrieval.core.graph import (
    FieldPriorityError,
    Observation,
    ToolDefinition,
    ToolParameter,
    output_field_id,
    stable_id,
)
from retrieval.core.contracts import ToolCallContract, ToolExport, canonical_json


FIELD_STORE_VERSION = "qwen3_field_embeddings"
FIELD_DESCRIPTOR_VERSIONS = {
    "context": "propagation_field_descriptor_v1",
    "field": "propagation_field_semantics_descriptor_v3",
    "field_tool": "propagation_field_tool_descriptor_v1",
}
DEFAULT_FIELD_DESCRIPTOR_MODE = "context"
FIELD_DESCRIPTOR_VERSION = FIELD_DESCRIPTOR_VERSIONS[DEFAULT_FIELD_DESCRIPTOR_MODE]
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8101/v1"


@dataclass(frozen=True)
class FieldDescriptor:
    binding_id: str
    role: str
    tool: str
    field_name: str
    descriptor: Mapping[str, Any]
    text: str
    contract_revision: int | None = None
    source: str | None = None
    path: str | None = None


def _operation_descriptor(definition: ToolDefinition) -> dict[str, Any]:
    public = dict(definition.public_metadata or {})
    return {
        "tool": definition.tool,
        "method": public.get("method"),
        "path": public.get("path"),
    }


def _descriptor_text(payload: Mapping[str, Any], *, mode: str) -> str:
    if mode in {"field", "field_tool"}:
        lines = [
            "Tool field for semantic value propagation.",
            f"Field name: {payload['field_name']}",
        ]
        if payload.get("description"):
            lines.append(f"Description: {payload['description']}")
        lines.append(f"JSON schema: {canonical_json(payload.get('schema') or {})}")
        if mode == "field_tool":
            lines.append(f"Tool: {payload['operation'].get('tool') or ''}")
        return "\n".join(lines)
    operation = dict(payload["operation"])
    lines = [
        "Tool field for semantic value propagation.",
        f"Role: {payload['role']}",
        f"Tool: {operation.get('tool') or ''}",
        f"Method: {operation.get('method') or ''}",
        f"Operation path: {operation.get('path') or ''}",
        f"Field: {payload['field_name']}",
    ]
    if payload.get("description"):
        lines.append(f"Description: {payload['description']}")
    if payload.get("required") is not None:
        lines.append(f"Required input: {str(bool(payload['required'])).lower()}")
    if payload.get("source"):
        lines.append(f"Observable source: {payload['source']}")
    if payload.get("json_path"):
        lines.append(f"JSON path: {payload['json_path']}")
    lines.append(f"JSON schema: {canonical_json(payload.get('schema') or {})}")
    return "\n".join(lines)


def output_binding_id(field_id: str) -> str:
    """Return the stable identity used to score one contract output field."""

    return stable_id("field_binding", ["output", field_id])


def input_descriptor(
    definition: ToolDefinition,
    parameter: ToolParameter,
    *,
    mode: str = DEFAULT_FIELD_DESCRIPTOR_MODE,
) -> FieldDescriptor:
    if mode not in FIELD_DESCRIPTOR_VERSIONS:
        raise ValueError(f"unsupported field descriptor mode: {mode}")
    public_parameters = {
        str(row.get("name")): row
        for row in (
            dict(definition.public_metadata or {}).get("parameters") or ()
        )
    }
    public = public_parameters.get(parameter.name) or {}
    payload = {
        "version": FIELD_DESCRIPTOR_VERSIONS[mode],
        "field_name": parameter.name,
        "description": str(public.get("description") or ""),
        "schema": dict(parameter.schema),
    }
    if mode == "field_tool":
        payload["operation"] = _operation_descriptor(definition)
    elif mode == "context":
        payload.update({
            "role": "input",
            "operation": _operation_descriptor(definition),
            "required": parameter.required,
        })
    text = _descriptor_text(payload, mode=mode)
    return FieldDescriptor(
        binding_id=stable_id("field_binding", ["input", definition.tool, parameter.name]),
        role="input",
        tool=definition.tool,
        field_name=parameter.name,
        descriptor=payload,
        text=text,
    )


def export_descriptor(
    definition: ToolDefinition,
    contract: ToolCallContract,
    export: ToolExport,
    *,
    mode: str = DEFAULT_FIELD_DESCRIPTOR_MODE,
) -> FieldDescriptor:
    if mode not in FIELD_DESCRIPTOR_VERSIONS:
        raise ValueError(f"unsupported field descriptor mode: {mode}")
    description = ""
    if export.source == "arguments":
        public_parameters = {
            str(row.get("name")): row
            for row in (
                dict(definition.public_metadata or {}).get("parameters") or ()
            )
        }
        argument_name = export.path.removeprefix("$.")
        description = str(
            (public_parameters.get(argument_name) or {}).get("description") or ""
        )
    payload = {
        "version": FIELD_DESCRIPTOR_VERSIONS[mode],
        "field_name": export.name,
        "description": description,
        "schema": dict(export.value_schema),
    }
    if mode == "field_tool":
        payload["operation"] = _operation_descriptor(definition)
    elif mode == "context":
        payload.update({
            "role": "output",
            "operation": _operation_descriptor(definition),
            "source": export.source,
            "json_path": export.path,
        })
    text = _descriptor_text(payload, mode=mode)
    field_id = output_field_id(
        tool=definition.tool,
        location=export.source,
        export_name=export.name,
        field_path=export.path,
        value_schema=export.value_schema,
    )
    return FieldDescriptor(
        binding_id=output_binding_id(field_id),
        role="output",
        tool=definition.tool,
        field_name=export.name,
        contract_revision=None,
        source=export.source,
        path=export.path,
        descriptor=payload,
        text=text,
    )


def contract_descriptors(
    definition: ToolDefinition,
    contract: ToolCallContract,
    *,
    mode: str = DEFAULT_FIELD_DESCRIPTOR_MODE,
) -> tuple[FieldDescriptor, ...]:
    return tuple(
        [
            input_descriptor(definition, row, mode=mode)
            for row in definition.parameters
        ]
        + [
            export_descriptor(definition, contract, row, mode=mode)
            for row in contract.exports
            # A recursive wildcard is not one semantic field.  It remains a
            # propagation fallback but never receives a shared field vector.
            if row.path != "$..[*]"
            and (row.source == "arguments" or row.propagate)
        ]
    )


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: np.ndarray
    requests: int
    prompt_tokens: int


class TextEmbedder(Protocol):
    model: str
    dimensions: int
    base_url: str

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class QwenVllmTextEmbedder:
    """Qwen3 embeddings through a local vLLM server."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        batch_size: int = 16,
        max_concurrent: int = 20,
    ) -> None:
        if model != DEFAULT_EMBEDDING_MODEL:
            raise FieldPriorityError(
                f"field propagation requires {DEFAULT_EMBEDDING_MODEL}"
            )
        if dimensions != DEFAULT_EMBEDDING_DIMENSIONS:
            raise FieldPriorityError(
                f"field propagation requires {DEFAULT_EMBEDDING_DIMENSIONS} dimensions"
            )
        if batch_size < 1:
            raise ValueError("Qwen embedding batch_size must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_concurrent = max(1, max_concurrent)

    def _embed_batch(self, batch: Sequence[str]) -> tuple[np.ndarray, int]:
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json={
                        "model": self.model,
                        "input": list(batch),
                    },
                    timeout=120,
                )
                if response.status_code >= 400:
                    error: Mapping[str, Any] = {}
                    try:
                        payload = response.json()
                        if isinstance(payload, Mapping):
                            candidate = payload.get("error") or payload
                            if isinstance(candidate, Mapping):
                                error = candidate
                    except Exception:
                        pass
                    detail = ": ".join(
                        value for value in (
                            str(error.get("code") or "").strip(),
                            str(error.get("message") or "").strip(),
                        ) if value
                    ) or f"HTTP {response.status_code}"
                    if response.status_code not in {408, 409, 429}:
                        raise FieldPriorityError(
                            f"Qwen field embedding rejected: {detail}"
                        )
                response.raise_for_status()
                payload = response.json()
                rows = sorted(payload["data"], key=lambda row: int(row["index"]))
                vectors = np.asarray(
                    [row["embedding"] for row in rows], dtype=np.float32
                )
                if vectors.shape != (len(batch), self.dimensions):
                    raise ValueError(
                        "unexpected Qwen embedding shape: "
                        f"{vectors.shape}, expected {(len(batch), self.dimensions)}"
                    )
                if not np.isfinite(vectors).all():
                    raise ValueError("Qwen returned non-finite embeddings")
                usage = payload.get("usage") or {}
                return vectors, int(
                    usage.get("prompt_tokens") or usage.get("total_tokens") or 0
                )
            except FieldPriorityError:
                raise
            except Exception as exc:  # transient network/server errors are retried
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise FieldPriorityError(
            f"Qwen field embedding failed after 3 attempts: {last_error}"
        ) from last_error

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        values = list(texts)
        if not values:
            return EmbeddingBatch(
                np.empty((0, self.dimensions), dtype=np.float32), 0, 0
            )
        batches = [
            values[index:index + self.batch_size]
            for index in range(0, len(values), self.batch_size)
        ]
        results: list[tuple[np.ndarray, int] | None] = [None] * len(batches)
        # Validate model access, request shape, and vector shape once before
        # fanning out the remaining local requests.
        results[0] = self._embed_batch(batches[0])
        remaining = batches[1:]
        if remaining:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_concurrent, len(remaining))
            ) as pool:
                futures = {
                    pool.submit(self._embed_batch, batch): index
                    for index, batch in enumerate(remaining, start=1)
                }
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
        complete = [row for row in results if row is not None]
        vectors = np.concatenate([row[0] for row in complete], axis=0)
        return EmbeddingBatch(
            vectors=vectors,
            requests=len(batches),
            prompt_tokens=sum(row[1] for row in complete),
        )


class SQLiteFieldEmbeddingStore:
    """Run-local field vectors, keyed directly by their binding identity."""

    def __init__(
        self,
        path: str | Path,
        *,
        create: bool = False,
        reset: bool = False,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        base_url: str | None = None,
        descriptor_mode: str | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only and (create or reset):
            raise FieldPriorityError(
                "a read-only field embedding store cannot be created or reset"
            )
        if reset:
            self.path.unlink(missing_ok=True)
        if not self.path.exists() and not create:
            raise FieldPriorityError(
                f"field embedding store does not exist: {self.path}"
            )
        if self.read_only:
            self.db = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro&immutable=1",
                uri=True,
            )
            self.db.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        if not self.read_only:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bindings (
                    binding_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    contract_revision INTEGER,
                    source TEXT,
                    path TEXT,
                    descriptor_text TEXT NOT NULL,
                    vector BLOB NOT NULL
                );
            """)
            self.db.commit()
        effective_base_url = (
            base_url
            or self.get_meta("base_url")
            or DEFAULT_EMBEDDING_BASE_URL
        )
        existing_descriptor_version = self.get_meta("descriptor_version")
        if descriptor_mode is None:
            descriptor_mode = next(
                (
                    mode for mode, version in FIELD_DESCRIPTOR_VERSIONS.items()
                    if version == existing_descriptor_version
                ),
                DEFAULT_FIELD_DESCRIPTOR_MODE,
            )
        if descriptor_mode not in FIELD_DESCRIPTOR_VERSIONS:
            self.db.close()
            raise FieldPriorityError(
                f"unsupported field descriptor mode: {descriptor_mode}"
            )
        expected = {
            "store_version": FIELD_STORE_VERSION,
            "descriptor_version": FIELD_DESCRIPTOR_VERSIONS[descriptor_mode],
            "model": model,
            "dimensions": dimensions,
            "base_url": str(effective_base_url).rstrip("/"),
        }
        existing_version = self.get_meta("store_version")
        if existing_version is None:
            if not create:
                self.db.close()
                raise FieldPriorityError(
                    f"field embedding store is uninitialized: {self.path}"
                )
            for key, value in expected.items():
                self.set_meta(key, value)
            self.set_meta("embedding_requests", 0)
            self.set_meta("embedding_prompt_tokens", 0)
        else:
            mismatches = {
                key: (self.get_meta(key), value)
                for key, value in expected.items()
                if self.get_meta(key) != value
            }
            if mismatches:
                self.db.close()
                raise FieldPriorityError(
                    "field embedding store metadata mismatch: "
                    f"{canonical_json(mismatches)}"
                )
        self.model = model
        self.dimensions = dimensions
        self.base_url = str(effective_base_url).rstrip("/")
        self.descriptor_mode = descriptor_mode
        self._vectors_by_text: dict[str, np.ndarray] | None = None

    def close(self) -> None:
        if not self.read_only:
            self.db.commit()
        self.db.close()

    def get_meta(self, key: str) -> Any:
        row = self.db.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value: Any) -> None:
        if self.read_only:
            raise FieldPriorityError("field embedding store is read-only")
        self.db.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            (key, canonical_json(value)),
        )
        self.db.commit()

    def require_complete(self) -> None:
        if self.get_meta("initial_complete") is not True:
            raise FieldPriorityError(
                "field embedding store has no complete initial manifest"
            )

    def missing_binding_ids(
        self, descriptors: Sequence[FieldDescriptor]
    ) -> list[str]:
        expected = {row.binding_id for row in descriptors}
        if not expected:
            return []
        placeholders = ",".join("?" for _ in expected)
        present = {
            str(row[0])
            for row in self.db.execute(
                f"SELECT binding_id FROM bindings WHERE binding_id IN ({placeholders})",
                tuple(sorted(expected)),
            )
        }
        return sorted(expected - present)

    def vectors_by_text(
        self, texts: Collection[str]
    ) -> dict[str, np.ndarray]:
        """Reuse the exact vector already assigned to identical field text."""

        wanted = set(map(str, texts))
        if not wanted:
            return {}
        if self._vectors_by_text is None:
            self._vectors_by_text = {}
            for text, raw in self.db.execute(
                "SELECT descriptor_text, vector FROM bindings"
            ):
                key = str(text)
                if key in self._vectors_by_text:
                    continue
                vector = np.frombuffer(raw, dtype="<f4").astype(
                    np.float32, copy=True
                )
                if (
                    vector.shape != (self.dimensions,)
                    or not np.isfinite(vector).all()
                ):
                    raise FieldPriorityError(
                        "corrupt field vector for descriptor text: "
                        f"{key[:80]}"
                    )
                self._vectors_by_text[key] = vector
        return {
            text: self._vectors_by_text[text]
            for text in wanted
            if text in self._vectors_by_text
        }

    def put(
        self,
        descriptors: Sequence[FieldDescriptor],
        vectors: Mapping[str, np.ndarray],
        *,
        embedding_requests: int = 0,
        embedding_prompt_tokens: int = 0,
    ) -> None:
        if self.read_only:
            raise FieldPriorityError("field embedding store is read-only")
        with self.db:
            for row in sorted(descriptors, key=lambda item: item.binding_id):
                if self.db.execute(
                    "SELECT 1 FROM bindings WHERE binding_id=?",
                    (row.binding_id,),
                ).fetchone() is not None:
                    continue
                vector = np.asarray(vectors.get(row.binding_id), dtype=np.float32)
                if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
                    raise FieldPriorityError(
                        f"missing or invalid vector for binding {row.binding_id}"
                    )
                self.db.execute(
                    "INSERT INTO bindings VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        row.binding_id,
                        row.role,
                        row.tool,
                        row.field_name,
                        row.contract_revision,
                        row.source,
                        row.path,
                        row.text,
                        vector.astype("<f4", copy=False).tobytes(),
                    ),
                )
                if self._vectors_by_text is not None:
                    self._vectors_by_text.setdefault(row.text, vector.copy())
            for key, increment in (
                ("embedding_requests", embedding_requests),
                ("embedding_prompt_tokens", embedding_prompt_tokens),
            ):
                current = int(self.get_meta(key) or 0)
                self.db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                    (key, canonical_json(current + int(increment))),
                )

    def finalize_initial(self) -> None:
        self.set_meta("initial_complete", True)

    def binding_vectors(
        self, *, tools: Collection[str] | None = None
    ) -> dict[str, tuple[np.ndarray, str]]:
        parameters: tuple[str, ...] = ()
        where = ""
        if tools is not None:
            parameters = tuple(sorted(set(map(str, tools))))
            if not parameters:
                return {}
            placeholders = ",".join("?" for _ in parameters)
            where = f"WHERE b.tool IN ({placeholders})"
        rows: dict[str, tuple[np.ndarray, str]] = {}
        for binding_id, role, raw in self.db.execute(f"""
            SELECT binding_id, role, vector
            FROM bindings AS b
            {where}
        """, parameters):
            vector = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
                raise FieldPriorityError(f"corrupt field vector: {binding_id}")
            rows[str(binding_id)] = (vector, str(role))
        return rows

    def counts(self) -> dict[str, int]:
        bindings = int(
            self.db.execute("SELECT count(*) FROM bindings").fetchone()[0]
        )
        descriptors = int(
            self.db.execute(
                "SELECT count(DISTINCT descriptor_text) FROM bindings"
            ).fetchone()[0]
        )
        return {"descriptors": descriptors, "bindings": bindings}


class SQLiteFieldPriorityIndex:
    def __init__(
        self,
        store: SQLiteFieldEmbeddingStore,
        definitions: Sequence[ToolDefinition],
        *,
        embedder: TextEmbedder | None = None,
        require_complete: bool = True,
    ) -> None:
        if require_complete:
            store.require_complete()
        self.store = store
        self.definitions = {row.tool: row for row in definitions}
        self._active_tools = frozenset(self.definitions)
        self.embedder = embedder
        if embedder is not None and (
            embedder.model != store.model
            or embedder.dimensions != store.dimensions
            or embedder.base_url.rstrip("/") != store.base_url
        ):
            raise FieldPriorityError("field embedder does not match its store")
        self._vectors: dict[str, np.ndarray] = {}
        self._output_rows: dict[str, int] = {}
        self._input_columns: dict[str, int] = {}
        self._anonymous_output_bindings: set[str] = set()
        self._score_source_rows: dict[str, int | None] = {}
        self._score_target_columns: dict[tuple[str, str], int | None] = {}
        self._pair_scores = np.empty((0, 0), dtype=np.float32)
        self._reload()

    def _reload(self) -> None:
        rows = self.store.binding_vectors(tools=self._active_tools)
        self._vectors = {}
        roles: dict[str, str] = {}
        for binding_id, (vector, role) in rows.items():
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0.0:
                raise FieldPriorityError(f"zero-norm field vector: {binding_id}")
            self._vectors[binding_id] = vector / norm
            roles[binding_id] = role
        output_ids = sorted(
            binding_id for binding_id, role in roles.items() if role == "output"
        )
        input_ids = sorted(
            binding_id for binding_id, role in roles.items() if role == "input"
        )
        self._output_rows = {
            binding_id: index for index, binding_id in enumerate(output_ids)
        }
        self._input_columns = {
            binding_id: index for index, binding_id in enumerate(input_ids)
        }
        if self._active_tools:
            placeholders = ",".join("?" for _ in self._active_tools)
            self._anonymous_output_bindings = {
                str(row[0])
                for row in self.store.db.execute(
                    "SELECT binding_id FROM bindings "
                    "WHERE role='output' AND path='$..[*]' "
                    f"AND tool IN ({placeholders})",
                    tuple(sorted(self._active_tools)),
                )
            }
        else:
            self._anonymous_output_bindings = set()
        self._score_source_rows.clear()
        self._score_target_columns.clear()
        if output_ids and input_ids:
            output_matrix = np.stack(
                [self._vectors[binding_id] for binding_id in output_ids]
            )
            input_matrix = np.stack(
                [self._vectors[binding_id] for binding_id in input_ids]
            )
            self._pair_scores = output_matrix @ input_matrix.T
        else:
            self._pair_scores = np.empty(
                (len(output_ids), len(input_ids)), dtype=np.float32
            )

    def _ensure_descriptors(
        self, descriptors: Sequence[FieldDescriptor]
    ) -> int:
        missing = self.store.missing_binding_ids(descriptors)
        if not missing:
            return 0
        by_id = {row.binding_id: row for row in descriptors}
        ordered = [by_id[value] for value in missing]
        rows_by_text: dict[str, list[FieldDescriptor]] = {}
        for row in ordered:
            rows_by_text.setdefault(row.text, []).append(row)
        vectors_by_text = self.store.vectors_by_text(rows_by_text)
        new_texts = sorted(set(rows_by_text) - set(vectors_by_text))
        requests = 0
        prompt_tokens = 0
        if new_texts:
            if self.embedder is None:
                raise FieldPriorityError(
                    f"{len(new_texts)} field descriptors are missing and no Qwen "
                    "embedder is configured"
                )
            batch = self.embedder.embed(new_texts)
            if batch.vectors.shape != (len(new_texts), self.store.dimensions):
                raise FieldPriorityError("field embedder returned an invalid matrix")
            vectors_by_text.update({
                text: batch.vectors[index]
                for index, text in enumerate(new_texts)
            })
            requests = batch.requests
            prompt_tokens = batch.prompt_tokens
        vectors = {
            row.binding_id: vectors_by_text[row.text]
            for row in ordered
        }
        self.store.put(
            ordered,
            vectors,
            embedding_requests=requests,
            embedding_prompt_tokens=prompt_tokens,
        )
        self._reload()
        return len(new_texts)

    def ensure_contract(
        self, definition: ToolDefinition, contract: ToolCallContract
    ) -> int:
        return self._ensure_descriptors(contract_descriptors(
            definition, contract, mode=self.store.descriptor_mode
        ))

    def ensure_manifest(
        self, contracts: Mapping[str, ToolCallContract]
    ) -> int:
        descriptors = tuple(
            descriptor
            for tool, contract in sorted(contracts.items())
            for descriptor in contract_descriptors(
                self.definitions[tool], contract, mode=self.store.descriptor_mode
            )
        )
        return self._ensure_descriptors(descriptors)

    def score(
        self,
        observation: Observation,
        *,
        target_tool: str,
        target_parameter: ToolParameter,
    ) -> float | None:
        if observation.field_id not in self._score_source_rows:
            source_binding = output_binding_id(observation.field_id)
            # A recursive wildcard is not one semantic field.
            self._score_source_rows[observation.field_id] = (
                None
                if source_binding in self._anonymous_output_bindings
                else self._output_rows.get(source_binding)
            )
        target_key = (target_tool, target_parameter.name)
        if target_key not in self._score_target_columns:
            target_binding = stable_id(
                "field_binding",
                ["input", target_tool, target_parameter.name],
            )
            self._score_target_columns[target_key] = self._input_columns.get(
                target_binding
            )
        source_row = self._score_source_rows[observation.field_id]
        target_column = self._score_target_columns[target_key]
        if source_row is None or target_column is None:
            return None
        return float(self._pair_scores[source_row, target_column])

    def build_metadata(self) -> Mapping[str, Any]:
        return {
            "strategy": "embedding",
            "ordering": "cosine",
            "store_version": FIELD_STORE_VERSION,
            "descriptor_version": FIELD_DESCRIPTOR_VERSIONS[
                self.store.descriptor_mode
            ],
            "descriptor_mode": self.store.descriptor_mode,
            "path": str(self.store.path),
            "model": self.store.model,
            "dimensions": self.store.dimensions,
            "base_url": self.store.base_url,
            **self.store.counts(),
            "embedding_requests": int(
                self.store.get_meta("embedding_requests") or 0
            ),
            "embedding_prompt_tokens": int(
                self.store.get_meta("embedding_prompt_tokens") or 0
            ),
            "cached_pair_scores": int(self._pair_scores.size),
        }
