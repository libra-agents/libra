"""Common parallel graph builder for runtimes with one clean root state."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from retrieval.core.fields import (
    QwenVllmTextEmbedder,
    SQLiteFieldEmbeddingStore,
    SQLiteFieldPriorityIndex,
)
from retrieval.core.parallel import (
    ParallelBuildConfig,
    ParallelDependencyGraphBuilder,
)
from retrieval.core.graph import (
    BuildConfig,
    InvocationResult,
    SearchBudget,
    SQLiteBuildStore,
    StateSnapshot,
    ToolDefinition,
    TransitionDelta,
    classify_outcome_schemas,
)
from retrieval.core.contracts import (
    ContractAdapter,
    ContractGenerator,
    DEFAULT_FEEDBACK_POLICY,
    DeepSeekFlash,
    MINIMAL_SUCCESS_TEMPLATE_POLICY,
    OnlineContractSet,
    load_active_contract_manifest,
)


class ToolInvoker(Protocol):
    def __call__(self, tool: str, args: Mapping[str, Any]) -> InvocationResult:
        ...


def unchanged_result(
    status: str,
    response: Any,
    machine_status: Mapping[str, Any],
) -> InvocationResult:
    return InvocationResult(
        status=status,
        response=response,
        machine_status=machine_status,
        transition=TransitionDelta("stateless-root", "stateless-root", {}),
    )


def _invoke_batch(
    invoker: ToolInvoker,
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        try:
            result = invoker(
                str(request["tool"]), dict(request.get("args") or {})
            )
            outcome = None
            if "success_schema" in request and "failure_schema" in request:
                outcome = classify_outcome_schemas(
                    dict(request["success_schema"]),
                    dict(request["failure_schema"]),
                    result,
                )
            rows.append({
                "ok": True,
                "result": result.to_dict(),
                "outcome": outcome,
            })
        except Exception as exc:
            rows.append({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            })
    return rows, os.getpid()


class BatchInvocationPool:
    """Execute coordinator-selected batches in persistent workers."""

    def __init__(
        self,
        invoker: ToolInvoker,
        *,
        workers: int,
        processes: bool,
    ) -> None:
        self.invoker = invoker
        if processes:
            self.executor: concurrent.futures.Executor = (
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            )
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
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
        loop = asyncio.get_running_loop()
        raw_rows, worker = await loop.run_in_executor(
            self.executor,
            _invoke_batch,
            self.invoker,
            [dict(request) for request in requests],
        )
        rows = []
        for raw in raw_rows:
            if raw["ok"]:
                rows.append((
                    InvocationResult.from_dict(raw["result"]),
                    None,
                    None,
                    raw.get("outcome"),
                ))
            else:
                rows.append((None, None, raw, None))
        return rows, worker

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
        del state_handle, state_digest, save_state_id
        rows, worker = await self.invoke_batch(
            [{"tool": tool, "args": dict(args)}],
            avoid_worker=avoid_worker,
        )
        result, _, error, _ = rows[0]
        if result is None:
            raise RuntimeError(str(error))
        return result, None, worker

    async def invoke(
        self,
        *,
        state_handle: str,
        state_digest: str | None = None,
        tool: str,
        args: Mapping[str, Any],
        save_state_id: str | None,
    ) -> tuple[InvocationResult, str | None]:
        result, saved, _ = await self.invoke_with_worker(
            state_handle=state_handle,
            state_digest=state_digest,
            tool=tool,
            args=args,
            save_state_id=save_state_id,
        )
        return result, saved

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class StatelessRootRuntime:
    runtime_id: str
    metadata: Mapping[str, Any]

    def save_state(self, state_id: str) -> str:
        del state_id
        return "stateless-root"

    def load_state(self, state_handle: str) -> None:
        if state_handle != "stateless-root":
            raise ValueError(f"unknown stateless handle: {state_handle}")

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot("stateless-root", {})

    def invoke(self, tool: str, args: Mapping[str, Any]) -> InvocationResult:
        raise RuntimeError(
            f"{tool} was sent to the coordinator instead of a worker"
        )

    def source_metadata(self) -> Mapping[str, Any]:
        return dict(self.metadata)


def add_graph_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("fresh", "warm"), required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--field-embeddings", type=Path, required=True)
    parser.add_argument(
        "--scheduling-field-embeddings", type=Path, required=True
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--embedding-base-url", required=True)


def build_stateless_graph(
    args: argparse.Namespace,
    *,
    definitions: Sequence[ToolDefinition],
    metadata: Mapping[str, Mapping[str, Any]],
    metadata_id: str,
    invocation_pool: BatchInvocationPool,
    runtime_id: str | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
    runtime: Any | None = None,
    max_states: int = 1,
    contract_adapter: ContractAdapter,
    state_domains: Mapping[str, str] | None = None,
    calls_per_worker_round: int = 30,
    worker_ipc_batch_size: int = 30,
) -> dict[str, Any]:
    manifest, _ = load_active_contract_manifest(
        args.contracts,
        metadata,
        metadata_id=metadata_id,
        adapter=contract_adapter,
    )
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
            adapter=contract_adapter,
        )
    contract_set = OnlineContractSet(
        manifest=manifest,
        metadata=metadata,
        generator=generator,
        count_all_history=True,
        max_total_llm_calls=3 * len(definitions),
    )
    field_store = SQLiteFieldEmbeddingStore(
        args.field_embeddings,
        base_url=args.embedding_base_url,
    )
    scheduling_store = SQLiteFieldEmbeddingStore(
        args.scheduling_field_embeddings,
        base_url=args.embedding_base_url,
    )
    store = SQLiteBuildStore(
        args.work_dir / "observations.sqlite",
        persist_tasks=False,
        persist_calls=False,
    )
    try:
        if field_store.descriptor_mode != "field":
            raise ValueError("argument binding requires field descriptors")
        if scheduling_store.descriptor_mode != "context":
            raise ValueError("global scheduling requires context descriptors")
        field_priority = SQLiteFieldPriorityIndex(
            field_store,
            definitions,
            embedder=QwenVllmTextEmbedder(base_url=args.embedding_base_url),
        )
        scheduling_priority = SQLiteFieldPriorityIndex(
            scheduling_store,
            definitions,
            embedder=QwenVllmTextEmbedder(base_url=args.embedding_base_url),
        )
        if runtime is None:
            if runtime_id is None:
                raise ValueError("stateless graph requires a runtime id")
            runtime = StatelessRootRuntime(
                runtime_id, dict(runtime_metadata or {})
            )
        max_external = 3 * len(definitions)
        max_repairs = (
            0 if generator is None
            else max(0, max_external - contract_set.total_llm_calls())
        )
        builder = ParallelDependencyGraphBuilder(
            runtime,
            invocation_pool,
            definitions,
            contract_set,
            store=store,
            config=BuildConfig(
                budget=SearchBudget(
                    max_calls=args.max_calls,
                    max_states=max_states,
                    max_depth=12,
                    max_seconds=82_800,
                    max_external_model_calls=max_external,
                    max_contract_repair_calls=max_repairs,
                    max_retries=100_000,
                ),
                trial_repetitions=2,
            ),
            parallel=ParallelBuildConfig(
                workers=96,
                calls_per_worker_epoch=calls_per_worker_round,
                worker_ipc_batch_size=worker_ipc_batch_size,
                prefetch_rounds=1,
                prefetch_calls_per_worker=8,
                contract_epoch_calls=96 * calls_per_worker_round,
                fairness_share=0.20,
                prioritize_state_progress=bool(state_domains),
                worker_retries=1,
                worker_failure_threshold=0.02,
                telemetry_path=args.telemetry_output,
                heartbeat_interval_seconds=60.0,
                storage_paths=(args.work_dir,),
                contract_repair_concurrency=16,
            ),
            field_priority=field_priority,
            scheduling_priority=scheduling_priority,
            state_domains=state_domains,
        )
        graph = asyncio.run(builder.run_async())
        builder.write_outputs(
            graph,
            graph_path=args.output,
            evidence_path=args.work_dir / "evidence.jsonl",
        )
        contract_set.manifest.write(args.contracts)
        return graph
    finally:
        invocation_pool.close()
        store.close()
        scheduling_store.close()
        field_store.close()
