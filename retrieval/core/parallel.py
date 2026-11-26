"""Bulk-round process orchestration for stateful dependency discovery.

The coordinator owns every mutable search object and is the sole SQLite
writer.  Runtime workers only load one frozen state, execute an invocation,
optionally persist its post-state handle, and return an ``InvocationResult``.
This keeps runtime process isolation while keeping scheduling out of SQLite.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import heapq
import json
import os
import resource
import stat
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from retrieval.core.graph import (
    DependencyGraphBuilder,
    InvocationResult,
    Observation,
    SearchState,
    SearchTask,
    ToolDefinition,
    binding_observations,
    canonical_json,
    classify_outcome,
    iter_json_values,
    json_safe,
    schema_accepts,
    serialized_bindings,
    stable_id,
    structural_feedback,
    system_monotonic,
)
from retrieval.core.contracts import (
    MAX_LLM_CALLS_PER_TOOL,
    OnlineContractSet,
    ToolCallContract,
)


PARALLEL_SCHEDULER_POLICY = "memory_bulk_round_score_selected_state_carry_v30"
FAIR_TASK_CLASSES = ("base", "event", "continuation", "provider")
STORAGE_CHECK_INTERVAL_SECONDS = 1200.0
_REUSABLE_TRIAL_REASONS = frozenset({
    "input_repeat",
    "input_ablation",
    "state_before",
    "state_after",
})
class AsyncInvocationPool(Protocol):
    async def invoke(
        self,
        *,
        state_handle: str,
        state_digest: str | None = None,
        tool: str,
        args: Mapping[str, Any],
        save_state_id: str | None,
    ) -> tuple[InvocationResult, str | None]: ...


@dataclass(frozen=True)
class ParallelBuildConfig:
    workers: int
    calls_per_worker_epoch: int = 30
    worker_ipc_batch_size: int | None = None
    prefetch_rounds: int = 0
    prefetch_calls_per_worker: int | None = None
    contract_epoch_calls: int | None = None
    fairness_share: float = 0.20
    worker_retries: int = 2
    worker_failure_threshold: float = 0.02
    telemetry_path: Path | None = None
    heartbeat_interval_seconds: float = 60.0
    storage_paths: tuple[Path, ...] = ()
    contract_repair_concurrency: int = 16
    prioritize_state_progress: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("parallel workers must be positive")
        if self.calls_per_worker_epoch < 1:
            raise ValueError("calls_per_worker_epoch must be positive")
        if self.worker_ipc_batch_size is not None and not (
            1 <= self.worker_ipc_batch_size <= self.calls_per_worker_epoch
        ):
            raise ValueError(
                "worker_ipc_batch_size must be between 1 and calls_per_worker_epoch"
            )
        if self.prefetch_rounds < 0:
            raise ValueError("prefetch_rounds must be non-negative")
        if self.prefetch_rounds and self.prefetch_calls_per_worker is None:
            raise ValueError(
                "prefetch_calls_per_worker is required for prefetch rounds"
            )
        if self.prefetch_calls_per_worker is not None and not (
            1 <= self.prefetch_calls_per_worker <= self.calls_per_worker_epoch
        ):
            raise ValueError(
                "prefetch_calls_per_worker must be between 1 and "
                "calls_per_worker_epoch"
            )
        if self.contract_epoch_calls is not None and self.contract_epoch_calls < 1:
            raise ValueError("contract_epoch_calls must be positive")
        if not 0.0 <= self.fairness_share < 1.0:
            raise ValueError("fairness_share must be in [0, 1)")
        if self.worker_retries < 0:
            raise ValueError("worker_retries must be non-negative")
        if not 0.0 <= self.worker_failure_threshold <= 1.0:
            raise ValueError("worker_failure_threshold must be in [0, 1]")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        if self.contract_repair_concurrency < 1:
            raise ValueError("contract repair concurrency must be positive")

    @property
    def epoch_calls(self) -> int:
        return self.contract_epoch_calls or self.round_capacity

    @property
    def round_capacity(self) -> int:
        return self.workers * self.calls_per_worker_epoch

    def round_capacity_for(self, completed_rounds: int) -> int:
        if (
            completed_rounds < self.prefetch_rounds
            and self.prefetch_calls_per_worker is not None
        ):
            return self.workers * self.prefetch_calls_per_worker
        return self.round_capacity

    @property
    def ipc_batch_size(self) -> int:
        return self.worker_ipc_batch_size or self.calls_per_worker_epoch


@dataclass
class _PendingInvocation:
    sequence: int
    owner: asyncio.Task[Any]
    task_id: str | None
    observation_call: str | None
    trigger_source_tools: tuple[str, ...]
    state_producer_tool: str | None
    state: SearchState
    tool: str
    args: Mapping[str, Any]
    save_on_change: bool
    binding_key: str
    reason: str
    future: asyncio.Future[
        tuple[InvocationResult, str | None, int, str | None] | None
    ]


def _partition_worker_batches(
    requests: Sequence[_PendingInvocation],
    *,
    workers: int,
    max_batch_size: int,
    observed_state_writer_tools: set[str],
) -> list[list[_PendingInvocation]]:
    """Balance known state writers while retaining state-local call order.

    A changed invocation invalidates the worker's loaded immutable state and
    may also persist a new handle. Clustering those calls makes one slow batch
    determine the latency of the whole frozen round. Writer history changes
    only physical worker assignment: candidate selection, call ordinals, and
    stable coordinator reduction remain untouched.
    """

    if not requests:
        return []
    if workers < 1 or max_batch_size < 1:
        raise ValueError("workers and max_batch_size must be positive")

    ordered = sorted(
        requests,
        key=lambda request: (request.state.handle, request.sequence),
    )
    minimum_batches = (len(ordered) + max_batch_size - 1) // max_batch_size
    batch_count = min(
        len(ordered),
        max(minimum_batches, min(workers, len(ordered))),
    )
    minimum_size, larger_batches = divmod(len(ordered), batch_count)
    target_sizes = [
        minimum_size + int(index < larger_batches)
        for index in range(batch_count)
    ]
    batches: list[list[_PendingInvocation]] = [
        [] for _ in range(batch_count)
    ]

    writers = [
        request
        for request in ordered
        if request.tool in observed_state_writer_tools
    ]
    nonwriters = [
        request
        for request in ordered
        if request.tool not in observed_state_writer_tools
    ]

    batch_index = 0
    for request in writers:
        while len(batches[batch_index]) >= target_sizes[batch_index]:
            batch_index = (batch_index + 1) % batch_count
        batches[batch_index].append(request)
        batch_index = (batch_index + 1) % batch_count

    batch_index = 0
    for request in nonwriters:
        while len(batches[batch_index]) >= target_sizes[batch_index]:
            batch_index += 1
        batches[batch_index].append(request)

    for batch in batches:
        batch.sort(key=lambda request: (request.state.handle, request.sequence))
    if any(len(batch) > max_batch_size for batch in batches):
        raise RuntimeError("worker batch exceeded configured IPC size")
    return batches


@dataclass(frozen=True)
class _PriorityEntry:
    task: SearchTask
    task_class: str
    enqueue_sequence: int
    semantic_missing: int
    semantic_score: float
    tie: str


@dataclass
class _LazyFrontier:
    id: str
    state_id: str
    observation_call: str | None
    task_class: str
    enqueue_sequence: int
    blueprint_id: str
    count: int
    next_index: int = 0


def _task_class(task: SearchTask) -> str:
    if task.provider_search:
        return "provider"
    if task.anchor_search:
        return "event"
    if task.observation_call is None and task.next_combination == 0:
        return "base"
    if task.observation_call is not None and task.next_combination == 0:
        return "event"
    return "continuation"


class ParallelDependencyGraphBuilder(DependencyGraphBuilder):
    """Run many task state machines concurrently around one serial coordinator."""

    def __init__(
        self,
        runtime: Any,
        invocation_pool: AsyncInvocationPool,
        definitions: Sequence[ToolDefinition],
        contracts: OnlineContractSet,
        *,
        store: Any,
        parallel: ParallelBuildConfig,
        config: Any = None,
        field_priority: Any = None,
        scheduling_priority: Any = None,
        state_domains: Mapping[str, str] | None = None,
    ) -> None:
        configured_budget = getattr(config, "budget", None)
        if (
            getattr(configured_budget, "max_storage_bytes", None) is not None
            and not parallel.storage_paths
        ):
            raise ValueError(
                "parallel max_storage_bytes requires explicit storage_paths; "
                "pass (store.path,) for an intentional SQLite-only budget"
            )
        self.invocation_pool = invocation_pool
        self.parallel = parallel
        self.state_domains = dict(state_domains or {})
        if parallel.prioritize_state_progress and not self.state_domains:
            raise ValueError(
                "prioritize_state_progress requires adapter state domains"
            )
        self._priority_entries: dict[str, _PriorityEntry] = {}
        self._semantic_heap: list[tuple[int, float, int, str, str]] = []
        self._fair_heaps: dict[str, list[tuple[int, str, str]]] = {
            task_class: [] for task_class in FAIR_TASK_CLASSES
        }
        self._lazy_blueprints: dict[str, tuple[dict[str, Any], ...]] = {}
        self._lazy_frontiers: dict[str, _LazyFrontier] = {}
        self._frontier_task_ids: dict[str, str] = {}
        self._observation_frontier_specs: dict[
            tuple[str, str, tuple[tuple[str, int], ...]],
            tuple[tuple[str, int, int, bool], ...],
        ] = {}
        self._observation_frontier_blueprints: dict[
            tuple[
                str,
                str,
                tuple[tuple[str, int], ...],
                str | None,
                tuple[tuple[str, int, int], ...],
            ],
            tuple[str, tuple[dict[str, Any], ...]],
        ] = {}
        self._frontier_order_ties: dict[tuple[Any, ...], str] = {}
        self._parallel_contract_revision_key: tuple[
            tuple[str, int], ...
        ] | None = None
        self._seen_task_ids: set[str] = set()
        self._stale_task_updates: dict[str, SearchTask] = {}
        self._logged_edge_provenances: set[str] = set()
        self._logged_success_tools: set[str] = set()
        self._pending_edge_log_events: list[dict[str, Any]] = []
        self._dispatch_slot = 0
        self._fair_task_class_cursor = 0
        self._enqueue_sequence = 0
        self._leased_task_ids: set[str] = set()
        self._leased_entries: dict[str, _PriorityEntry] = {}
        self._deferred_tasks: dict[str, SearchTask] = {}
        # Serial discovery sees a committed call before another task can reach
        # the same concrete binding.  A frozen bulk fill does not: several
        # observation-triggered tasks can produce identical arguments before
        # any result is reduced.  Reserve that exact template binding in memory
        # for the current round so parallelism does not turn it into duplicate
        # runtime calls.  Different template indexes remain distinct.
        self._round_binding_owners: dict[str, str] = {}
        self._round_binding_deferred_entries: dict[str, _PriorityEntry] = {}
        self._round_task_combinations: dict[
            str,
            tuple[dict[str, Any], dict[str, tuple[Observation, ...]]] | None,
        ] = {}
        self._pending_feedback: dict[
            str, dict[str, Mapping[str, Any]]
        ] = defaultdict(dict)
        self._pending_state_repairs: dict[str, tuple[dict[str, Any], InvocationResult, ToolCallContract]] = {}
        self._incomplete_task_bindings: dict[str, str] = {}
        self._incomplete_binding_owners: dict[str, str] = {}
        for task_id, binding_key in self._incomplete_task_bindings.items():
            previous = self._incomplete_binding_owners.setdefault(
                binding_key, task_id
            )
            if previous != task_id:
                raise RuntimeError(
                    "multiple incomplete tasks own one primary binding: "
                    f"{binding_key}"
                )
        self._state_trial_owners: dict[str, str] = {}
        self._task_state_trial_claims: dict[str, str] = {}
        self._retained_state_replays: dict[str, dict[str, Any]] = {}
        self._trial_binding_reuses = 0
        self._worker_dispatches = 0
        self._worker_retries = 0
        self._worker_terminal_failures = 0
        self._last_worker_error: dict[str, Any] | None = None
        self._last_coordinator_error: dict[str, Any] | None = None
        self._work_items_completed = 0
        self._terminal_failure_futures: set[asyncio.Task[Any]] = set()
        self._future_task_ids: dict[asyncio.Task[Any], str] = {}
        self._active_inflight_count = 0
        self._completion_counter = 0
        self._completion_order: dict[asyncio.Task[Any], int] = {}
        self._bulk_run_active = False
        self._pending_invocations: list[_PendingInvocation] = []
        self._pending_by_owner: dict[asyncio.Task[Any], _PendingInvocation] = {}
        self._bulk_invocation_requested: set[asyncio.Task[Any]] = set()
        self._dispatch_slot_returned: set[asyncio.Task[Any]] = set()
        self._invocation_ordinals: dict[asyncio.Task[Any], int] = {}
        self._preclassified_outcomes: dict[asyncio.Task[Any], str] = {}
        # This is an ephemeral performance hint, never a ranking or candidate
        # input; graph contents remain defined by stable reduction order.
        self._observed_state_writer_tools: set[str] = set()
        self._invocation_sequence = 0
        self._rounds_completed = 0
        self._round_phase = "initializing"
        self._last_round_timings: dict[str, Any] = {}
        self._last_fill_timings: dict[str, float | int] = {}
        self._last_epoch_tail_seconds = 0.0
        self._last_epoch_tail_timings: dict[str, float] = {}
        self._max_epoch_tail_seconds = 0.0
        self._epoch_tail_seconds_total = 0.0
        self._epochs_completed = 0
        self._epoch_boundary_pending = False
        self._outstanding_reservations: dict[str, int] = {}
        super().__init__(
            runtime,
            definitions,
            contracts,
            store=store,
            config=config,
            field_priority=field_priority,
            scheduling_priority=scheduling_priority,
        )
        # The parallel scheduler has its own global in-memory heaps; the serial
        # scheduler's per-state deque mirror is unused.
        del self.scheduled_task_ids
        self.scheduler_policy = PARALLEL_SCHEDULER_POLICY
        budget_limits = dict(self.config.budget.__dict__)
        store.set_meta("parallel_budget_limits", budget_limits)
        self._epoch_start_calls = self.calls_used
        self._last_telemetry = system_monotonic()
        self._last_storage_inodes = 0
        self._last_storage_bytes = self._storage_bytes()
        self._last_storage_check = system_monotonic()
        self._validate_incomplete_task_queue()
        self._rebuild_state_trial_owners()
        self._validate_retained_state_replays()

    def _frontier_ordering_key(
        self, task: SearchTask
    ) -> tuple[int, int, int, float, str]:
        # Shared blueprints are state-independent.  Order their target fields
        # by the raw semantic score; whether this particular state owns the
        # one tuple priority slot is applied later when its live queue
        # entry is created.
        score = self._task_field_score(task)
        template_order = (
            task.template_index
            if task.observation_call is None and not task.provider_search
            else 0
        )
        state_progress_order = 0
        state = self.states.get(task.state_id)
        if (
            self.parallel.prioritize_state_progress
            and task.observation_call is None
            and state is not None
            and state.depth > 0
        ):
            if task.tool in self._observed_state_writer_tools:
                state_progress_order = 0
            elif task.tool not in self._first_success_call_by_tool:
                state_progress_order = 1
            else:
                state_progress_order = 2
        # State/source identity is already part of the frontier and the global
        # queue tie.  Keeping it out of the intra-frontier blueprint order lets
        # many state-specific exhaustive frontiers share one compact immutable
        # blueprint instead of duplicating hundreds of task rows.
        tie_key = (
            task.tool,
            task.revision,
            task.template_index,
            task.provider_search,
            task.anchor_search,
            task.next_combination,
        )
        tie = self._frontier_order_ties.get(tie_key)
        if tie is None:
            tie = stable_id("parallel_lazy_frontier_order", tie_key)
            self._frontier_order_ties[tie_key] = tie
        return (
            template_order,
            state_progress_order,
            1 if score is None else 0,
            -float(score or 0.0),
            tie,
        )

    def _create_lazy_frontier(
        self,
        tasks: Sequence[SearchTask],
        *,
        namespace: Any,
        canonical_task_ids: bool = False,
    ) -> None:
        ordered = sorted(tasks, key=self._frontier_ordering_key)
        if not ordered:
            return
        task_class = _task_class(ordered[0])
        if any(_task_class(task) != task_class for task in ordered):
            raise ValueError("one lazy frontier cannot mix task classes")
        start_sequence = self._enqueue_sequence + 1
        self._enqueue_sequence += len(ordered)
        frontier_id = stable_id(
            "parallel_lazy_frontier",
            [namespace, [task.id for task in ordered]],
        )
        if frontier_id in self._lazy_frontiers:
            return
        blueprints: list[dict[str, Any]] = []
        for task in ordered:
            row: dict[str, Any] = {
                "tool": task.tool,
                "revision": task.revision,
                "template_index": task.template_index,
                "provider_search": task.provider_search,
                "anchor_search": task.anchor_search,
                "next_combination": task.next_combination,
            }
            if not canonical_task_ids:
                expected_id = stable_id(
                    "task",
                    (
                        [
                            task.state_id,
                            task.tool,
                            task.revision,
                            task.template_index,
                            task.observation_call,
                        ]
                        if task.observation_call is not None
                        else [
                            task.state_id,
                            task.tool,
                            task.revision,
                            task.template_index,
                        ]
                    ),
                )
                if task.id != expected_id:
                    row["id"] = task.id
            blueprints.append(row)
        blueprint_id = stable_id("parallel_lazy_blueprint", blueprints)
        self._lazy_blueprints.setdefault(blueprint_id, tuple(blueprints))
        self._lazy_frontiers[frontier_id] = _LazyFrontier(
            id=frontier_id,
            state_id=ordered[0].state_id,
            observation_call=ordered[0].observation_call,
            task_class=task_class,
            enqueue_sequence=start_sequence,
            blueprint_id=blueprint_id,
            count=len(blueprints),
        )
        self._advance_lazy_frontier(frontier_id)

    def _current_parallel_contract_revision_key(
        self,
    ) -> tuple[tuple[str, int], ...]:
        cached = self._parallel_contract_revision_key
        if cached is None:
            cached = tuple(
                (tool, contract.revision)
                for tool, contract in sorted(
                    self.contracts.manifest.contracts.items()
                )
            )
            self._parallel_contract_revision_key = cached
        return cached

    def _observation_frontier_blueprint(
        self,
        state: SearchState,
        observation_call: str,
        task_class: str,
        contract_revision_key: tuple[tuple[str, int], ...],
    ) -> tuple[str, tuple[dict[str, Any], ...]] | None:
        specs_key = (observation_call, task_class, contract_revision_key)
        specs = self._observation_frontier_specs.get(specs_key)
        if specs is None:
            specs = tuple(
                row
                for row in self._observation_blueprints(observation_call)
                if ("provider" if row[3] else "event") == task_class
            )
            self._observation_frontier_specs[specs_key] = specs
        if not specs:
            return None
        state_domain = self._state_domain(state)
        if state_domain is not None:
            specs = tuple(
                row for row in specs
                if self.state_domains.get(row[0]) == state_domain
            )
        if not specs:
            return None
        baseline_signature = tuple(
            (tool, revision, template_index)
            for tool, revision, template_index, _ in specs
            if (
                state.id,
                tool,
                revision,
                template_index,
            ) in self.literal_baselines
        )
        cache_key = (
            observation_call,
            task_class,
            contract_revision_key,
            state_domain,
            baseline_signature,
        )
        cached = self._observation_frontier_blueprints.get(cache_key)
        if cached is not None:
            return cached
        ordered = sorted(
            (
                SearchTask(
                    # This task exists only long enough to compute the shared
                    # state-independent ordering key; its stable ID is rebuilt
                    # when a concrete frontier head is materialized.
                    id="",
                    state_id=state.id,
                    tool=tool,
                    revision=revision,
                    template_index=template_index,
                    observation_call=observation_call,
                    provider_search=provider_search,
                )
                for tool, revision, template_index, provider_search in specs
            ),
            key=self._frontier_ordering_key,
        )
        blueprints = tuple({
            "tool": task.tool,
            "revision": task.revision,
            "template_index": task.template_index,
            "provider_search": task.provider_search,
            "next_combination": task.next_combination,
        } for task in ordered)
        blueprint_id = stable_id("parallel_lazy_blueprint", blueprints)
        self._lazy_blueprints.setdefault(blueprint_id, blueprints)
        result = (blueprint_id, blueprints)
        self._observation_frontier_blueprints[cache_key] = result
        return result

    def _create_shared_observation_frontier(
        self,
        *,
        state: SearchState,
        observation_call: str,
        task_class: str,
        contract_revision_key: tuple[tuple[str, int], ...],
    ) -> None:
        cached = self._observation_frontier_blueprint(
            state,
            observation_call,
            task_class,
            contract_revision_key,
        )
        if cached is None:
            return
        blueprint_id, blueprints = cached
        namespace = [
            "observation-v2",
            state.id,
            observation_call,
            task_class,
            contract_revision_key,
        ]
        frontier_id = stable_id(
            "parallel_lazy_frontier_v2", [namespace, blueprint_id]
        )
        if frontier_id in self._lazy_frontiers:
            return
        start_sequence = self._enqueue_sequence + 1
        self._enqueue_sequence += len(blueprints)
        self._lazy_frontiers[frontier_id] = _LazyFrontier(
            id=frontier_id,
            state_id=state.id,
            observation_call=observation_call,
            task_class=task_class,
            enqueue_sequence=start_sequence,
            blueprint_id=blueprint_id,
            count=len(blueprints),
        )
        self._advance_lazy_frontier(frontier_id)

    def _frontier_task(self, frontier: _LazyFrontier, index: int) -> SearchTask:
        blueprint = self._lazy_blueprints[frontier.blueprint_id][index]
        task_id = str(blueprint.get("id") or stable_id(
            "task",
            (
                [
                    frontier.state_id,
                    str(blueprint["tool"]),
                    int(blueprint["revision"]),
                    int(blueprint["template_index"]),
                    frontier.observation_call,
                ]
                if frontier.observation_call is not None
                else [
                    frontier.state_id,
                    str(blueprint["tool"]),
                    int(blueprint["revision"]),
                    int(blueprint["template_index"]),
                ]
            ),
        ))
        return SearchTask(
            id=task_id,
            state_id=frontier.state_id,
            tool=str(blueprint["tool"]),
            revision=int(blueprint["revision"]),
            template_index=int(blueprint["template_index"]),
            observation_call=frontier.observation_call,
            provider_search=bool(blueprint.get("provider_search", False)),
            anchor_search=bool(blueprint.get("anchor_search", False)),
            next_combination=int(blueprint.get("next_combination", 0)),
        )

    def _put_priority_entry(self, entry: _PriorityEntry) -> bool:
        task_id = entry.task.id
        if task_id in self._priority_entries or task_id in self._leased_task_ids:
            return False
        self._priority_entries[task_id] = entry
        heapq.heappush(self._semantic_heap, (
            entry.semantic_missing,
            -entry.semantic_score,
            -entry.enqueue_sequence,
            entry.tie,
            task_id,
        ))
        heapq.heappush(self._fair_heaps[entry.task_class], (
            entry.enqueue_sequence,
            entry.tie,
            task_id,
        ))
        return True

    def _advance_lazy_frontier(self, frontier_id: str) -> None:
        frontier = self._lazy_frontiers.get(frontier_id)
        if frontier is None:
            return
        while frontier.next_index < frontier.count:
            index = frontier.next_index
            frontier.next_index += 1
            task = self._frontier_task(frontier, index)
            if not self._task_runs_in_state_domain(task):
                self._seen_task_ids.add(task.id)
                continue
            if self._observation_task_is_pruned(task):
                self._seen_task_ids.add(task.id)
                self.observation_tasks_pruned_after_success += 1
                continue
            if task.id in self._seen_task_ids:
                continue
            score = self._task_semantic_score(task)
            tie = stable_id("parallel_task_schedule", [
                task.state_id,
                task.tool,
                task.revision,
                task.template_index,
                task.observation_call,
                task.provider_search,
                task.anchor_search,
                task.next_combination,
            ])
            entry = _PriorityEntry(
                task=task,
                task_class=_task_class(task),
                enqueue_sequence=frontier.enqueue_sequence + index,
                semantic_missing=1 if score is None else 0,
                semantic_score=float(score or 0.0),
                tie=tie,
            )
            self._seen_task_ids.add(task.id)
            if self._put_priority_entry(entry):
                self._frontier_task_ids[task.id] = frontier_id
                return
        self._lazy_frontiers.pop(frontier_id, None)

    def _enqueue_state(self, state: SearchState) -> None:
        tasks: list[SearchTask] = []
        for tool, contract in self.contracts.manifest.contracts.items():
            if not self._tool_runs_in_state_domain(tool, state):
                continue
            if (tool, contract.revision) in self.blocked_contract_revisions:
                continue
            for index, template in enumerate(contract.templates):
                if state.depth > 0 and template.expected == "failure":
                    continue
                tasks.append(SearchTask(
                    id=stable_id(
                        "task", [state.id, tool, contract.revision, index]
                    ),
                    state_id=state.id,
                    tool=tool,
                    revision=contract.revision,
                    template_index=index,
                ))
        self._create_lazy_frontier(
            tasks,
            namespace=[
                "base",
                state.id,
                self._current_parallel_contract_revision_key(),
            ],
            canonical_task_ids=True,
        )

    def _tool_runs_in_state_domain(
        self, tool: str, state: SearchState
    ) -> bool:
        state_domain = self._state_domain(state)
        if state_domain is None:
            return True
        return self.state_domains.get(tool) == state_domain

    def _state_domain(self, state: SearchState) -> str | None:
        if state.depth == 0 or not self.state_domains:
            return None
        producer = self._state_producer_tool(state)
        return self.state_domains.get(str(producer or ""))

    def _task_runs_in_state_domain(self, task: SearchTask) -> bool:
        state = self.states.get(task.state_id)
        return state is None or self._tool_runs_in_state_domain(
            task.tool, state
        )

    def _enqueue_observation_tasks(
        self, state: SearchState, observation_call: str
    ) -> None:
        contract_revision_key = (
            self._current_parallel_contract_revision_key()
        )
        for task_class in ("event", "provider"):
            self._create_shared_observation_frontier(
                state=state,
                observation_call=observation_call,
                task_class=task_class,
                contract_revision_key=contract_revision_key,
            )

    def _index_new_observations(
        self, observations: Sequence[Observation]
    ) -> None:
        super()._index_new_observations(observations)
        affected = {row.source_call for row in observations}
        for cache in (
            self._observation_frontier_specs,
            self._observation_frontier_blueprints,
        ):
            for key in tuple(cache):
                if key[0] in affected:
                    cache.pop(key, None)

    def _enqueue_tool_revision(self, tool: str, revision: int) -> None:
        if (tool, revision) in self.blocked_contract_revisions:
            return
        contract = self.contracts.get(tool)
        if contract is None or contract.revision != revision:
            return
        definition = self.definitions[tool]
        observation_blueprints: list[
            tuple[str, tuple[tuple[int, bool], ...]]
        ] = []
        # The all-tool blueprint cache is deliberately not used here: after a
        # revision it is stale for this tool, and rebuilding it once per
        # repaired tool scans the whole registry repeatedly. Compute only this
        # tool's compatibility once, then reuse it across every state.
        for observation_call in self._observation_source_calls():
            observations = self._observations_for_source_call(observation_call)
            if not any(
                schema_accepts(parameter.schema, observation.value)
                for parameter in definition.parameters
                for observation in observations
            ):
                continue
            blueprints = tuple(
                (index, not template.has_observations())
                for index, template in enumerate(contract.templates)
                if template.expected == "success"
            )
            if blueprints:
                observation_blueprints.append((observation_call, blueprints))
        for state in self.states.values():
            if not self._tool_runs_in_state_domain(tool, state):
                continue
            base_tasks = [
                SearchTask(
                    id=stable_id("task", [state.id, tool, revision, index]),
                    state_id=state.id,
                    tool=tool,
                    revision=revision,
                    template_index=index,
                )
                for index, template in enumerate(contract.templates)
                if not (state.depth > 0 and template.expected == "failure")
            ]
            self._create_lazy_frontier(
                base_tasks,
                namespace=["revision-base", state.id, tool, revision],
                canonical_task_ids=True,
            )
            for observation_call, blueprints in observation_blueprints:
                by_class: dict[str, list[SearchTask]] = defaultdict(list)
                for index, provider in blueprints:
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
                        provider_search=provider,
                    )
                    by_class[_task_class(task)].append(task)
                for task_class, selected in sorted(by_class.items()):
                    self._create_lazy_frontier(
                        selected,
                        namespace=[
                            "revision-observation",
                            state.id,
                            observation_call,
                            tool,
                            revision,
                            task_class,
                        ],
                        canonical_task_ids=True,
                    )

    def _schedule_task(self, task: SearchTask, *, front: bool = False) -> None:
        del front
        if task.completed:
            return
        if not self._task_runs_in_state_domain(task):
            task.completed = True
            self._seen_task_ids.add(task.id)
            self._stale_task_updates[task.id] = task
            return
        if self._observation_task_is_pruned(task):
            task.completed = True
            self.observation_tasks_pruned_after_success += 1
            self._stale_task_updates[task.id] = task
            return
        if task.id in self._round_binding_deferred_entries:
            # The exact original entry is restored after stable reduction.
            # Re-enqueuing it here would let fill select the same duplicate
            # repeatedly before the owning call has been committed.
            return
        if task.id in self._leased_task_ids:
            self._deferred_tasks[task.id] = task
            return
        if task.id in self._priority_entries:
            return
        score = self._task_semantic_score(task)
        tie = stable_id("parallel_task_schedule", [
            task.state_id,
            task.tool,
            task.revision,
            task.template_index,
            task.observation_call,
            task.provider_search,
            task.anchor_search,
            task.next_combination,
        ])
        self._enqueue_sequence += 1
        self._seen_task_ids.add(task.id)
        self._put_priority_entry(_PriorityEntry(
            task=task,
            task_class=_task_class(task),
            enqueue_sequence=self._enqueue_sequence,
            semantic_missing=1 if score is None else 0,
            semantic_score=float(score or 0.0),
            tie=tie,
        ))

    def _schedule_tasks(self, tasks: Any) -> None:
        ordered = sorted(
            tasks,
            key=lambda task: stable_id("parallel_task_insert", task.to_dict()),
        )
        for task in ordered:
            if task.completed:
                continue
            if not self._task_runs_in_state_domain(task):
                task.completed = True
                self._seen_task_ids.add(task.id)
                self._stale_task_updates[task.id] = task
                continue
            if task.id in self._leased_task_ids:
                self._deferred_tasks[task.id] = task
                continue
            if task.id in self._priority_entries:
                continue
            score = self._task_semantic_score(task)
            tie = stable_id("parallel_task_schedule", [
                task.state_id,
                task.tool,
                task.revision,
                task.template_index,
                task.observation_call,
                task.provider_search,
                task.anchor_search,
                task.next_combination,
            ])
            self._enqueue_sequence += 1
            self._seen_task_ids.add(task.id)
            self._put_priority_entry(_PriorityEntry(
                task=task,
                task_class=_task_class(task),
                enqueue_sequence=self._enqueue_sequence,
                semantic_missing=1 if score is None else 0,
                semantic_score=float(score or 0.0),
                tie=tie,
            ))

    def _pop_priority_heap(
        self,
        lane: str,
        heap: list[Any],
    ) -> SearchTask | None:
        while True:
            if not heap:
                return None
            row = heapq.heappop(heap)
            task_id = str(row[-1])
            entry = self._priority_entries.get(task_id)
            if entry is None:
                continue
            expected = (
                (
                    entry.semantic_missing,
                    -entry.semantic_score,
                    -entry.enqueue_sequence,
                    entry.tie,
                    task_id,
                )
                if lane == "semantic"
                else (entry.enqueue_sequence, entry.tie, task_id)
            )
            if row != expected:
                continue
            if self._observation_task_is_pruned(entry.task):
                self._priority_entries.pop(task_id, None)
                entry.task.completed = True
                self.observation_tasks_pruned_after_success += 1
                self._stale_task_updates[task_id] = entry.task
                frontier_id = self._frontier_task_ids.pop(task_id, None)
                if frontier_id is not None:
                    self._advance_lazy_frontier(frontier_id)
                continue
            incomplete = self._incomplete_binding(entry.task.id) is not None
            if (
                lane == "semantic"
                and entry.task.observation_call is not None
                and not incomplete
            ):
                # A producer's single preferred state can move when a later
                # successful state transition consumes its observation.  Do
                # not rescore the whole queue: lazily demote any old-state head
                # when it reaches the semantic heap.  The new child frontier
                # is scored with the updated mapping when it is enqueued.
                score = self._task_semantic_score(entry.task)
                semantic_missing = 1 if score is None else 0
                semantic_score = float(score or 0.0)
                if (
                    semantic_missing != entry.semantic_missing
                    or semantic_score != entry.semantic_score
                ):
                    entry = _PriorityEntry(
                        task=entry.task,
                        task_class=entry.task_class,
                        enqueue_sequence=entry.enqueue_sequence,
                        semantic_missing=semantic_missing,
                        semantic_score=semantic_score,
                        tie=entry.tie,
                    )
                    self._priority_entries[task_id] = entry
                    heapq.heappush(self._semantic_heap, (
                        entry.semantic_missing,
                        -entry.semantic_score,
                        -entry.enqueue_sequence,
                        entry.tie,
                        task_id,
                    ))
                    continue
            active_contract = self.contracts.get(entry.task.tool)
            if (
                active_contract is None
                or (
                    active_contract.revision != entry.task.revision
                    and self._compatible_incomplete_contract(entry.task) is None
                )
            ):
                # Contract publication makes old-revision tasks obsolete.
                # Drop them lazily when their heap head is reached instead of
                # rescoring every queued task at each epoch boundary.
                self._priority_entries.pop(task_id, None)
                self._discard_incomplete_task_state(task_id)
                entry.task.completed = True
                # Coalesce stale revision heads until the next progress write.
                self._stale_task_updates[task_id] = entry.task
                frontier_id = self._frontier_task_ids.pop(task_id, None)
                if frontier_id is not None:
                    self._advance_lazy_frontier(frontier_id)
                continue
            self._priority_entries.pop(task_id, None)
            self._leased_entries[task_id] = entry
            return entry.task

    def _pop_priority_task(self, lane: str) -> SearchTask | None:
        if lane == "semantic":
            return self._pop_priority_heap(lane, self._semantic_heap)
        if lane != "fair":
            raise ValueError(f"unknown priority lane: {lane}")

        # The protection lane is oldest-first within each work class, while
        # its four classes rotate so that continuation tuples are not hidden
        # behind a large backlog of base/event frontier heads.
        for offset in range(len(FAIR_TASK_CLASSES)):
            index = (
                self._fair_task_class_cursor + offset
            ) % len(FAIR_TASK_CLASSES)
            task_class = FAIR_TASK_CLASSES[index]
            task = self._pop_priority_heap(
                lane, self._fair_heaps[task_class]
            )
            if task is not None:
                self._fair_task_class_cursor = (
                    index + 1
                ) % len(FAIR_TASK_CLASSES)
                return task
        return None

    def _select_priority_task(self) -> SearchTask | None:
        self._dispatch_slot += 1
        fair = (
            self.parallel.fairness_share > 0
            and (self._dispatch_slot * self.parallel.fairness_share) // 1
            > ((self._dispatch_slot - 1) * self.parallel.fairness_share) // 1
        )
        primary = "fair" if fair else "semantic"
        fallback = "semantic" if fair else "fair"
        task = self._pop_priority_task(primary)
        if task is None:
            task = self._pop_priority_task(fallback)
        if task is None:
            self._dispatch_slot -= 1
        return task

    def _buffer_feedback(
        self,
        tool: str,
        feedback: Mapping[str, Any],
        *,
        kind: str,
        source_call_id: str | None,
    ) -> None:
        row = {
            "kind": kind,
            "feedback": dict(feedback),
            "source_call_id": source_call_id,
        }
        fingerprint = stable_id("parallel_repair_feedback", json_safe(row))
        self._pending_feedback[tool][fingerprint] = row

    def _current_discovery_task_id(self) -> str | None:
        current = asyncio.current_task()
        return self._future_task_ids.get(current) if current is not None else None

    def _compatible_incomplete_contract(
        self, task: SearchTask
    ) -> tuple[ToolCallContract, int] | None:
        """Keep one exact committed call alive across template-stable repair.

        State-export and outcome repairs can change a contract revision while
        preserving its selected invocation template.  The successful call
        that triggered such a repair still has input/state trials to finish;
        discarding it here loses the only evidence that may motivate the next
        repair.  A template-changing repair remains incompatible and follows
        the normal stale-task cleanup path.
        """

        binding_key = self._incomplete_task_bindings.get(task.id)
        if binding_key is None:
            return None
        call = self.store.call_by_binding(binding_key)
        active = self.contracts.get(task.tool)
        call_template_index = (
            int(call["template_index"])
            if call is not None and call.get("template_index") is not None
            else None
        )
        if (
            call is None
            or active is None
            or active.revision < task.revision
            or str(call.get("state_id") or "") != task.state_id
            or str(call.get("tool") or "") != task.tool
            or int(call.get("contract_revision") or 0) != task.revision
            or call_template_index != task.template_index
        ):
            return None
        template_index = self._matching_selected_template_index(active, call)
        if template_index is None:
            return None
        return active, template_index

    def _retained_state_handle_is_available(self, handle: str) -> bool:
        available = getattr(self.runtime, "state_handle_available", None)
        return bool(available(handle)) if available is not None else True

    def _validate_retained_state_replays(self) -> None:
        for task_id, record in self._retained_state_replays.items():
            if task_id not in self._incomplete_task_bindings:
                raise RuntimeError(
                    f"retained state replay has no incomplete task: {task_id}"
                )
            source_binding = str(record.get("source_binding") or "")
            handle = str(record.get("handle") or "")
            source_call = str(record.get("source_call") or "")
            call = self.store.call_by_binding(source_binding)
            primary = self.store.call_by_binding(
                self._incomplete_task_bindings[task_id]
            )
            if (
                not source_binding
                or not handle
                or primary is None
                or call is None
                or str(call.get("id") or "") != source_call
                or str(call.get("saved_state_handle") or "") != handle
                or not self._retained_state_handle_is_available(handle)
            ):
                raise RuntimeError(
                    f"retained state replay is incomplete for task {task_id}"
                )

    def _validate_incomplete_task_queue(self) -> None:
        if not self._incomplete_task_bindings:
            return
        missing = sorted(
            set(self._incomplete_task_bindings) - set(self._priority_entries)
        )
        if missing:
            raise RuntimeError(
                "incomplete tasks are absent from the live scheduler: "
                + ", ".join(missing[:10])
            )

    def _drop_retained_state_replay(
        self,
        task_id: str,
        *,
        discard: bool,
    ) -> dict[str, Any] | None:
        record = self._retained_state_replays.pop(task_id, None)
        if record is not None and discard:
            handle = str(record.get("handle") or "")
            if handle:
                super()._discard_unreferenced_state_handle(handle)
        return record

    def _discard_unreferenced_state_handle(self, handle: str) -> None:
        if any(
            str(record.get("handle") or "") == handle
            for record in self._retained_state_replays.values()
        ):
            return
        super()._discard_unreferenced_state_handle(handle)

    def _discard_incomplete_task_state(self, task_id: str) -> None:
        self._drop_retained_state_replay(task_id, discard=True)
        self._release_state_trial_claim(task_id)
        super()._discard_incomplete_task_state(task_id)

    def _reused_trial_invocation(
        self,
        binding_key: str,
    ) -> tuple[dict[str, Any], InvocationResult, str, str | None] | None:
        call = self.store.call_by_binding(binding_key)
        if call is None:
            return None
        self._trial_binding_reuses += 1
        saved_handle = str(call.get("saved_state_handle") or "") or None
        return (
            call,
            InvocationResult.from_dict(call["result"]),
            str(call.get("outcome") or "unknown"),
            saved_handle,
        )

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
        task_id = self._current_discovery_task_id()

        if reason in _REUSABLE_TRIAL_REASONS:
            if reason == "state_after" and task_id is not None:
                replay = self._retained_state_replays.get(task_id)
                if replay is None:
                    raise RuntimeError(
                        f"state_after has no retained replay for task {task_id}"
                    )
                recorded = replay.get("after_binding")
                if recorded not in (None, binding_key):
                    raise RuntimeError(
                        f"state_after binding changed for task {task_id}"
                    )
                replay["after_binding"] = binding_key
            reused = self._reused_trial_invocation(binding_key)
            if reused is not None:
                if reason == "state_after" and task_id is not None:
                    # The caller's finally block now owns deletion of the
                    # replay handle. Removing the retention record here makes
                    # that exact discard visible to the normal lifecycle.
                    self._drop_retained_state_replay(task_id, discard=False)
                return reused

        if reason == "state_source_replay":
            if task_id is None:
                raise RuntimeError("state replay has no owning discovery task")
            replay = self._retained_state_replays.get(task_id)
            if (
                replay is not None
                and str(replay.get("source_binding") or "") != binding_key
            ):
                raise RuntimeError(
                    f"state replay binding changed for task {task_id}"
                )
            reused = self._reused_trial_invocation(binding_key)
            if reused is not None:
                handle = reused[3]
                if replay is not None:
                    if (
                        handle != str(replay.get("handle") or "")
                        or handle is None
                        or not self._retained_state_handle_is_available(handle)
                    ):
                        raise RuntimeError(
                            "retained replay handle disappeared for task "
                            f"{task_id}"
                        )
                elif reused[2] == "success" and handle is not None:
                    if not self._retained_state_handle_is_available(handle):
                        raise RuntimeError(
                            "completed state replay handle is unavailable for "
                            f"task {task_id}"
                        )
                    # The exact source invocation is already recorded. Recover
                    # its still-available immutable handle instead of issuing
                    # a second call under a suffixed binding key.
                    self._retained_state_replays[task_id] = {
                        "source_binding": binding_key,
                        "after_binding": None,
                        "handle": handle,
                        "source_call": str(reused[0]["id"]),
                    }
                return reused
            if replay is not None:
                raise RuntimeError(
                    f"retained replay call disappeared for task {task_id}"
                )

        invoked = await super()._invoke_async(
            state=state,
            tool=tool,
            revision=revision,
            args=args,
            bindings=bindings,
            expected=expected,
            reason=reason,
            binding_key=binding_key,
            template_index=template_index,
            allow_repair=allow_repair,
            save_on_change=save_on_change,
        )
        if invoked is None:
            return None

        if reason == "state_source_replay" and task_id is not None:
            call, _result, outcome, handle = invoked
            if outcome == "success" and handle is not None:
                existing = self._retained_state_replays.get(task_id)
                if existing is not None:
                    raise RuntimeError(
                        f"task {task_id} already retains a state replay"
                    )
                self._retained_state_replays[task_id] = {
                    "source_binding": binding_key,
                    "after_binding": None,
                    "handle": handle,
                    "source_call": str(call["id"]),
                }
        elif reason == "state_after" and task_id is not None:
            self._drop_retained_state_replay(task_id, discard=False)
        return invoked

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
        if self._bulk_run_active:
            owner = asyncio.current_task()
            if owner is None:
                raise RuntimeError("bulk invocation has no owning discovery task")
            self._bulk_invocation_requested.add(owner)
            self._invocation_sequence += 1
            task_id = self._future_task_ids.get(owner)
            leased = self._leased_entries.get(task_id) if task_id else None
            observation_call = (
                leased.task.observation_call if leased is not None else None
            )
            request = _PendingInvocation(
                sequence=self._invocation_sequence,
                owner=owner,
                task_id=task_id,
                observation_call=observation_call,
                trigger_source_tools=tuple(sorted({
                    row.source_tool
                    for row in self._observations_for_source_call(
                        observation_call
                    )
                })),
                state_producer_tool=self._state_producer_tools.get(state.id),
                state=state,
                tool=tool,
                args=dict(args),
                save_on_change=save_on_change,
                binding_key=binding_key,
                reason=reason,
                future=asyncio.get_running_loop().create_future(),
            )
            self._pending_invocations.append(request)
            self._pending_by_owner[owner] = request
            payload = await request.future
            if self._pending_by_owner.get(owner) is request:
                self._pending_by_owner.pop(owner, None)
            if payload is None:
                self._terminal_failure_futures.add(owner)
                return None
            result, saved_handle, ordinal, outcome = payload
            self._invocation_ordinals[owner] = ordinal
            if outcome is not None:
                self._preclassified_outcomes[owner] = outcome
            return result, saved_handle

        base_save_state_id = (
            stable_id("parallel_post_state", [binding_key, reason])
            if save_on_change else None
        )
        self._worker_dispatches += 1
        result: InvocationResult | None = None
        saved_handle: str | None = None
        previous_worker: int | None = None
        had_worker_failure = False
        for attempt in range(self.parallel.worker_retries + 1):
            if (
                self.calls_used - self._epoch_start_calls
                >= self.parallel.epoch_calls
            ):
                self._epoch_boundary_pending = True
                break
            current = asyncio.current_task()
            task_id = self._future_task_ids.get(current) if current else None
            remaining_reservation = (
                self._outstanding_reservations.get(task_id, 0)
                if task_id is not None else 0
            )
            if remaining_reservation <= 0 and (
                self.calls_used
                + sum(self._outstanding_reservations.values())
                + 1
                > self.config.budget.max_calls
            ):
                self._record_budget_violation("max_calls")
                break
            if not self._budget_available():
                break
            # Pre-commit every actual worker attempt. A worker may die after
            # executing the runtime call but before returning its result.
            self.calls_used += 1
            if task_id is not None and remaining_reservation > 0:
                self._outstanding_reservations[task_id] = (
                    remaining_reservation - 1
                )
            self._emit_due_heartbeat()
            save_state_id = (
                stable_id(
                    "parallel_post_state_attempt",
                    [base_save_state_id, task_id, self.calls_used],
                )
                if base_save_state_id is not None else None
            )
            try:
                invoke_with_worker = getattr(
                    self.invocation_pool, "invoke_with_worker", None
                )
                if invoke_with_worker is None:
                    result, saved_handle = await self.invocation_pool.invoke(
                        state_handle=state.handle,
                        tool=tool,
                        args=args,
                        save_state_id=save_state_id,
                    )
                else:
                    result, saved_handle, previous_worker = await invoke_with_worker(
                        state_handle=state.handle,
                        tool=tool,
                        args=args,
                        save_state_id=save_state_id,
                        avoid_worker=previous_worker,
                    )
                break
            except Exception as exc:  # worker death is orchestration evidence
                had_worker_failure = True
                failed_worker = getattr(exc, "worker_id", None)
                if failed_worker is not None:
                    previous_worker = int(failed_worker)
                if attempt < self.parallel.worker_retries:
                    max_retries = self.config.budget.max_retries
                    if (
                        max_retries is not None
                        and self._worker_retries >= max_retries
                    ):
                        self._record_budget_violation("max_retries")
                        break
                    self._worker_retries += 1
                    continue
                self.store.set_meta("parallel_last_worker_error", {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                    "tool": tool,
                    "state": state.id,
                })
        if result is None:
            current = asyncio.current_task()
            if current is not None and had_worker_failure:
                self._terminal_failure_futures.add(current)
            return None
        return result, saved_handle

    def _invocation_ordinal(self) -> int:
        current = asyncio.current_task()
        if current is not None and current in self._invocation_ordinals:
            return self._invocation_ordinals[current]
        return super()._invocation_ordinal()

    def _classify_invocation_outcome(
        self, contract: ToolCallContract, result: InvocationResult
    ) -> str:
        # Workers return only the frozen contract's schema classification.
        # The inherited invocation path first buffers any repair and then
        # applies the generic transport-success/schema-unknown fallback.
        current = asyncio.current_task()
        if current is not None:
            outcome = self._preclassified_outcomes.pop(current, None)
            if outcome is not None:
                return outcome
        return super()._classify_invocation_outcome(contract, result)

    def _request_contract_repair(
        self,
        tool: str,
        feedback: Mapping[str, Any],
        *,
        kind: str,
        source_call_id: str | None = None,
    ) -> ToolCallContract | None:
        # A frozen/no-online-repair run has no generator, so buffering this
        # feedback can only produce rejected repairs at the epoch boundary,
        # so it is skipped.
        if self.contracts.generator is None:
            return None
        self._buffer_feedback(
            tool,
            feedback,
            kind=kind,
            source_call_id=source_call_id,
        )
        return None

    def _resolve_state_export_contract(
        self,
        *,
        call: Mapping[str, Any],
        result: InvocationResult,
        contract: ToolCallContract,
    ) -> ToolCallContract:
        if result.transition.changed and self.contracts.generator is not None:
            state_key = stable_id(
                "parallel_state_repair", [call["id"], contract.revision]
            )
            self._pending_state_repairs[state_key] = (
                dict(call),
                result,
                contract,
            )
        return contract

    def _trial_budget_available(self, needed: int) -> bool:
        # Trials can cross frozen rounds, so they are not reserved atomically
        # inside the search state machine.
        del needed
        return self.stop_reason == "completed"

    def _task_start_budget_available(self, task: SearchTask) -> bool:
        del task
        return self.stop_reason == "completed"

    def _prepare_tuple_requeue(self, task: SearchTask) -> None:
        # The live queue owns this mutable task; do not put SQLite in the
        # no-call fill path.
        del task
        current = asyncio.current_task()
        if (
            self._bulk_run_active
            and current is not None
            and current not in self._bulk_invocation_requested
        ):
            self._dispatch_slot -= 1
            self._dispatch_slot_returned.add(current)

    def _state_trial_id_for_call(
        self,
        *,
        state: SearchState,
        tool: str,
        args: Mapping[str, Any],
        revision: int,
    ) -> str | None:
        if state.producer_call is None or state.producer_before_handle is None:
            return None
        return stable_id("state_trial", [
            state.producer_call, tool, dict(args), revision,
        ])

    def _state_trial_id_for_task(
        self,
        task: SearchTask,
        args: Mapping[str, Any],
    ) -> str | None:
        compatible = self._compatible_incomplete_contract(task)
        return self._state_trial_id_for_call(
            state=self.states[task.state_id],
            tool=task.tool,
            args=args,
            revision=(
                compatible[0].revision if compatible is not None
                else task.revision
            ),
        )

    def _claim_state_trial(self, task_id: str, trial_id: str) -> bool:
        owner = self._state_trial_owners.get(trial_id)
        if owner is not None and owner != task_id:
            return False
        previous = self._task_state_trial_claims.get(task_id)
        if previous is not None and previous != trial_id:
            raise RuntimeError(
                f"task {task_id} already owns state trial {previous}"
            )
        self._state_trial_owners[trial_id] = task_id
        self._task_state_trial_claims[task_id] = trial_id
        return True

    def _release_state_trial_claim(self, task_id: str) -> None:
        trial_id = self._task_state_trial_claims.pop(task_id, None)
        if trial_id is not None and self._state_trial_owners.get(trial_id) == task_id:
            self._state_trial_owners.pop(trial_id, None)

    def _rebuild_state_trial_owners(self) -> None:
        for task_id, binding_key in self._incomplete_task_bindings.items():
            call = self.store.call_by_binding(binding_key)
            if call is None:
                continue
            state = self.states.get(str(call.get("state_id") or ""))
            if state is None:
                raise RuntimeError(
                    f"incomplete call has no state: {binding_key}"
                )
            producer = (
                self.store.call(state.producer_call)
                if state.producer_call is not None else None
            )
            if state.producer_call is not None and (
                producer is None
                or not bool(
                    producer.get("result", {})
                    .get("transition", {})
                    .get("changed")
                )
            ):
                raise RuntimeError(
                    f"incomplete state has no changed producer: {state.id}"
                )
            entry = self._priority_entries.get(task_id)
            compatible = (
                self._compatible_incomplete_contract(entry.task)
                if entry is not None else None
            )
            trial_id = self._state_trial_id_for_call(
                state=state,
                tool=str(call.get("tool") or ""),
                args=dict(call.get("args") or {}),
                revision=(
                    compatible[0].revision if compatible is not None
                    else int(call.get("contract_revision") or 0)
                ),
            )
            if trial_id is not None and not self._claim_state_trial(
                task_id, trial_id
            ):
                raise RuntimeError(
                    "multiple incomplete tasks own one state trial: "
                    f"{trial_id}"
                )

    def _defer_round_binding_task(self, task: SearchTask) -> bool:
        entry = self._leased_entries.pop(task.id, None)
        if entry is None:
            raise RuntimeError("round binding deferral has no leased priority entry")
        self._round_binding_deferred_entries[task.id] = entry
        return False

    def _reserve_round_task_binding(self, task: SearchTask) -> bool:
        """Admit one unfinished exact binding and state trial to the round."""

        incomplete = self._incomplete_call_for_task(task)
        if incomplete is not None:
            binding_key, call = incomplete
            args = dict(call.get("args") or {})
        else:
            combination = self._combination(task)
            if combination is None:
                self._round_task_combinations[task.id] = None
                return True
            args, _ = combination
            binding_key = stable_id("binding", [
                task.state_id,
                task.tool,
                task.revision,
                task.template_index,
                args,
            ])
        incomplete_owner = self._incomplete_binding_owners.get(binding_key)
        if incomplete_owner is not None and incomplete_owner != task.id:
            return self._defer_round_binding_task(task)

        committed = self.store.seen_binding(binding_key)
        round_owner = self._round_binding_owners.get(binding_key)
        if not committed and round_owner is not None and round_owner != task.id:
            return self._defer_round_binding_task(task)

        trial_id = (
            self._state_trial_id_for_task(task, args)
            if not committed or incomplete_owner == task.id
            else None
        )
        if (
            trial_id is not None
            and self._state_trial_owners.get(trial_id) not in (None, task.id)
        ):
            return self._defer_round_binding_task(task)

        if not committed and round_owner is None:
            self._round_binding_owners[binding_key] = task.id
        if trial_id is not None and not self._claim_state_trial(task.id, trial_id):
            raise RuntimeError("state trial owner changed during one selection")
        if incomplete is None:
            self._round_task_combinations[task.id] = combination
        return True

    def _release_round_binding_reservations(self) -> None:
        """Restore duplicate tasks only after the owning round is reduced."""

        deferred = sorted(
            self._round_binding_deferred_entries.values(),
            key=lambda entry: (
                entry.enqueue_sequence,
                entry.tie,
                entry.task.id,
            ),
        )
        self._round_binding_owners.clear()
        self._round_binding_deferred_entries.clear()
        for entry in deferred:
            if not entry.task.completed:
                self._put_priority_entry(entry)

    def _incomplete_binding(self, task_id: str) -> str | None:
        return self._incomplete_task_bindings.get(task_id)

    def _mark_incomplete_binding(self, task_id: str, binding_key: str) -> None:
        owner = self._incomplete_binding_owners.get(binding_key)
        if owner is not None and owner != task_id:
            raise RuntimeError(
                f"primary binding {binding_key} is already owned by {owner}"
            )
        previous = self._incomplete_task_bindings.get(task_id)
        if previous is not None and previous != binding_key:
            raise RuntimeError(
                f"task {task_id} already owns primary binding {previous}"
            )
        self._incomplete_task_bindings[task_id] = binding_key
        self._incomplete_binding_owners[binding_key] = task_id
        if not self._bulk_run_active:
            self._persist_parallel_meta()

    def _try_mark_incomplete_binding(
        self,
        task: SearchTask,
        binding_key: str,
        args: Mapping[str, Any],
    ) -> bool:
        """Claim the binding reached after skipping any committed calls."""

        incomplete_owner = self._incomplete_binding_owners.get(binding_key)
        round_owner = self._round_binding_owners.get(binding_key)
        trial_id = self._state_trial_id_for_task(task, args)
        trial_owner = (
            self._state_trial_owners.get(trial_id)
            if trial_id is not None else None
        )
        if (
            incomplete_owner not in (None, task.id)
            or round_owner not in (None, task.id)
            or trial_owner not in (None, task.id)
        ):
            # The task produced no actual invocation, so its provisional lane
            # selection must not consume one of the round's 80/20 call slots.
            current = (
                asyncio.current_task() if self._bulk_run_active else None
            )
            if (
                current is None
                or current not in self._bulk_invocation_requested
            ):
                self._dispatch_slot -= 1
                if current is not None:
                    self._dispatch_slot_returned.add(current)
            self._defer_round_binding_task(task)
            return False
        if round_owner is None:
            self._round_binding_owners[binding_key] = task.id
        if trial_id is not None and not self._claim_state_trial(task.id, trial_id):
            raise RuntimeError("state trial owner changed before worker dispatch")
        self._mark_incomplete_binding(task.id, binding_key)
        return True

    def _clear_incomplete_binding(self, task_id: str) -> None:
        # Normal state_after completion already releases its replay. This is a
        # defensive completion invariant for a source+after pair that takes
        # the no-invocation fast path.
        self._drop_retained_state_replay(task_id, discard=True)
        binding_key = self._incomplete_task_bindings.pop(task_id, None)
        if (
            binding_key is not None
            and self._incomplete_binding_owners.get(binding_key) == task_id
        ):
            self._incomplete_binding_owners.pop(binding_key, None)
        self._release_state_trial_claim(task_id)
        if not self._bulk_run_active:
            self._persist_parallel_meta()

    def _release_task(
        self, task_id: str, *, restore_leased: bool = False
    ) -> None:
        self._leased_task_ids.discard(task_id)
        leased_entry = self._leased_entries.pop(task_id, None)
        task = self._deferred_tasks.pop(task_id, None)
        if task is not None:
            if (
                restore_leased
                and leased_entry is not None
                and task_id in self._incomplete_task_bindings
            ):
                # Keep the exact concrete tuple's score while its committed
                # call is still completing repeat/ablation/state trials.
                self._put_priority_entry(_PriorityEntry(
                    task=task,
                    task_class=leased_entry.task_class,
                    enqueue_sequence=leased_entry.enqueue_sequence,
                    semantic_missing=leased_entry.semantic_missing,
                    semantic_score=leased_entry.semantic_score,
                    tie=leased_entry.tie,
                ))
            else:
                self._schedule_task(task)
        elif (
            restore_leased
            and leased_entry is not None
            and not leased_entry.task.completed
        ):
            # Pausing at an epoch boundary must not alter discovery order.
            # Restore the exact entry instead of rescoring or reenqueuing it.
            self._put_priority_entry(leased_entry)

    def _coalesced_epoch_feedback(
        self, rows: Mapping[str, Mapping[str, Any]]
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Merge same-tool epoch feedback into one deterministic prompt."""

        ordered = [dict(row) for _, row in sorted(rows.items())]
        literal_sources = tuple(sorted({
            str(row["source_call_id"])
            for row in ordered
            if str(row.get("kind") or "") == "literal"
            and row.get("source_call_id")
        }))
        if len(ordered) == 1:
            return dict(ordered[0].get("feedback") or {}), literal_sources
        return ({
            "event": "epoch_feedback_batch",
            "items": [
                {
                    "kind": str(row.get("kind") or "outcome"),
                    "feedback": dict(row.get("feedback") or {}),
                }
                for row in ordered
            ],
            "conclusion": (
                "repair_all_compatible_contract_defects_in_one_revision"
            ),
        }, literal_sources)

    def _effective_llm_calls(self, tool: str | None = None) -> int:
        if tool is not None:
            return self.contracts.llm_calls(tool)
        return self.contracts.total_llm_calls()

    def _repair_contracts_concurrently(
        self,
        requests: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, tuple[ToolCallContract | None, ToolCallContract | None]]:
        """Generate across tools concurrently, then publish deterministically."""

        if not requests:
            return {}
        # Instance-installed repair functions are test/adaptor hooks that may
        # mutate the active set; preserve their serial behavior.
        if (
            self.contracts.generator is None
            or self.contracts.__dict__.get("repair") is not None
        ):
            return {
                tool: (
                    previous := self.contracts.get(tool),
                    self._repair_contract(tool, requests[tool]),
                )
                for tool in sorted(requests)
            }

        repair_calls_used = max(
            0,
            self.contracts.total_llm_calls() - self.initial_llm_calls,
        )
        repair_limit = self.config.budget.max_contract_repair_calls
        repair_remaining = (
            sum(
                max(0, MAX_LLM_CALLS_PER_TOOL - self._effective_llm_calls(tool))
                for tool in requests
            )
            if repair_limit is None
            else max(0, repair_limit - repair_calls_used)
        )
        total_calls = self.contracts.total_llm_calls()
        effective_total_calls = self._effective_llm_calls()
        external_remaining = (
            repair_remaining
            if self.contracts.max_total_llm_calls is None
            else max(
                0, self.contracts.max_total_llm_calls - effective_total_calls
            )
        )
        available = min(repair_remaining, external_remaining)
        planned: dict[
            str,
            tuple[ToolCallContract, int, Mapping[str, Any]],
        ] = {}
        results: dict[
            str, tuple[ToolCallContract | None, ToolCallContract | None]
        ] = {}
        for tool in sorted(requests):
            previous = self.contracts.get(tool)
            if previous is None:
                results[tool] = (None, None)
                continue
            per_tool_remaining = max(
                0, MAX_LLM_CALLS_PER_TOOL - self._effective_llm_calls(tool)
            )
            allowance = min(per_tool_remaining, available)
            if allowance <= 0:
                if available <= 0 and per_tool_remaining > 0:
                    self.contract_repairs_budget_rejected += 1
                else:
                    self.contract_repair_requests += 1
                    self.contract_repairs_rejected += 1
                results[tool] = (previous, previous)
                continue
            self.contract_repair_requests += 1
            planned[tool] = (
                previous,
                total_calls + allowance,
                requests[tool],
            )
            available -= allowance

        generated: dict[
            str, tuple[OnlineContractSet, ToolCallContract | None]
        ] = {}
        if planned:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(
                    self.parallel.contract_repair_concurrency,
                    len(planned),
                )
            ) as pool:
                futures = {
                    pool.submit(
                        self._generate_contract_repair_candidate,
                        tool,
                        feedback,
                        max_total_llm_calls=max_total,
                    ): tool
                    for tool, (_, max_total, feedback) in planned.items()
                }
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=self.parallel.heartbeat_interval_seconds,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        generated[futures[future]] = future.result()
                    if pending:
                        self._emit_due_heartbeat()

        for tool in sorted(planned):
            previous = planned[tool][0]
            candidate_set, repaired = generated[tool]
            published = self._publish_contract_repair_candidate(
                tool=tool,
                previous=previous,
                candidate_set=candidate_set,
                repaired=repaired,
            )
            results[tool] = (previous, published)
            self._emit_due_heartbeat()
            if self.stop_reason == "field_embedding_error":
                break
        self._persist_contract_repair_meta()
        return results

    def _apply_pending_repairs(self) -> None:
        tail_started = system_monotonic()
        state_rows_by_tool: dict[
            str,
            list[tuple[str, dict[str, Any], InvocationResult, ToolCallContract]],
        ] = defaultdict(list)
        for key in sorted(self._pending_state_repairs):
            call, result, contract = self._pending_state_repairs[key]
            state_rows_by_tool[str(call.get("tool") or "")].append(
                (key, call, result, contract)
            )

        requests: dict[str, Mapping[str, Any]] = {}
        literal_sources: dict[str, tuple[str, ...]] = {}
        state_repair_tools: set[str] = set()
        # State-export feedback is normalization-sensitive, so it gets the one
        # repair slot for that tool in this epoch. Other feedback for the same
        # tool remains buffered for the following epoch.
        for tool, rows in sorted(state_rows_by_tool.items()):
            active = self.contracts.get(tool)
            if active is None:
                continue
            unmatched = [
                result
                for _, _, result, _ in rows
                if not self._contract_has_matching_state_exports(result, active)
            ]
            if unmatched:
                state_repair_tools.add(tool)
                requests[tool] = self._state_export_repair_feedback(
                    unmatched[0], active
                )
        for tool, rows in sorted(self._pending_feedback.items()):
            if tool in state_repair_tools:
                continue
            feedback, sources = self._coalesced_epoch_feedback(rows)
            requests[tool] = feedback
            literal_sources[tool] = sources

        repair_started = system_monotonic()
        repaired = self._repair_contracts_concurrently(requests)
        repair_seconds = system_monotonic() - repair_started
        accepted_tools = {
            tool
            for tool, (previous, active) in repaired.items()
            if (
                previous is not None
                and active is not None
                and active.revision > previous.revision
            )
        }
        if accepted_tools:
            self._observation_task_blueprints.clear()
            self._observation_frontier_specs.clear()
            self._observation_frontier_blueprints.clear()
            self._parallel_contract_revision_key = None
            self._semantic_task_score_cache.clear()
            self._argument_choices_cache.clear()
            self._observation_choices_cache.clear()
            self._parameter_observation_scores_cache.clear()
        enqueue_started = system_monotonic()
        for tool in sorted(repaired):
            previous, active = repaired[tool]
            if tool in accepted_tools:
                assert active is not None
                self._enqueue_tool_revision(tool, active.revision)
                for source_call_id in literal_sources.get(tool, ()):
                    source_call = self.store.call(source_call_id)
                    if source_call is not None:
                        self._record_literal_baseline(source_call)
            if self.stop_reason == "field_embedding_error":
                return
            self._emit_due_heartbeat()
        enqueue_seconds = system_monotonic() - enqueue_started

        processed_feedback_tools = set(self._pending_feedback) - state_repair_tools
        for tool in processed_feedback_tools:
            self._pending_feedback.pop(tool, None)
        projection_started = system_monotonic()
        for tool, rows in sorted(state_rows_by_tool.items()):
            active = self.contracts.get(tool)
            if active is None:
                continue
            for _, call, result, _ in rows:
                parent = self.states.get(str(call.get("state_id")))
                saved_handle = call.get("saved_state_handle")
                selected_template_index = (
                    self._matching_selected_template_index(active, call)
                )
                if (
                    parent is not None
                    and saved_handle
                    and active.revision
                    > int(call.get("contract_revision") or 0)
                    and selected_template_index is not None
                ):
                    # Re-project already successful transitions immediately;
                    # new state exports need no source replay.  The persisted
                    # call keeps its executed revision; this active-revision
                    # view is safe only while the selected template is exact.
                    projected_call = {
                        **call,
                        "contract_revision": active.revision,
                        "template_index": selected_template_index,
                    }
                    self._create_state(
                        parent,
                        projected_call,
                        result,
                        active,
                        handle=str(saved_handle),
                    )
        # Contract publication makes old-revision tasks stale immediately.
        # Only incomplete tasks can own transient primary/replay handles, so a
        # bounded scan of that set releases superseded state without walking
        # or rebuilding the global queue. The queue entries themselves remain
        # lazily discarded by _pop_priority_task.
        for task_id, binding_key in tuple(self._incomplete_task_bindings.items()):
            call = self.store.call_by_binding(binding_key)
            entry = (
                self._priority_entries.get(task_id)
                or self._leased_entries.get(task_id)
                or self._round_binding_deferred_entries.get(task_id)
            )
            tool = str(
                (call or {}).get("tool")
                or (entry.task.tool if entry is not None else "")
            )
            revision = int(
                (call or {}).get("contract_revision")
                or (entry.task.revision if entry is not None else 0)
            )
            active = self.contracts.get(tool) if tool in accepted_tools else None
            if (
                active is not None
                and active.revision > revision
            ):
                compatible = (
                    self._compatible_incomplete_contract(entry.task)
                    if entry is not None else None
                )
                if compatible is None:
                    self._discard_incomplete_task_state(task_id)
                else:
                    # The primary call remains exact, but any
                    # partially executed state trial belongs to the old
                    # revision. Release only that transient replay/claim so
                    # the active revision can start its own trial identity.
                    self._drop_retained_state_replay(task_id, discard=True)
                    self._release_state_trial_claim(task_id)
        self._pending_state_repairs.clear()
        self._last_epoch_tail_timings = {
            "repair_generation_publish_seconds": repair_seconds,
            "revision_enqueue_seconds": enqueue_seconds,
            "state_projection_seconds": (
                system_monotonic() - projection_started
            ),
            "apply_pending_repairs_seconds": (
                system_monotonic() - tail_started
            ),
        }

    def _rerank_priority_queue(self) -> None:
        """Recompute semantic scores after epoch-only contract publication."""

        current = list(self._priority_entries.values())
        self._priority_entries.clear()
        self._semantic_heap.clear()
        for heap in self._fair_heaps.values():
            heap.clear()
        for entry in current:
            task = entry.task
            if task.completed:
                continue
            score = self._task_semantic_score(task)
            self._put_priority_entry(_PriorityEntry(
                task=task,
                task_class=entry.task_class,
                enqueue_sequence=entry.enqueue_sequence,
                semantic_missing=1 if score is None else 0,
                semantic_score=float(score or 0.0),
                tie=entry.tie,
            ))

    def _storage_bytes(self) -> int:
        total = 0
        visited: set[tuple[int, int]] = set()
        paths = self.parallel.storage_paths or (self.store.path,)
        for root in paths:
            path = Path(root)
            candidates = (
                chain((path,), path.rglob("*")) if path.is_dir() else (path,)
            )
            for candidate in candidates:
                try:
                    metadata = candidate.lstat()
                except OSError:
                    continue
                identity = (int(metadata.st_dev), int(metadata.st_ino))
                if identity in visited:
                    continue
                visited.add(identity)
                if stat.S_ISREG(metadata.st_mode):
                    total += int(metadata.st_size)
        # This is the same traversal used for the storage budget.  Keep its
        # unique-inode count for telemetry; never add a second filesystem walk.
        self._last_storage_inodes = len(visited)
        return total

    def _refresh_storage_bytes(self, *, force: bool = False) -> int:
        """Return cached usage unless a progress boundary forces a scan."""

        if force:
            self._last_storage_bytes = self._storage_bytes()
            # Record completion, not scan start: a scan can be slower than a
            # heartbeat interval, and the next heartbeat should not rescan
            # the worker state tree immediately.
            self._last_storage_check = system_monotonic()
        return self._last_storage_bytes

    def _queue_sizes(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entry in self._priority_entries.values():
            counts[entry.task_class] += 1
        for entry in self._leased_entries.values():
            counts[entry.task_class] += 1
        lazy: dict[str, int] = defaultdict(int)
        for frontier in self._lazy_frontiers.values():
            lazy[frontier.task_class] += max(
                0, frontier.count - frontier.next_index
            )
        return {
            task_class: int(counts.get(task_class, 0)) + int(lazy.get(task_class, 0))
            for task_class in ("base", "event", "continuation", "provider")
        }

    def _committed_usage(self, *, storage_bytes: int | None = None) -> dict[str, Any]:
        return {
            "calls": self.calls_used,
            "returned_calls": self.store.count("calls"),
            "states": len(self.states),
            "external_model_calls": self._effective_llm_calls(),
            "contract_repair_llm_calls": max(
                0,
                self.contracts.total_llm_calls() - self.initial_llm_calls,
            ),
            "contract_repair_requests": self.contract_repair_requests,
            "contract_repairs_accepted": self.contract_repairs_accepted,
            "contract_repairs_rejected": self.contract_repairs_rejected,
            "contract_repairs_budget_rejected": (
                self.contract_repairs_budget_rejected
            ),
            "retries": self._worker_retries,
            "storage_bytes": (
                self._storage_bytes() if storage_bytes is None else storage_bytes
            ),
            "storage_inodes": self._last_storage_inodes,
            "budget_violations": sorted(self.budget_violations),
        }

    def _reservation_observation(self) -> dict[str, int]:
        """Return a compact observable summary of in-memory reservations.

        These reservations exist only while one frozen round is in memory.
        """

        if not self._outstanding_reservations:
            return {}
        return {
            "bulk_uncommitted": sum(self._outstanding_reservations.values()),
        }

    def _parallel_meta_values(self) -> dict[str, Any]:
        storage_bytes = self._last_storage_bytes
        return {
            "parallel_worker_dispatches": self._worker_dispatches,
            "parallel_worker_retries": self._worker_retries,
            "parallel_worker_terminal_failures": self._worker_terminal_failures,
            "parallel_epochs_completed": self._epochs_completed,
            "parallel_rounds_completed": self._rounds_completed,
            "parallel_max_epoch_tail_seconds": self._max_epoch_tail_seconds,
            "parallel_epoch_tail_seconds_total": self._epoch_tail_seconds_total,
            "parallel_work_items_completed": self._work_items_completed,
            "calls_used": self.calls_used,
            "elapsed_seconds": self._elapsed(),
            "observation_tasks_pruned_after_success": (
                self.observation_tasks_pruned_after_success
            ),
            "budget_violations": sorted(self.budget_violations),
            "parallel_committed_usage": self._committed_usage(
                storage_bytes=storage_bytes
            ),
            "parallel_trial_binding_reuses": self._trial_binding_reuses,
        }

    def _persist_parallel_meta(self) -> None:
        with self.store.batch():
            values = self._parallel_meta_values()
            if self._last_worker_error is not None:
                values["parallel_last_worker_error"] = self._last_worker_error
            self.store.set_meta_many(values)
        self._stale_task_updates.clear()

    def _telemetry(
        self,
        event: str,
        *,
        inflight: int = 0,
    ) -> None:
        if self.parallel.telemetry_path is None:
            return
        path = self.parallel.telemetry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Heartbeats stay lightweight; recursive storage accounting is
        # refreshed separately on a 20-minute cadence.
        storage_bytes = self._last_storage_bytes
        queue_sizes = self._queue_sizes()
        failures = {
            "terminal_work_items": self._worker_terminal_failures,
            "completed_work_items": self._work_items_completed,
            "worker_retries": self._worker_retries,
            "rate": (
                self._worker_terminal_failures / max(1, self._work_items_completed)
            ),
        }
        memory_scale = 1024 if os.name == "posix" else 1
        row = {
            "event": event,
            "elapsed_seconds": round(self._elapsed(), 3),
            "calls": self.calls_used,
            "states": len(self.states),
            "observation_tasks_pruned_after_success": (
                self.observation_tasks_pruned_after_success
            ),
            "scheduled_tasks": sum(queue_sizes.values()),
            "materialized_tasks": len(self._priority_entries),
            "lazy_frontier_generators": len(self._lazy_frontiers),
            "inflight": inflight,
            "active_workers": inflight,
            "epochs_completed": self._epochs_completed,
            "epoch": self._epochs_completed,
            "round": self._rounds_completed,
            "phase": self._round_phase,
            "round_capacity": self.parallel.round_capacity_for(
                self._rounds_completed
            ),
            "last_round_timings": dict(self._last_round_timings),
            "last_epoch_tail_seconds": self._last_epoch_tail_seconds,
            "last_epoch_tail_timings": dict(self._last_epoch_tail_timings),
            "max_epoch_tail_seconds": self._max_epoch_tail_seconds,
            "epoch_tail_seconds_total": self._epoch_tail_seconds_total,
            "worker_dispatches": self._worker_dispatches,
            "returned_invocations": self.store.count("calls"),
            "trial_binding_reuses": self._trial_binding_reuses,
            "worker_retries": self._worker_retries,
            "worker_terminal_failures": self._worker_terminal_failures,
            "committed_usage": self._committed_usage(storage_bytes=storage_bytes),
            "outstanding_reservations": self._reservation_observation(),
            "queue_sizes": queue_sizes,
            "failures": failures,
            "coordinator_error": self._last_coordinator_error,
            "memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * memory_scale,
            "observation_store_path": str(self.store.path),
            "storage_bytes": storage_bytes,
            "storage_inodes": self._last_storage_inodes,
            "stop_reason": self.stop_reason,
        }
        with path.open("a", encoding="utf-8") as handle:
            for event in self._pending_edge_log_events:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._pending_edge_log_events.clear()
        self._last_telemetry = system_monotonic()

    def _record_discovered_edge(
        self,
        kind: str,
        row: Mapping[str, Any],
    ) -> None:
        path = self.parallel.telemetry_path
        if path is None:
            return
        if kind == "input":
            if (
                str(row.get("source_location")) == "arguments"
                or str(row["source"]) == str(row["target"])
            ):
                return
            provenance = {
                "kind": "input",
                "source": str(row["source"]),
                "target": str(row["target"]),
                "target_parameter": str(row["target_parameter"]),
                "source_location": str(row["source_location"]),
                "source_export": str(row["source_export"]),
                "source_path": str(row["source_path"]),
            }
        else:
            provenance = {
                "kind": "state",
                "source": str(row["source"]),
                "target": str(row["target"]),
                "target_parameter": None,
                "state_changes": list(row.get("state_changes") or ()),
            }
        key = canonical_json(provenance)
        if key in self._logged_edge_provenances:
            return
        self._logged_edge_provenances.add(key)
        ordinals: list[int] = []
        for field in ("success_calls", "control_calls", "evidence_calls"):
            for call_id in row.get(field) or ():
                recorded_call = self.store.call(str(call_id))
                if (
                    recorded_call is not None
                    and recorded_call.get("ordinal") is not None
                ):
                    ordinals.append(int(recorded_call["ordinal"]))
        event = {
            "event": "edge_discovered",
            "call_ordinal": max(ordinals, default=self.calls_used),
            "elapsed_seconds": round(self._elapsed(), 3),
            "required": bool(row.get("required", kind == "state")),
            **provenance,
        }
        self._pending_edge_log_events.append(event)

    def _record_first_tool_success(self, call: Mapping[str, Any]) -> None:
        if self.parallel.telemetry_path is None:
            return
        tool = str(call["tool"])
        if tool in self._logged_success_tools:
            return
        self._logged_success_tools.add(tool)
        result = dict(call.get("result") or {})
        transition = dict(result.get("transition") or {})
        state = self.states.get(str(call.get("state_id") or ""))
        self._pending_edge_log_events.append({
            "event": "tool_first_success",
            "tool": tool,
            "call_ordinal": int(call.get("ordinal") or self.calls_used),
            "elapsed_seconds": round(self._elapsed(), 3),
            "reason": str(call.get("reason") or ""),
            "state_changed": bool(transition.get("changed")),
            "state_depth": state.depth if state is not None else None,
            "state_producer": (
                self._state_producer_tool(state) if state is not None else None
            ),
        })

    def _emit_due_heartbeat(self) -> None:
        """Keep telemetry live while synchronous coordinator commits are busy.

        A large worker batch can make many coordinator callbacks ready at once.
        Each callback performs serial SQLite post-processing, so the outer
        ``asyncio.wait`` timeout may not regain control for several minutes.
        Queue insertion and invocation pre-commit are safe coordinator-owned
        progress points from which to emit an overdue heartbeat.
        """

        if getattr(self.store, "_batch_depth", 0) > 0:
            return
        last_telemetry = getattr(self, "_last_telemetry", None)
        if last_telemetry is None or (
            system_monotonic() - last_telemetry
            < self.parallel.heartbeat_interval_seconds
        ):
            return
        self._telemetry(
            "heartbeat",
            inflight=self._active_inflight_count,
        )

    def _check_non_call_budgets(self, *, force_storage: bool = False) -> None:
        budget = self.config.budget
        if (
            budget.max_external_model_calls is not None
            and self._effective_llm_calls() > budget.max_external_model_calls
        ):
            self._record_budget_violation("max_external_model_calls")
        if (
            budget.max_contract_repair_calls is not None
            and max(
                0,
                self.contracts.total_llm_calls() - self.initial_llm_calls,
            ) > budget.max_contract_repair_calls
        ):
            self._record_budget_violation("max_contract_repair_calls")
        if (
            budget.max_retries is not None
            and self._worker_retries > budget.max_retries
        ):
            self._record_budget_violation("max_retries")
        storage_due = (
            system_monotonic() - self._last_storage_check
            >= STORAGE_CHECK_INTERVAL_SECONDS
        )
        if force_storage or (
            budget.max_storage_bytes is not None and storage_due
        ):
            self._refresh_storage_bytes(force=True)
        if (
            budget.max_storage_bytes is not None
            and self._last_storage_bytes > budget.max_storage_bytes
        ):
            self._record_budget_violation("max_storage_bytes")

    def _prewarm_observation_choice_domains(self) -> None:
        """Build immutable parameter domains during the excluded warmup round."""

        if not self.states:
            return
        state = next(iter(self.states.values()))
        for tool, definition in self.definitions.items():
            contract = self.contracts.get(tool)
            if contract is None:
                continue
            for parameter in definition.parameters:
                for template in contract.templates:
                    if template.expected != "success":
                        continue
                    self._choices(
                        state,
                        tool,
                        parameter,
                        template,
                        allow_observations=True,
                        allow_seed_replacements=True,
                        preferred_observation_call=None,
                    )
                    break

    def _record_work_item_completion(self, *, terminal_failure: bool) -> None:
        self._work_items_completed += 1
        if terminal_failure:
            self._worker_terminal_failures += 1
        failure_rate = (
            self._worker_terminal_failures / max(1, self._work_items_completed)
        )
        if failure_rate > self.parallel.worker_failure_threshold:
            self.stop_reason = "worker_failure_rate"

    def _available_dispatch_calls(self) -> int:
        inflight_reserved = sum(self._outstanding_reservations.values())
        global_available = (
            self.config.budget.max_calls
            - self.calls_used
            - inflight_reserved
        )
        epoch_available = (
            self.parallel.epoch_calls
            - (self.calls_used - self._epoch_start_calls)
            - inflight_reserved
        )
        return min(global_available, epoch_available)

    def _finish_contract_epoch(self, event: str) -> None:
        started = system_monotonic()
        self._round_phase = "epoch_tail"
        self._apply_pending_repairs()
        # _apply_pending_repairs invalidates affected caches before it enqueues
        # and scores the accepted revisions.  Do not clear those freshly built
        # scores a second time here; old-revision tasks are discarded lazily by
        # _pop_priority_task and unchanged-tool queue scores remain valid.
        self._last_epoch_tail_seconds = system_monotonic() - started
        self._max_epoch_tail_seconds = max(
            self._max_epoch_tail_seconds, self._last_epoch_tail_seconds
        )
        self._epoch_tail_seconds_total += self._last_epoch_tail_seconds
        self._epochs_completed += 1
        self._epoch_start_calls = self.calls_used
        self._epoch_boundary_pending = False
        self._telemetry(event, inflight=0)
        # Full rounds are also complete epochs and otherwise jump
        # directly to the next fill. Check cached/non-call budgets here; an
        # exact storage scan occurs only when the 20-minute cadence is due.
        self._check_non_call_budgets()

    def _commit_finished_future(
        self,
        future: asyncio.Task[None],
        inflight: dict[asyncio.Task[None], tuple[str, int]],
    ) -> None:
        task_id, _ = inflight.pop(future)
        self._finish_future_dispatch_slot(future)
        self._active_inflight_count = len(inflight)
        self._future_task_ids.pop(future, None)
        self._completion_order.pop(future, None)
        self._outstanding_reservations.pop(task_id, None)
        self._release_task(
            task_id,
            restore_leased=(task_id in self._incomplete_task_bindings),
        )
        terminal_failure = future in self._terminal_failure_futures
        self._terminal_failure_futures.discard(future)
        self._record_work_item_completion(terminal_failure=terminal_failure)
        exception = future.exception()
        if exception is not None:
            self._last_coordinator_error = {
                "type": type(exception).__name__,
                "message": str(exception)[:2000],
                "task_id": task_id,
            }
            self.store.set_meta(
                "parallel_coordinator_error",
                self._last_coordinator_error,
            )
            self.stop_reason = "coordinator_error"
            raise exception

    def _finish_future_dispatch_slot(self, future: asyncio.Task[Any]) -> None:
        """Return a provisional queue lane when no invocation was requested."""

        if future in self._dispatch_slot_returned:
            self._dispatch_slot_returned.discard(future)
        elif future not in self._bulk_invocation_requested:
            self._dispatch_slot -= 1
        self._bulk_invocation_requested.discard(future)

    def _cleanup_bulk_tasks(
        self, active: dict[asyncio.Task[None], str]
    ) -> None:
        completed = getattr(self, "_bulk_done_futures", None)
        if completed is None:
            done = [future for future in active if future.done()]
        else:
            done = []
            while completed:
                done.append(completed.popleft())
        for future in done:
            task_id = active.pop(future, None)
            if task_id is None:
                continue
            self._finish_future_dispatch_slot(future)
            self._future_task_ids.pop(future, None)
            self._invocation_ordinals.pop(future, None)
            self._preclassified_outcomes.pop(future, None)
            self._pending_by_owner.pop(future, None)
            self._outstanding_reservations.pop(task_id, None)
            self._release_task(
                task_id,
                restore_leased=(task_id in self._incomplete_task_bindings),
            )
            if future.cancelled():
                continue
            self._terminal_failure_futures.discard(future)
            # Bulk work-item accounting is per requested runtime invocation,
            # not per discovery coroutine (one coroutine can request several
            # invocations across successive rounds).
            exception = future.exception()
            if exception is not None:
                self._last_coordinator_error = {
                    "type": type(exception).__name__,
                    "message": str(exception)[:2000],
                    "task_id": task_id,
                }
                self.store.set_meta(
                    "parallel_coordinator_error",
                    self._last_coordinator_error,
                )
                self.stop_reason = "coordinator_error"

    async def _fill_bulk_requests(
        self, active: dict[asyncio.Task[None], str]
    ) -> None:
        global_remaining = (
            self.config.budget.max_calls
            - self.calls_used
        )
        epoch_remaining = self.parallel.epoch_calls - (
            self.calls_used - self._epoch_start_calls
        )
        capacity = min(
            self.parallel.round_capacity_for(self._rounds_completed),
            global_remaining,
            epoch_remaining,
        )
        if capacity <= 0:
            self._last_fill_timings = {
                "seconds": 0.0,
                "selected": 0,
                "pending_invocations": len(self._pending_invocations),
            }
            return
        fill_started = system_monotonic()
        selected_count = 0
        selection_seconds = 0.0
        frontier_seconds = 0.0
        coroutine_seconds = 0.0
        while (
            self.stop_reason == "completed"
            and len(self._pending_invocations) < capacity
        ):
            self._cleanup_bulk_tasks(active)
            selection_started = system_monotonic()
            selected = (
                self._select_priority_task()
                if len(active) < capacity else None
            )
            selection_seconds += system_monotonic() - selection_started
            if selected is not None:
                if not self._reserve_round_task_binding(selected):
                    # This priority slot did not produce an invocation. Reuse
                    # the same 80/20 lane position for the next unique task and
                    # keep filling to the actual-call capacity.
                    self._dispatch_slot -= 1
                    continue
                self._leased_task_ids.add(selected.id)
                self._outstanding_reservations[selected.id] = 1
                frontier_id = self._frontier_task_ids.pop(selected.id, None)
                if frontier_id is not None:
                    frontier_started = system_monotonic()
                    self._advance_lazy_frontier(frontier_id)
                    frontier_seconds += system_monotonic() - frontier_started
                combination_prepared = (
                    selected.id in self._round_task_combinations
                )
                prepared_combination = self._round_task_combinations.pop(
                    selected.id, None
                )
                future = asyncio.create_task(self._run_task_async(
                    selected,
                    combination_prepared=combination_prepared,
                    prepared_combination=prepared_combination,
                ))
                active[future] = selected.id
                completed = getattr(self, "_bulk_done_futures", None)
                if completed is not None:
                    future.add_done_callback(completed.append)
                self._future_task_ids[future] = selected.id
                selected_count += 1
                coroutine_started = system_monotonic()
                await asyncio.sleep(0)
                coroutine_seconds += system_monotonic() - coroutine_started
                continue

            # A newly launched coroutine can traverse reused bindings before
            # reaching its next invocation suspension point.  Yield until each
            # active coroutine has either finished or registered a pending
            # invocation; otherwise the coordinator can mistake scheduler lag
            # for an empty queue (or busy-spin and starve those coroutines).
            all_waiting = bool(active) and all(
                future in self._pending_by_owner for future in active
            )
            if not active or all_waiting:
                break
            coroutine_started = system_monotonic()
            await asyncio.sleep(0)
            coroutine_seconds += system_monotonic() - coroutine_started
        self._last_fill_timings = {
            "seconds": round(system_monotonic() - fill_started, 6),
            "selection_seconds": round(selection_seconds, 6),
            "frontier_seconds": round(frontier_seconds, 6),
            "coroutine_seconds": round(coroutine_seconds, 6),
            "selected": selected_count,
            "pending_invocations": len(self._pending_invocations),
        }

    async def _invoke_pending_request(
        self,
        request: _PendingInvocation,
        ordinal: int,
    ) -> tuple[
        _PendingInvocation,
        tuple[InvocationResult, str | None, int, str | None] | None,
    ]:
        previous_worker: int | None = None
        current_ordinal = ordinal
        for attempt in range(self.parallel.worker_retries + 1):
            avoid_worker = previous_worker
            save_state_id = (
                stable_id(
                    "parallel_post_state_attempt",
                    [
                        request.binding_key,
                        request.reason,
                        request.task_id,
                        current_ordinal,
                    ],
                )
                if request.save_on_change else None
            )
            try:
                invoke_with_worker = getattr(
                    self.invocation_pool, "invoke_with_worker", None
                )
                if invoke_with_worker is None:
                    result, saved_handle = await self.invocation_pool.invoke(
                        state_handle=request.state.handle,
                        state_digest=request.state.digest,
                        tool=request.tool,
                        args=request.args,
                        save_state_id=save_state_id,
                    )
                    worker = None
                else:
                    result, saved_handle, worker = await invoke_with_worker(
                        state_handle=request.state.handle,
                        state_digest=request.state.digest,
                        tool=request.tool,
                        args=request.args,
                        save_state_id=save_state_id,
                        avoid_worker=avoid_worker,
                    )
                    previous_worker = worker
                return request, (result, saved_handle, current_ordinal, None)
            except Exception as exc:
                failed_worker = getattr(exc, "worker_id", None)
                if failed_worker is not None:
                    previous_worker = int(failed_worker)
                can_retry = attempt < self.parallel.worker_retries
                retry_limit = self.config.budget.max_retries
                if retry_limit is not None and self._worker_retries >= retry_limit:
                    can_retry = False
                    self._record_budget_violation("max_retries")
                epoch_remaining = self.parallel.epoch_calls - (
                    self.calls_used - self._epoch_start_calls
                )
                global_remaining = (
                    self.config.budget.max_calls
                    - self.calls_used
                )
                if min(epoch_remaining, global_remaining) <= 0:
                    can_retry = False
                if not can_retry:
                    self.store.set_meta("parallel_last_worker_error", {
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                        "tool": request.tool,
                        "state": request.state.id,
                    })
                    return request, None
                self._worker_retries += 1
                self._worker_dispatches += 1
                self.calls_used += 1
                current_ordinal = self.calls_used
        return request, None

    async def _invoke_pending_batch(
        self,
        requests: Sequence[_PendingInvocation],
        ordinals: Sequence[int],
        *,
        avoid_worker: int | None = None,
    ) -> list[
        tuple[
            _PendingInvocation,
            tuple[InvocationResult, str | None, int, str | None] | None,
            int | None,
        ]
    ]:
        invoke_batch = getattr(self.invocation_pool, "invoke_batch")
        payloads: list[dict[str, Any]] = []
        if len(requests) != len(ordinals):
            raise ValueError("worker batch requests/ordinals length mismatch")
        for request, ordinal in zip(requests, ordinals):
            contract = self.contracts.get(request.tool)
            if contract is None:
                raise RuntimeError("pending invocation has no active contract")
            payloads.append({
                "state_handle": request.state.handle,
                "state_digest": request.state.digest,
                "state_is_initial": (
                    request.state.depth == 0
                    and request.state.producer_call is None
                ),
                "tool": request.tool,
                "args": json_safe(request.args),
                "success_schema": json_safe(contract.success_schema),
                "failure_schema": json_safe(contract.failure_schema),
                "save_state_id": (
                    stable_id(
                        "parallel_post_state_attempt",
                        [
                            request.binding_key,
                            request.reason,
                            request.task_id,
                            ordinal,
                        ],
                    )
                    if request.save_on_change else None
                ),
            })

        def discard_batch_handles() -> None:
            for payload in payloads:
                handle = payload.get("save_state_id")
                if handle:
                    self._discard_unreferenced_state_handle(str(handle))

        try:
            returned, worker = await invoke_batch(
                payloads, avoid_worker=avoid_worker
            )
        except Exception as exc:
            discard_batch_handles()
            worker = getattr(exc, "worker_id", None)
            self._last_worker_error = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
                "batch_size": len(requests),
                "worker": worker,
            }
            return [(request, None, worker) for request in requests]
        actual_count = len(returned) if isinstance(returned, Sequence) else None
        if actual_count != len(requests):
            discard_batch_handles()
            self._last_worker_error = {
                "type": "WorkerBatchResultCountMismatch",
                "message": (
                    "worker batch result count did not match request count"
                ),
                "expected_count": len(requests),
                "actual_count": actual_count,
                "worker": worker,
            }
            return [(request, None, worker) for request in requests]
        rows: list[
            tuple[
                _PendingInvocation,
                tuple[InvocationResult, str | None, int, str | None] | None,
                int | None,
            ]
        ] = []
        for request, ordinal, raw in zip(requests, ordinals, returned):
            result, saved_handle, error, outcome = raw
            if result is None:
                self._last_worker_error = {
                    "type": str((error or {}).get("error_type") or "WorkerError"),
                    "message": str((error or {}).get("error") or "")[:1000],
                    "tool": request.tool,
                    "state": request.state.id,
                    "worker": worker,
                }
                rows.append((request, None, worker))
            else:
                rows.append((
                    request,
                    (result, saved_handle, ordinal, outcome),
                    worker,
                ))
        return rows

    async def _dispatch_bulk_round(
        self, active: dict[asyncio.Task[None], str]
    ) -> int:
        global_remaining = (
            self.config.budget.max_calls
            - self.calls_used
        )
        epoch_remaining = self.parallel.epoch_calls - (
            self.calls_used - self._epoch_start_calls
        )
        candidate_count = min(
            len(self._pending_invocations),
            self.parallel.round_capacity_for(self._rounds_completed),
            global_remaining,
            epoch_remaining,
        )
        if candidate_count <= 0:
            return 0
        round_requests = self._pending_invocations[:candidate_count]
        del self._pending_invocations[:candidate_count]
        selected: list[_PendingInvocation] = []
        primary_by_binding: dict[str, _PendingInvocation] = {}
        coalesced: list[tuple[_PendingInvocation, _PendingInvocation]] = []
        for request in round_requests:
            primary = primary_by_binding.get(request.binding_key)
            if primary is None:
                primary_by_binding[request.binding_key] = request
                selected.append(request)
            else:
                coalesced.append((request, primary))
        count = len(selected)
        round_started = system_monotonic()
        first_ordinal = self.calls_used + 1
        self.calls_used += count
        self._worker_dispatches += count
        self._outstanding_reservations = {
            request.task_id: 1
            for request in self._pending_invocations
            if request.task_id is not None
        }
        self._outstanding_reservations.update({
            request.task_id: 0
            for request in round_requests
            if request.task_id is not None
        })
        self._round_phase = "invoke"
        self._active_inflight_count = min(self.parallel.workers, count)
        invocation_started = system_monotonic()
        worker_retries_before_round = self._worker_retries
        retry_headroom_after_initial: int | None = None
        retry_terminal_without_headroom = 0
        retry_blocked_reasons: set[str] = set()
        worker_batch_count = 0
        max_worker_batch_size = 0
        max_known_state_writers_per_batch = 0
        invoke_batch = getattr(self.invocation_pool, "invoke_batch", None)
        if invoke_batch is None:
            futures = [
                asyncio.create_task(
                    self._invoke_pending_request(
                        request,
                        first_ordinal + index,
                    )
                )
                for index, request in enumerate(selected)
            ]
            completion_order: dict[asyncio.Task[Any], int] = {}
            completion_counter = 0
            for future in futures:
                def mark(done: asyncio.Task[Any]) -> None:
                    nonlocal completion_counter
                    completion_counter += 1
                    completion_order[done] = completion_counter
                future.add_done_callback(mark)
            remaining = set(futures)
            completed_rows: list[
                tuple[
                    _PendingInvocation,
                    tuple[
                        InvocationResult, str | None, int, str | None
                    ] | None,
                ]
            ] = []
            while remaining:
                done, remaining = await asyncio.wait(
                    remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.parallel.heartbeat_interval_seconds,
                )
                if not done:
                    self._telemetry(
                        "heartbeat",
                        inflight=min(self.parallel.workers, len(remaining)),
                    )
                    continue
                for future in sorted(
                    done, key=lambda row: completion_order.get(row, 0)
                ):
                    completed_rows.append(future.result())
        else:
            # Spread calls that have actually changed state in an earlier
            # frozen round. They invalidate the loaded state and can persist a
            # handle, so clustering them creates a long-tail worker batch.
            # Everything is still grouped by immutable handle inside a batch,
            # and stable reduction below follows original dispatch sequence.
            ordinal_by_sequence = {
                request.sequence: first_ordinal + index
                for index, request in enumerate(sorted(
                    selected, key=lambda row: row.sequence
                ))
            }
            grouped_batches = _partition_worker_batches(
                selected,
                workers=self.parallel.workers,
                max_batch_size=self.parallel.ipc_batch_size,
                observed_state_writer_tools=(
                    self._observed_state_writer_tools
                ),
            )
            worker_batch_count = len(grouped_batches)
            max_worker_batch_size = max(map(len, grouped_batches), default=0)
            max_known_state_writers_per_batch = max(
                (
                    sum(
                        request.tool in self._observed_state_writer_tools
                        for request in batch
                    )
                    for batch in grouped_batches
                ),
                default=0,
            )
            async def collect_batches(
                batches: Sequence[
                    tuple[
                        Sequence[_PendingInvocation],
                        Sequence[int],
                        int | None,
                    ]
                ],
            ) -> list[
                tuple[
                    _PendingInvocation,
                    tuple[
                        InvocationResult, str | None, int, str | None
                    ] | None,
                    int | None,
                ]
            ]:
                batch_futures = [
                    asyncio.create_task(self._invoke_pending_batch(
                        batch,
                        ordinals,
                        avoid_worker=avoid_worker,
                    ))
                    for batch, ordinals, avoid_worker in batches
                ]
                completion_order: dict[asyncio.Task[Any], int] = {}
                completion_counter = 0
                for future in batch_futures:
                    def mark(done: asyncio.Task[Any]) -> None:
                        nonlocal completion_counter
                        completion_counter += 1
                        completion_order[done] = completion_counter
                    future.add_done_callback(mark)
                remaining = set(batch_futures)
                rows: list[
                    tuple[
                        _PendingInvocation,
                        tuple[
                            InvocationResult, str | None, int, str | None
                        ] | None,
                        int | None,
                    ]
                ] = []
                while remaining:
                    done, remaining = await asyncio.wait(
                        remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=self.parallel.heartbeat_interval_seconds,
                    )
                    if not done:
                        self._telemetry(
                            "heartbeat",
                            inflight=min(self.parallel.workers, len(remaining)),
                        )
                        continue
                    for future in sorted(
                        done, key=lambda row: completion_order.get(row, 0)
                    ):
                        rows.extend(future.result())
                return rows

            initial_batches = [
                (
                    batch,
                    [
                        ordinal_by_sequence[request.sequence]
                        for request in batch
                    ],
                    None,
                )
                for batch in grouped_batches
            ]
            attempt_rows = await collect_batches(initial_batches)
            completed_rows = [
                (request, payload)
                for request, payload, _ in attempt_rows
                if payload is not None
            ]
            failed_rows = [
                (request, worker)
                for request, payload, worker in attempt_rows
                if payload is None
            ]
            global_retry_headroom = max(
                0,
                self.config.budget.max_calls
                - self.calls_used
            )
            # The 2,880 frozen primary requests remain one bulk round. Retry
            # waves run before stable reduction with the same frozen contract;
            # their actual calls are charged globally and close this epoch
            # immediately afterward.
            retry_limit = self.config.budget.max_retries
            retry_headroom_after_initial = min(
                global_retry_headroom,
                (
                    max(0, retry_limit - self._worker_retries)
                    if retry_limit is not None
                    else global_retry_headroom
                ),
            )

            # Retries are frozen waves. Failed rows are selected by original
            # dispatch sequence, so retry ordinals and call IDs never depend on
            # which worker batch happened to finish first.
            for retry_wave in range(1, self.parallel.worker_retries + 1):
                if not failed_rows:
                    break
                retry_limit = self.config.budget.max_retries
                retry_remaining = (
                    len(failed_rows)
                    if retry_limit is None
                    else max(0, retry_limit - self._worker_retries)
                )
                global_remaining = max(
                    0,
                    self.config.budget.max_calls
                    - self.calls_used
                )
                retry_count = min(
                    len(failed_rows),
                    retry_remaining,
                    global_remaining,
                )
                ordered_failed = sorted(
                    failed_rows, key=lambda row: row[0].sequence
                )
                selected_failed = ordered_failed[:retry_count]
                selected_sequences = {
                    request.sequence for request, _ in selected_failed
                }
                unselected_failed = [
                    (request, worker)
                    for request, worker in failed_rows
                    if request.sequence not in selected_sequences
                ]
                completed_rows.extend(
                    (request, None) for request, _ in unselected_failed
                )
                retry_terminal_without_headroom += len(unselected_failed)
                if retry_count < len(failed_rows):
                    if global_remaining < len(failed_rows):
                        retry_blocked_reasons.add("global_calls")
                    if retry_remaining < len(failed_rows):
                        retry_blocked_reasons.add("aggregate_retries")
                if retry_count == 0:
                    if retry_remaining == 0:
                        self._record_budget_violation("max_retries")
                    failed_rows = []
                    break

                first_retry_ordinal = self.calls_used + 1
                retry_ordinals = {
                    request.sequence: first_retry_ordinal + index
                    for index, (request, _) in enumerate(selected_failed)
                }
                self.calls_used += retry_count
                self._worker_dispatches += retry_count
                self._worker_retries += retry_count

                # One retry batch has one worker to avoid. Within that group,
                # retain state locality without changing the stable ordinals.
                retry_groups: dict[
                    int | None, list[_PendingInvocation]
                ] = defaultdict(list)
                for request, failed_worker in selected_failed:
                    retry_groups[failed_worker].append(request)
                retry_batches: list[
                    tuple[
                        Sequence[_PendingInvocation],
                        Sequence[int],
                        int | None,
                    ]
                ] = []
                for failed_worker in sorted(
                    retry_groups,
                    key=lambda value: -1 if value is None else value,
                ):
                    requests_for_worker = sorted(
                        retry_groups[failed_worker],
                        key=lambda request: (
                            request.state.handle, request.sequence
                        ),
                    )
                    for offset in range(
                        0, len(requests_for_worker), self.parallel.ipc_batch_size
                    ):
                        batch = requests_for_worker[
                            offset:offset + self.parallel.ipc_batch_size
                        ]
                        retry_batches.append((
                            batch,
                            [retry_ordinals[row.sequence] for row in batch],
                            failed_worker,
                        ))
                attempt_rows = await collect_batches(retry_batches)
                completed_rows.extend(
                    (request, payload)
                    for request, payload, _ in attempt_rows
                    if payload is not None
                )
                failed_rows = [
                    (request, worker)
                    for request, payload, worker in attempt_rows
                    if payload is None
                ]
            if failed_rows:
                retry_blocked_reasons.add("per_work_item_retries")
            completed_rows.extend(
                (request, None) for request, _ in failed_rows
            )
        retry_attempts = self._worker_retries - worker_retries_before_round
        round_actual_calls = count + retry_attempts
        invocation_seconds = system_monotonic() - invocation_started
        for _, payload in completed_rows:
            self._record_work_item_completion(terminal_failure=payload is None)

        for request, payload in completed_rows:
            if payload is not None and payload[0].transition.changed:
                self._observed_state_writer_tools.add(request.tool)
        by_sequence = {request.sequence: payload for request, payload in completed_rows}
        self._round_phase = "reduce"
        reduce_started = system_monotonic()
        with self.store.batch():
            ordered_requests = sorted(selected, key=lambda row: row.sequence)
            # Queue every task wakeup in dispatch order. asyncio's ready queue
            # executes these callbacks FIFO, so graph effects retain the exact
            # stable reduction order without 2,880 coordinator sleep/wakeup
            # round trips. A task either finishes or suspends at its next
            # invocation request before the following barrier completes.
            for request in ordered_requests:
                if request.future.done():
                    raise RuntimeError(
                        "dispatched invocation lost its waiting task: "
                        f"sequence={request.sequence} tool={request.tool} "
                        f"reason={request.reason}"
                    )
                request.future.set_result(by_sequence[request.sequence])
            while any(
                not request.owner.done()
                and self._pending_by_owner.get(request.owner) in (None, request)
                for request in ordered_requests
            ):
                await asyncio.sleep(0)
            for request, primary in sorted(
                coalesced, key=lambda row: row[0].sequence
            ):
                if request.future.done():
                    raise RuntimeError(
                        "coalesced invocation lost its waiting task: "
                        f"sequence={request.sequence} tool={request.tool} "
                        f"reason={request.reason}"
                    )
                request.future.set_result(by_sequence[primary.sequence])
                while (
                    not request.owner.done()
                    and self._pending_by_owner.get(request.owner)
                    in (None, request)
                ):
                    await asyncio.sleep(0)
        warmup_orchestration_seconds = 0.0
        if self._rounds_completed == 0:
            warmup_started = system_monotonic()
            self._prewarm_observation_choice_domains()
            warmup_orchestration_seconds = system_monotonic() - warmup_started
        reduce_seconds = system_monotonic() - reduce_started
        self._rounds_completed += 1
        self._active_inflight_count = 0
        self._outstanding_reservations = {
            request.task_id: 1
            for request in self._pending_invocations
            if request.task_id is not None
        }
        self._last_round_timings = {
            "fill_seconds": float(self._last_fill_timings.get("seconds", 0.0)),
            "fill_selection_seconds": float(
                self._last_fill_timings.get("selection_seconds", 0.0)
            ),
            "fill_frontier_seconds": float(
                self._last_fill_timings.get("frontier_seconds", 0.0)
            ),
            "fill_coroutine_seconds": float(
                self._last_fill_timings.get("coroutine_seconds", 0.0)
            ),
            "fill_selected": float(self._last_fill_timings.get("selected", 0)),
            "invocation_seconds": round(invocation_seconds, 6),
            "reduce_seconds": round(reduce_seconds, 6),
            "warmup_orchestration_seconds": round(
                warmup_orchestration_seconds, 6
            ),
            "total_seconds": round(system_monotonic() - round_started, 6),
            "calls": float(round_actual_calls),
            "retry_attempts": retry_attempts,
            "retry_headroom_after_initial": retry_headroom_after_initial,
            "retry_terminal_without_headroom": (
                retry_terminal_without_headroom
            ),
            "retry_blocked_reasons": sorted(retry_blocked_reasons),
            "worker_batches": worker_batch_count,
            "max_worker_batch_size": max_worker_batch_size,
            "max_known_state_writers_per_batch": (
                max_known_state_writers_per_batch
            ),
            "observed_state_writer_tools": len(
                self._observed_state_writer_tools
            ),
        }
        self._cleanup_bulk_tasks(active)
        # Every owning call is now committed in stable dispatch order. Restore
        # exact-binding duplicates with their original priority records; on the
        # next fill they take the normal committed-call reuse path and can add
        # their own provenance/trials without another template invocation.
        self._release_round_binding_reservations()
        self._round_phase = "between_rounds"
        # A full round is also the contract-epoch boundary.  Its epoch row,
        # emitted immediately by run_async, already contains these timings;
        # avoid traversing the entire live queue twice for duplicate telemetry.
        if self.calls_used - self._epoch_start_calls < self.parallel.epoch_calls:
            self._telemetry("round", inflight=0)
        return round_actual_calls

    async def _cancel_bulk_tasks(
        self, active: dict[asyncio.Task[None], str]
    ) -> None:
        for future in active:
            if not future.done():
                future.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        for future, task_id in list(active.items()):
            self._finish_future_dispatch_slot(future)
            self._future_task_ids.pop(future, None)
            self._invocation_ordinals.pop(future, None)
            self._preclassified_outcomes.pop(future, None)
            self._pending_by_owner.pop(future, None)
            # A cancelled coroutine still owns the same mutable SearchTask and
            # possibly one incomplete saved state.  Put its original queue
            # entry back verbatim so neither work nor the handle is orphaned.
            self._release_task(
                task_id,
                restore_leased=(
                    future.cancelled()
                    or task_id in self._incomplete_task_bindings
                ),
            )
        active.clear()
        for request in self._pending_invocations:
            if not request.future.done():
                request.future.cancel()
        self._pending_invocations.clear()
        self._outstanding_reservations.clear()
        self._active_inflight_count = 0
        # Cancellation can happen after fill held duplicate tasks but before a
        # round reached stable reduction. No reservation survives that boundary,
        # and every held task returns as the exact entry that was selected.
        self._release_round_binding_reservations()

    async def run_async(self) -> dict[str, Any]:
        active: dict[asyncio.Task[None], str] = {}
        self._bulk_done_futures: deque[asyncio.Task[None]] = deque()
        self._outstanding_reservations = {}
        self.store.set_meta("parallel_outstanding_reservations", {})
        self._bulk_run_active = True
        self._round_phase = "starting"
        self._persist_parallel_meta()
        self._telemetry("start")
        self._check_non_call_budgets()
        try:
            while self.stop_reason == "completed":
                self._cleanup_bulk_tasks(active)
                await self._fill_bulk_requests(active)
                self._cleanup_bulk_tasks(active)
                if self._pending_invocations:
                    dispatched = await self._dispatch_bulk_round(active)
                    if dispatched == 0:
                        await self._cancel_bulk_tasks(active)
                    else:
                        self._cleanup_bulk_tasks(active)
                epoch_calls = self.calls_used - self._epoch_start_calls
                if epoch_calls >= self.parallel.epoch_calls:
                    await self._cancel_bulk_tasks(active)
                    # Finish repairs, incremental embeddings, and reranking
                    # even when this epoch consumes the final call budget.
                    self._finish_contract_epoch("epoch")
                    if (
                        self.calls_used >= self.config.budget.max_calls
                    ):
                        self._record_budget_violation("max_calls")
                        break
                    continue
                if (
                    self.calls_used >= self.config.budget.max_calls
                ):
                    await self._cancel_bulk_tasks(active)
                    if epoch_calls > 0:
                        self._finish_contract_epoch("partial_epoch")
                    self._record_budget_violation("max_calls")
                    break
                if (
                    self.config.budget.max_seconds is not None
                    and self._elapsed() >= self.config.budget.max_seconds
                ):
                    self._record_budget_violation("max_seconds")
                    break
                self._check_non_call_budgets()
                if self.stop_reason != "completed":
                    break
                if not active and not self._pending_invocations:
                    # A fill can end without a worker call when owners finish
                    # through trial-binding reuse. Restore tasks deferred behind
                    # those owners before deciding discovery is exhausted.
                    self._release_round_binding_reservations()
                    has_pending = bool(self._priority_entries)
                    if (
                        not has_pending
                        and (self._pending_feedback or self._pending_state_repairs)
                    ):
                        self._finish_contract_epoch("partial_epoch")
                        continue
                    if not has_pending:
                        break
                    # A completed bulk round can leave continuations and newly
                    # generated frontier heads in the live queue.
                    # Start another fill cycle; queue presence here is progress,
                    # not a coordinator failure.
                    continue
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            raise
        finally:
            await self._cancel_bulk_tasks(active)
            self._bulk_run_active = False
            if self.stop_reason == "completed":
                self._check_non_call_budgets(force_storage=True)
            self._round_phase = "stopped"
            self._telemetry("stop", inflight=0)
            self._persist_parallel_meta()
        return self.graph_artifact()

    def graph_artifact(self) -> dict[str, Any]:
        graph = super().graph_artifact()
        graph["build"]["usage"]["llm_calls"] = self._effective_llm_calls()
        graph["build"]["usage"]["contract_repair_llm_calls"] = max(
            0,
            self.contracts.total_llm_calls() - self.initial_llm_calls,
        )
        graph["build"]["parallel"] = {
            "workers": self.parallel.workers,
            "calls_per_worker_epoch": self.parallel.calls_per_worker_epoch,
            "worker_ipc_batch_size": self.parallel.ipc_batch_size,
            "calls_per_worker_round": self.parallel.calls_per_worker_epoch,
            "round_capacity": self.parallel.round_capacity,
            "prefetch_rounds": self.parallel.prefetch_rounds,
            "prefetch_calls_per_worker": (
                self.parallel.prefetch_calls_per_worker
            ),
            "prefetch_round_capacity": (
                self.parallel.workers
                * (self.parallel.prefetch_calls_per_worker or 0)
            ),
            "epoch_calls": self.parallel.epoch_calls,
            "fairness_share": self.parallel.fairness_share,
            "prioritize_state_progress": (
                self.parallel.prioritize_state_progress
            ),
            "state_domain_policy": (
                "adapter_isolated" if self.state_domains else "shared"
            ),
            "fair_task_class_order": list(FAIR_TASK_CLASSES),
            "worker_retry_limit": self.parallel.worker_retries,
            "work_item_unit": "actual_invocation",
            "raw_result_order": "completion",
            "graph_reduce_order": "dispatch_sequence",
            "commit_order": "dispatch_sequence",
            "frontier_policy": "lazy_exhaustive",
            "rounds_completed": self._rounds_completed,
            "epochs_completed": self._epochs_completed,
            "worker_dispatches": self._worker_dispatches,
            "returned_invocations": self.store.count("calls"),
            "trial_binding_reuses": self._trial_binding_reuses,
            "completed_input_trials": self.store.count("input_trials"),
            "completed_state_trials": self.store.count("state_trials"),
            "incomplete_trial_tasks": len(self._incomplete_task_bindings),
            "retained_state_replays": len(self._retained_state_replays),
            "worker_retries": self._worker_retries,
            "worker_terminal_failures": self._worker_terminal_failures,
            "coordinator_error": self._last_coordinator_error,
            "completed_work_items": self._work_items_completed,
            "observation_tasks_pruned_after_success": (
                self.observation_tasks_pruned_after_success
            ),
            "outstanding_reservations": self._reservation_observation(),
            "worker_failure_rate": (
                self._worker_terminal_failures / max(1, self._work_items_completed)
            ),
            "last_round_timings": dict(self._last_round_timings),
            "last_epoch_tail_seconds": self._last_epoch_tail_seconds,
            "last_epoch_tail_timings": dict(self._last_epoch_tail_timings),
            "max_epoch_tail_seconds": self._max_epoch_tail_seconds,
            "epoch_tail_seconds_total": self._epoch_tail_seconds_total,
        }
        return graph
