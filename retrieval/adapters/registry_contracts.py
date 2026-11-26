"""Contract adapter for public registries with explicit calling conventions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from retrieval.core.contracts import (
    DEFAULT_FEEDBACK_POLICY,
    ContractAdapter,
    FeedbackPolicy,
    ToolCallContract,
    build_contract_prompt,
    contract_prompt_rules,
)


REGISTRY_PROMPT_VERSION = "tool_call_contract_prompt_registry_v4"
STATEFUL_BUSINESS_OUTCOME_REGISTRY_PROMPT_VERSION = (
    "tool_call_contract_prompt_stateful_registry_v1"
)


def _build_registry_contract_prompt(
    metadata: Mapping[str, Any],
    *,
    revision: int,
    previous: Mapping[str, Any] | None = None,
    feedback: Sequence[Mapping[str, Any]] = (),
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
    stateful: bool,
) -> list[dict[str, str]]:
    calling_convention = dict(metadata["calling_convention"])
    if calling_convention.get("kind") == "json_object":
        protocol_rule = (
            "This runtime receives one JSON object: template.arguments are "
            "exactly that object's public keys, with their original spelling."
        )
    else:
        protocol_rule = (
            "This runtime uses the declared callable signature: "
            "template.arguments are exactly its public parameter names."
        )

    replacements = {
        "Use omit only for an optional parameter that the template intentionally omits.": (
            "Use omit when a parameter is intentionally absent from this "
            "template; omit is a call choice, not a model-filled value."
        ),
        "Omit is orthogonal to dependency: when an optional parameter is supplied and must reference a particular pre-existing value, use observation.": (
            "Omit is orthogonal to dependency: a field may be absent in one "
            "template and still require an upstream observation whenever supplied."
        ),
        "Caller-supplied arguments remain observable after the call; their mechanical $.parameter_name exports are normalized from the public signature.": (
            "Do not declare caller-argument exports; they are added mechanically "
            "from the public interface."
        ),
        "Export every scalar leaf shown in the public success response; use [*] for array elements.": (
            "For a structured public success response, do not repeat its scalar "
            "exports; exact public field labels and paths are added mechanically. "
            "For a scalar success response, declare exactly one response export "
            "at path $ and name it with the shortest snake_case semantic value "
            "name explicitly supported by this tool's public function, "
            "description, or return documentation. Use response_value only when "
            "that metadata gives the scalar no semantic identity."
        ),
    }
    if stateful:
        replacements[
            "A failure outcome schema must match top-level status rejected or error and put response constraints under output."
        ] = (
            "A failure outcome schema normally matches top-level status rejected "
            "or error. It may instead match top-level status success only when "
            "public metadata explicitly defines an output value as a "
            "business-level failure."
        )
    rules = [
        replacements.get(rule, rule)
        for rule in contract_prompt_rules()
    ]
    literal_rule = (
        "For a state-changing action, put first a success template that "
        "completes the operation described by the public metadata rather "
        "than a partial, no-op, or opposite transition. When the metadata "
        "describes a bounded degree, use its completing boundary; when a "
        "collection selects independently named components and lists the "
        "complete allowed set, include that full set. Prefer this one "
        "effect-complete template over extra reset or diversity templates."
        if stateful
        else (
            "Use the shortest literal that can plausibly succeed; when metadata "
            "states a minimum length, meet it without padding beyond that minimum."
        )
    )
    rules[1:1] = [
        "Return only fields defined by required_output_schema; metadata and calling-convention fields are input context, not contract output.",
        protocol_rule,
        "Copy template argument keys exactly from public_tool_metadata.parameters[*].name; never invent an alias or use a parameter name found only in prose.",
        "If public_tool_metadata.parameters is empty, every template arguments object must be empty.",
        "Every literal must validate against that parameter's exact JSON Schema using native JSON types; numbers and booleans are not quoted strings.",
        "Never emit a long payload as a literal. For a sequence, document, file, blob, encoded value, or any input whose plausible example would be long, use observation instead; every literal must remain short enough to inspect on one line.",
        literal_rule,
    ]
    rules.extend([
        "Keep outcome schemas minimal: constrain normalized status and output shape without copying the full public response object.",
    ])
    if stateful:
        rules.extend([
            "A transport-success response can still report a business-level failure. Use an output const or enum as a success/failure discriminator only when this tool's public metadata explicitly defines those values.",
            "For a read, query, or status operation, a returned boolean is successful result data unless public metadata explicitly says one value means the operation failed; otherwise both true and false belong to the success outcome.",
            "For an action, authentication, or mutation whose public metadata explicitly defines success and failure flag values, constrain those values in disjoint success and failure schemas.",
        ])
    rules.append(
        "Before returning, check every template key against public_tool_metadata.parameters[*].name, every required parameter against its template, and every literal against its listed schema."
    )
    messages = build_contract_prompt(
        metadata,
        revision=revision,
        previous=previous,
        feedback=feedback,
        feedback_policy=feedback_policy,
        rules=rules,
        output_schema=ToolCallContract.model_json_schema(),
    )
    messages[0]["content"] += (
        " Return compact JSON. Never write or repeat a long payload literal; "
        "represent that input as observation."
    )
    return messages


def build_registry_contract_prompt(
    metadata: Mapping[str, Any],
    *,
    revision: int,
    previous: Mapping[str, Any] | None = None,
    feedback: Sequence[Mapping[str, Any]] = (),
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
) -> list[dict[str, str]]:
    return _build_registry_contract_prompt(
        metadata,
        revision=revision,
        previous=previous,
        feedback=feedback,
        feedback_policy=feedback_policy,
        stateful=False,
    )


def build_stateful_business_outcome_registry_contract_prompt(
    metadata: Mapping[str, Any],
    *,
    revision: int,
    previous: Mapping[str, Any] | None = None,
    feedback: Sequence[Mapping[str, Any]] = (),
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
) -> list[dict[str, str]]:
    return _build_registry_contract_prompt(
        metadata,
        revision=revision,
        previous=previous,
        feedback=feedback,
        feedback_policy=feedback_policy,
        stateful=True,
    )


REGISTRY_CONTRACT_ADAPTER = ContractAdapter(
    prompt_version=REGISTRY_PROMPT_VERSION,
    build_prompt=build_registry_contract_prompt,
    allow_required_omit=True,
    allow_nonidentifier_export_names=True,
    bracket_nonidentifier_argument_paths=True,
    response_exports_only_for_public_scalars=True,
    tolerant_jsonpath=True,
)

STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER = ContractAdapter(
    prompt_version=STATEFUL_BUSINESS_OUTCOME_REGISTRY_PROMPT_VERSION,
    build_prompt=build_stateful_business_outcome_registry_contract_prompt,
    allow_required_omit=True,
    allow_nonidentifier_export_names=True,
    bracket_nonidentifier_argument_paths=True,
    response_exports_only_for_public_scalars=True,
    tolerant_jsonpath=True,
    complete_boolean_business_outcomes=True,
)
