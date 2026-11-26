"""Per-tool executable call contracts generated from public metadata.

Each contract describes one tool in isolation. Call templates distinguish values
that the contract provider can construct from values that must be observed at
runtime. Exports are explicit JSONPath selectors; graph construction never
recursively promotes arbitrary JSON children into candidates.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, AbstractSet, Callable, Literal, Mapping, Protocol, Sequence

from jsonpath_ng.ext.parser import ExtendedJsonPathParser
from jsonschema import Draft202012Validator, SchemaError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_serializer,
    model_validator,
)


CONTRACT_VERSION = "tool_call_contract_v2"
MANIFEST_VERSION = "tool_call_contract_manifest_v2"
PROMPT_VERSION = "tool_call_contract_prompt_v26"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
MAX_LLM_CALLS_PER_TOOL = 3
MINIMAL_SUCCESS_TEMPLATE_POLICY = "required_only_public_baseline_v1"
MODEL_DECLARED_TEMPLATE_POLICY = "model_declared_v1"
TemplatePolicy = Literal[
    "required_only_public_baseline_v1",
    "model_declared_v1",
]
TEMPLATE_POLICIES: frozenset[str] = frozenset({
    MINIMAL_SUCCESS_TEMPLATE_POLICY,
    MODEL_DECLARED_TEMPLATE_POLICY,
})
FeedbackPolicy = Literal["complete-same-tool", "structural-redacted"]
DEFAULT_FEEDBACK_POLICY: FeedbackPolicy = "complete-same-tool"
FEEDBACK_POLICIES: frozenset[str] = frozenset({
    "complete-same-tool",
    "structural-redacted",
})
_EXPORT_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ContractAdapter:
    """Calling-protocol choices around the shared contract semantics."""

    prompt_version: str
    build_prompt: Callable[..., list[dict[str, str]]]
    allow_required_omit: bool = False
    allow_nonidentifier_export_names: bool = False
    bracket_nonidentifier_argument_paths: bool = False
    response_exports_only_for_public_scalars: bool = False
    tolerant_jsonpath: bool = False
    complete_boolean_business_outcomes: bool = False


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scalar_key(*values: Any) -> str:
    """Type-sensitive in-memory identity for feedback values."""

    return canonical_json(values)


class ArgumentSpec(BaseModel):
    """How one public parameter is populated in a call template."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["literal", "observation", "omit"]
    value: Any = None

    @model_validator(mode="after")
    def validate_value(self) -> "ArgumentSpec":
        supplied = "value" in self.model_fields_set
        if self.kind == "literal" and not supplied:
            raise ValueError("literal argument requires value")
        if self.kind != "literal" and supplied:
            raise ValueError(f"{self.kind} argument cannot include value")
        return self

    @model_serializer(mode="plain")
    def serialize_argument(self) -> dict[str, Any]:
        row: dict[str, Any] = {"kind": self.kind}
        if self.kind == "literal":
            row["value"] = self.value
        return row


class CallTemplate(BaseModel):
    """One proposed success or failure invocation."""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, ArgumentSpec]
    expected: Literal["success", "failure"]

    def literal_arguments(self) -> dict[str, Any]:
        return {
            name: specification.value
            for name, specification in self.arguments.items()
            if specification.kind == "literal"
        }

    def has_observations(self) -> bool:
        return any(row.kind == "observation" for row in self.arguments.values())


class ToolExport(BaseModel):
    """A reusable value selected from one observable invocation record."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    name: str = Field(min_length=1)
    source: Literal["arguments", "response", "state_changes"]
    path: str = Field(min_length=1)
    value_schema: dict[str, Any] = Field(alias="schema", serialization_alias="schema")
    propagate: bool = Field(
        default=True,
        description=(
            "Whether this exact response/state value carries specific runtime "
            "information worth offering to later tool calls."
        ),
    )

    @model_validator(mode="after")
    def validate_export(self) -> "ToolExport":
        try:
            _compiled_jsonpath(self.path)
        except Exception as exc:
            raise ValueError(f"invalid JSONPath: {exc}") from exc
        try:
            Draft202012Validator.check_schema(self.value_schema)
        except SchemaError as exc:
            raise ValueError(f"invalid export JSON Schema: {exc.message}") from exc
        return self


class ToolCallContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[CONTRACT_VERSION] = CONTRACT_VERSION
    tool: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1, le=MAX_LLM_CALLS_PER_TOOL)
    templates: tuple[CallTemplate, ...] = Field(min_length=1)
    exports: tuple[ToolExport, ...] = ()
    success_schema: dict[str, Any]
    failure_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_contract(self) -> "ToolCallContract":
        if not any(template.expected == "success" for template in self.templates):
            raise ValueError("templates must include a success call")
        names = [row.name for row in self.exports]
        if len(names) != len(set(names)):
            raise ValueError("export names must be unique")
        try:
            Draft202012Validator.check_schema(self.success_schema)
            Draft202012Validator.check_schema(self.failure_schema)
        except SchemaError as exc:
            raise ValueError(f"invalid outcome JSON Schema: {exc.message}") from exc
        return self


def with_minimal_success_baseline(
    contract: ToolCallContract,
    *,
    parameter_names: Sequence[str],
    required_parameters: AbstractSet[str],
) -> ToolCallContract:
    """Prepend one canonical success call that omits every optional input.

    Model-proposed templates remain intact.  The extra baseline uses the first
    success template only as the source of required argument specifications;
    public optional parameters are explicitly omitted so their documented
    defaults can produce reusable outputs without adding another model call.
    """

    source = next(
        (template for template in contract.templates if template.expected == "success"),
        None,
    )
    if source is None:
        return contract
    required = set(required_parameters)
    arguments: dict[str, ArgumentSpec] = {}
    for name in parameter_names:
        if name in required:
            specification = source.arguments.get(name)
            if specification is None:
                return contract
            arguments[name] = specification
        else:
            arguments[name] = ArgumentSpec(kind="omit")
    minimal = CallTemplate(arguments=arguments, expected="success")
    if contract.templates[0] == minimal:
        return contract
    payload = contract.model_dump(mode="json", by_alias=True)
    payload["templates"] = [
        minimal.model_dump(mode="json"),
        *[template.model_dump(mode="json") for template in contract.templates],
    ]
    return ToolCallContract.model_validate(payload)


class ContractManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[MANIFEST_VERSION] = MANIFEST_VERSION
    prompt_version: str = PROMPT_VERSION
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY
    template_policy: TemplatePolicy = MINIMAL_SUCCESS_TEMPLATE_POLICY
    model: str
    metadata_id: str
    contracts: dict[str, ToolCallContract] = Field(default_factory=dict)
    unknown: dict[str, str] = Field(default_factory=dict)
    audit: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    generated_at_unix: int = Field(default_factory=lambda: int(time.time()))

    @classmethod
    def load(cls, path: str | Path) -> "ContractManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


def load_contracts(path: str | Path) -> dict[str, ToolCallContract]:
    """Load either working contract state or the committed semantic contract set."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("contracts must be a JSON object")
    rows = (
        payload["contracts"]
        if "contracts" in payload and "metadata_id" in payload
        else payload
    )
    if not isinstance(rows, Mapping):
        raise ValueError("contracts must map tool names to contract objects")
    contracts: dict[str, ToolCallContract] = {}
    for tool, row in rows.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"contract for {tool} must be a JSON object")
        declared_tool = row.get("tool", tool)
        if declared_tool != tool:
            raise ValueError(f"contract key and tool disagree for {tool}")
        contracts[str(tool)] = ToolCallContract.model_validate({
            **dict(row),
            "tool": str(tool),
        })
    return contracts


def write_contracts(
    contracts: Mapping[str, ToolCallContract], path: str | Path
) -> None:
    """Write only contract fields that can affect a clean graph build."""

    payload = {
        tool: contract.model_dump(
            mode="json",
            by_alias=True,
            exclude={"version", "tool"},
        )
        for tool, contract in sorted(contracts.items())
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_template_policy(
    manifest: ContractManifest,
    requested: TemplatePolicy,
) -> None:
    """Select a template policy without relabeling normalized templates.

    The required-only policy prepends a deterministic success baseline.  That
    transformation is intentionally one-way: changing only the manifest label
    cannot recover the original model-declared template list.
    """

    if requested not in TEMPLATE_POLICIES:
        raise ValueError(f"unknown template policy: {requested}")
    if (
        manifest.template_policy == MINIMAL_SUCCESS_TEMPLATE_POLICY
        and requested == MODEL_DECLARED_TEMPLATE_POLICY
    ):
        raise ValueError(
            "cannot downgrade required-only templates to model-declared; "
            "load the original baseline-free contract seed"
        )
    manifest.template_policy = requested


class ContractLLM(Protocol):
    model: str

    def complete(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[str, Mapping[str, Any]]: ...


class DeepSeekFlash:
    """Small OpenAI-compatible client with reasoning disabled."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        transport_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or ""
        if not self.api_key:
            raise ValueError("Missing DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.transport_retries = transport_retries

    def complete(
        self, messages: Sequence[Mapping[str, str]]
    ) -> tuple[str, Mapping[str, Any]]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": list(messages),
                "temperature": 0,
                "seed": 42,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.transport_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return (
                    str(payload["choices"][0]["message"]["content"]),
                    dict(payload.get("usage") or {}),
                )
            except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
                last_error = exc
                if attempt + 1 < self.transport_retries:
                    time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"DeepSeek request failed: {last_error}") from last_error


def _extract_object(raw: str) -> Mapping[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, Mapping):
        raise ValueError("contract response must be a JSON object")
    return value


def normalized_invocation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"enum": ["success", "rejected", "error"]},
            "output": {},
            "machine": {"type": "object"},
            "state_changed": {"type": "boolean"},
            "changes": {},
        },
        "required": ["status", "output", "machine", "state_changed", "changes"],
        "additionalProperties": False,
    }


def normalize_outcome_schema(
    schema: Mapping[str, Any], *, expected: Literal["success", "failure"]
) -> dict[str, Any]:
    """Lift a raw response schema into the normalized invocation record.

    Models sometimes return the public response schema directly even when the
    requested contract is over invocation records. The lift is structural and
    independent of any tool or domain.
    """
    row = normalize_json_schema(schema)
    properties = row.get("properties")
    required = row.get("required")
    if (
        isinstance(properties, Mapping)
        and "status" in properties
        and "output" in properties
        and isinstance(required, list)
        and "status" in required
        and "output" in required
    ):
        status = properties.get("status")
        if isinstance(status, Mapping):
            declared: set[str] = set()
            if isinstance(status.get("const"), str):
                declared = {str(status["const"])}
            elif (
                isinstance(status.get("enum"), list)
                and status["enum"]
                and all(isinstance(value, str) for value in status["enum"])
            ):
                declared = set(map(str, status["enum"]))
            expected_statuses = (
                {"success"}
                if expected == "success"
                else {"rejected", "error"}
            )
            output = properties.get("output")
            output_properties = (
                output.get("properties")
                if isinstance(output, Mapping)
                else None
            )
            business_failure = (
                expected == "failure"
                and declared == {"success"}
                and isinstance(output, Mapping)
                and (
                    "const" in output
                    or isinstance(output.get("enum"), list)
                    or (
                        isinstance(output_properties, Mapping)
                        and any(
                            isinstance(child, Mapping)
                            and (
                                "const" in child
                                or isinstance(child.get("enum"), list)
                            )
                            for child in output_properties.values()
                        )
                    )
                )
            )
            try:
                Draft202012Validator.check_schema(status)
                validator = Draft202012Validator(status)
                declared_is_satisfiable = all(
                    validator.is_valid(value) for value in declared
                )
            except SchemaError:
                declared_is_satisfiable = False
            # Be conservative: only an explicit const/enum can prove that the
            # envelope accepts no status outside the requested polarity.
            if (
                declared
                and (declared <= expected_statuses or business_failure)
                and declared_is_satisfiable
            ):
                return row
    status_schema: dict[str, Any] = (
        {"const": "success"}
        if expected == "success"
        else {"enum": ["rejected", "error"]}
    )
    return {
        "type": "object",
        "properties": {
            "status": status_schema,
            "output": row,
        },
        "required": ["status", "output"],
    }


def complete_boolean_business_failure(
    success_schema: Mapping[str, Any],
    failure_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the opposite boolean as failure when success names one value.

    The business-outcome adapter asks the model to use a boolean const only
    when public metadata says that the two values distinguish success from
    failure.  A transport rejection is already classified as failure by the
    runtime, so the explicit business-level complement is the useful schema.
    """

    output = success_schema.get("properties", {}).get("output")
    if not isinstance(output, Mapping):
        return dict(failure_schema)

    found: list[tuple[tuple[str, ...], bool]] = []

    def visit(schema: Mapping[str, Any], path: tuple[str, ...]) -> None:
        value = schema.get("const")
        if isinstance(value, bool):
            found.append((path, value))
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            for name, child in properties.items():
                if isinstance(child, Mapping):
                    visit(child, (*path, str(name)))

    visit(output, ())
    if len(found) != 1:
        return dict(failure_schema)

    path, success_value = found[0]
    opposite: dict[str, Any] = {"const": not success_value}
    for name in reversed(path):
        opposite = {
            "type": "object",
            "properties": {name: opposite},
            "required": [name],
        }
    return {
        "type": "object",
        "properties": {
            "status": {"const": "success"},
            "output": opposite,
        },
        "required": ["status", "output"],
    }


def normalize_json_schema(schema: Any) -> Any:
    """Normalize common pre-2020 tuple syntax without changing its meaning."""
    if isinstance(schema, list):
        return [normalize_json_schema(row) for row in schema]
    if not isinstance(schema, Mapping):
        return schema
    row = {
        str(key): normalize_json_schema(value)
        for key, value in schema.items()
    }
    if isinstance(row.get("items"), list):
        row["prefixItems"] = row.pop("items")
        row["items"] = {}
    return row


_SYMBOLIC_FEEDBACK_KEYS = {
    "classified",
    "conclusion",
    "event",
    "expected",
    "export",
    "exports",
    "location",
    "outcome",
    "parameter",
    "parameters",
    "path",
    "paths",
    "source",
    "status",
    "transport_status",
    "type",
    "validation_errors",
    "wildcard",
}


def redact_runtime_feedback(
    value: Any,
    *,
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
    depth: int = 0,
    key: str | None = None,
) -> Any:
    """Render same-tool feedback according to the configured feedback policy.

    Complete feedback exposes only ``invocation_result`` verbatim.  Other
    feedback remains structural so coordinator annotations cannot become an
    accidental cross-tool value channel.  Structural feedback retains keys,
    symbolic paths/outcomes, types, and bounded collection shape while masking
    concrete runtime scalars.
    """
    if feedback_policy not in FEEDBACK_POLICIES:
        raise ValueError(f"unknown feedback policy: {feedback_policy}")
    if depth >= 8:
        return "<truncated>"
    if key == "invocation_result" and feedback_policy == "complete-same-tool":
        return value
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return "<boolean>"
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_runtime_feedback(
                child,
                feedback_policy=feedback_policy,
                depth=depth + 1,
                key=str(child_key),
            )
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        rendered = [
            redact_runtime_feedback(
                child,
                feedback_policy=feedback_policy,
                depth=depth + 1,
                key=key,
            )
            for child in value[:32]
        ]
        if len(value) > 32:
            rendered.append(f"<truncated:{len(value) - 32}>")
        return rendered
    if isinstance(value, str):
        if key in _SYMBOLIC_FEEDBACK_KEYS:
            return value
        return f"<string:length={len(value)}>"
    if isinstance(value, int):
        return "<integer>"
    if isinstance(value, float):
        return "<number>"
    return f"<{type(value).__name__}>"


def _render_feedback(
    feedback: Sequence[Mapping[str, Any]], feedback_policy: FeedbackPolicy
) -> list[Any]:
    return [
        redact_runtime_feedback(row, feedback_policy=feedback_policy)
        for row in feedback
    ]


def _scalar_keys(value: Any) -> set[str]:
    """Return type-sensitive identities for scalar leaves."""
    if value is None or isinstance(value, bool | int | float | str):
        return {scalar_key(type(value).__name__, value)}
    if isinstance(value, Mapping):
        return {
            key
            for child in value.values()
            for key in _scalar_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {
            key
            for child in value
            for key in _scalar_keys(child)
        }
    return set()


def feedback_derived_literal_errors(
    previous: ToolCallContract,
    candidate: ToolCallContract,
    metadata: Mapping[str, Any],
    feedback: Sequence[Mapping[str, Any]],
    *,
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
) -> list[str]:
    """Reject newly introduced literals copied only from visible feedback.

    The check is intentionally narrow and domain-free.  A scalar remains legal
    when it was already a literal in the active contract or occurs in public
    metadata.  Otherwise, a value visible only through the selected feedback
    policy cannot become a persisted template literal.
    """
    visible = _render_feedback(feedback, feedback_policy)
    feedback_scalars = _scalar_keys(visible)
    if not feedback_scalars:
        return []
    permitted = _scalar_keys(metadata)
    for template in previous.templates:
        for specification in template.arguments.values():
            if specification.kind == "literal":
                permitted.update(_scalar_keys(specification.value))
    forbidden = feedback_scalars - permitted
    if not forbidden:
        return []
    errors: list[str] = []
    for template_index, template in enumerate(candidate.templates):
        for argument, specification in sorted(template.arguments.items()):
            if specification.kind != "literal":
                continue
            if _scalar_keys(specification.value) & forbidden:
                errors.append(
                    f"feedback_literal:template:{template_index}:argument:{argument}"
                )
    return sorted(set(errors))


def contract_prompt_rules() -> list[str]:
    """Shared dependency semantics used by calling-protocol adapters."""

    return [
        "Describe exactly one supplied tool and return only the complete JSON contract.",
        "Represent every required public parameter in every call template.",
        "Use literal when a caller can construct the value from public information alone.",
        "Use observation when successful use needs an exact value learned at runtime.",
        "Dependency-required is independent of whether the public signature marks a parameter required: use observation exactly when this call must refer to a particular pre-existing value that a caller cannot choose or derive from public information.",
        "A schema-valid placeholder or guessed identifier, token, handle, or reference is not a constructible literal when an arbitrary same-type value would fail; never use a dummy value merely because it has the right JSON type.",
        "Every literal in an expected=success template must be a plausible value that can work as written using public information alone; otherwise use observation.",
        "A success template may and should contain multiple observation arguments when successful use needs multiple exact runtime values.",
        "Test constructibility against success: if replacing an input with another arbitrary schema-valid value would require different pre-existing setup or make the call fail, that input is not independently constructible and must use observation.",
        "Inputs that this call freely chooses for a new value—such as initial attributes, a newly chosen password, user-authored content, amounts, quantities, dates, flags, and search or filter criteria—are constructible literals unless the current tool metadata explicitly requires a pre-existing reference.",
        "For a call that authenticates, reads, changes, deletes, sends to, or relates a pre-existing record, use observation for every identity-bearing account value, credential, recipient/contact, identifier, handle, code, or path that must match that record. A caller typing the value does not make it independently constructible.",
        "In an expected-success authentication or sign-in template, both the existing account identifier and its existing secret are observations; never invent either. Account creation instead constructs its new identifier and secret. Credential recovery observes the existing account and issued recovery code but constructs the replacement secret.",
        "Do not use observation merely because another tool could output a schema-compatible value; the value must be tied to a particular pre-existing entity, credential, file, or state needed by this call.",
        "Use omit only for an optional parameter that the template intentionally omits.",
        "Omit is orthogonal to dependency: when an optional parameter is supplied and must reference a particular pre-existing value, use observation.",
        "Across the proposed success templates, give every optional public parameter that a caller can construct at least one literal example; another success template may still omit it.",
        "Literals are frozen proposals; ordinary propagation will not replace them.",
        "Include at least one proposed success template.",
        "Declare every reusable value with an explicit JSONPath export.",
        "Caller-supplied arguments remain observable after the call; their mechanical $.parameter_name exports are normalized from the public signature.",
        "Export every scalar leaf shown in the public success response; use [*] for array elements.",
        "For every response or state_changes export, set propagate=true when its exact runtime value identifies, locates, describes, authenticates, or carries specific entity/content/state information that a later tool call could meaningfully reuse.",
        "Every identifier or foreign key, opaque handle/token/code, address/contact, path/URL, timestamp/date, user-authored content, entity attribute, and monetary value must propagate, even when echoed or easy to replace with some other schema-valid value.",
        "Set propagate=false only for generic acknowledgement/status/error text, boolean flags, aggregate counts/ratings/ranks, pagination/control metadata, or display-only summaries that neither identify nor describe a particular entity or content item.",
        "Judge propagation from this tool alone and never from the runtime scalar's magnitude; a small integer or short string can still be an opaque resource reference.",
        "Exports may select arguments, response, or state_changes and require a JSON Schema.",
        "When same-tool feedback supports semantic names and exact paths for scalar state fields, export every such field; never return only a representative subset.",
        "Same-tool JIT feedback is rendered exactly according to feedback_policy; use only the supplied representation to map meaningful field names to exact paths.",
        "Every state export must name exactly one semantic field and use one explicit JSONPath.",
        "A state_changes export must select reusable scalar leaves, not only a whole array or object container.",
        "Never use recursive catch-all exports such as $..[*]; omit a hidden value when its field identity cannot be supported by the supplied context.",
        "Never copy a concrete runtime scalar from invocation_result into a template, export name, path, or schema.",
        "Do not export arbitrary descendants whose semantic identity is supported by neither public metadata nor supplied same-tool feedback.",
        "Use standard JSON Schema Draft 2020-12 for observable success and failure outcomes.",
        "Outcome schemas are over the normalized invocation record, never the raw response alone.",
        "A success outcome schema must match top-level status success and put response constraints under output.",
        "A failure outcome schema must match top-level status rejected or error and put response constraints under output.",
        "The success and failure schemas must describe disjoint observable outcomes.",
        "Use only this tool's public metadata and supplied same-tool feedback.",
        "Do not infer another tool, graph, task, benchmark, fixture, or application rule.",
    ]


def build_contract_prompt(
    metadata: Mapping[str, Any],
    *,
    revision: int,
    previous: Mapping[str, Any] | None = None,
    feedback: Sequence[Mapping[str, Any]] = (),
    feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
    rules: Sequence[str] | None = None,
    output_schema: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    if feedback_policy not in FEEDBACK_POLICIES:
        raise ValueError(f"unknown feedback policy: {feedback_policy}")
    if output_schema is None:
        selected_output_schema = ToolCallContract.model_json_schema()
        selected_output_schema["$defs"]["ToolExport"]["properties"]["name"][
            "pattern"
        ] = _EXPORT_IDENTIFIER.pattern
    else:
        selected_output_schema = dict(output_schema)
    selected_rules = list(rules) if rules is not None else contract_prompt_rules()
    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "revision": revision,
        "feedback_policy": feedback_policy,
        "public_tool_metadata": metadata,
        "invocation_record_schema": normalized_invocation_schema(),
        "required_output_schema": selected_output_schema,
        "rules": selected_rules,
    }
    if previous is not None:
        payload["previous_contract"] = previous
        payload["instruction"] = "Return one complete replacement contract."
    if feedback:
        payload["same_tool_invocation_feedback"] = _render_feedback(
            feedback, feedback_policy
        )
        if feedback_policy == "complete-same-tool":
            payload["feedback_note"] = (
                "Feedback is restricted to this tool. invocation_result is the "
                "complete return record for the call that triggered JIT repair; "
                "other feedback scalars remain structural. Use runtime values only "
                "to understand field structure and never copy a concrete feedback "
                "value into any persisted contract literal. Named state columns or "
                "keys are structural evidence: enumerate every scalar field whose "
                "semantic name and exact path the record supports."
            )
        else:
            payload["feedback_note"] = (
                "Feedback is restricted to this tool and contains structural keys, "
                "paths, types, and collection shape with concrete runtime scalars "
                "masked. Never turn a feedback placeholder into a persisted contract "
                "literal."
            )
    return [
        {
            "role": "system",
            "content": (
                "Write an executable call contract for one tool from its public metadata "
                "and supplied same-tool feedback only."
            ),
        },
        {"role": "user", "content": canonical_json(payload)},
    ]


DEFAULT_CONTRACT_ADAPTER = ContractAdapter(
    prompt_version=PROMPT_VERSION,
    build_prompt=build_contract_prompt,
)


def validate_contract_metadata(
    contract: ToolCallContract,
    metadata: Mapping[str, Any],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> list[str]:
    tool_id = str(metadata.get("tool") or metadata.get("name") or "")
    errors: list[str] = []
    if contract.tool != tool_id:
        errors.append("tool:mismatch")
    if not adapter.allow_nonidentifier_export_names:
        errors.extend(
            f"export:{export.name}:invalid_name"
            for export in contract.exports
            if _EXPORT_IDENTIFIER.fullmatch(export.name) is None
        )
    parameters = {
        str(row.get("name")): row for row in metadata.get("parameters") or ()
    }
    required = {
        name for name, row in parameters.items() if bool(row.get("required"))
    }
    schemas = {
        name: dict(row.get("schema") or {}) for name, row in parameters.items()
    }
    for index, template in enumerate(contract.templates):
        names = set(template.arguments)
        extra = sorted(names - set(parameters))
        if extra:
            errors.append(f"template:{index}:unknown_arguments:" + ",".join(extra))
        missing = sorted(required - names)
        if missing:
            errors.append(f"template:{index}:missing_required:" + ",".join(missing))
        for name, specification in template.arguments.items():
            if name not in parameters:
                continue
            if (
                not adapter.allow_required_omit
                and specification.kind == "omit"
                and name in required
            ):
                errors.append(f"template:{index}:required_omitted:{name}")
            if specification.kind == "literal" and not Draft202012Validator(
                schemas[name]
            ).is_valid(specification.value):
                errors.append(f"template:{index}:schema_rejected:{name}")
    supplied = {
        name
        for template in contract.templates
        for name, specification in template.arguments.items()
        if specification.kind != "omit"
    }
    argument_exports = [row for row in contract.exports if row.source == "arguments"]
    for name in sorted(supplied):
        marker = {name: "__contract_argument_marker__"}
        if not any(
            (
                _jsonpath_find(row.path, marker)
                if adapter.tolerant_jsonpath
                else _compiled_jsonpath(row.path).find(marker)
            )
            for row in argument_exports
        ):
            errors.append(f"argument:{name}:missing_export")
    response_example = (metadata.get("response_schemas") or {}).get("success")
    for export in contract.exports:
        if (
            export.source != "arguments"
            and "propagate" not in export.model_fields_set
        ):
            errors.append(f"export:{export.name}:missing_propagate")
    if (
        response_example not in (None, {}, [])
        and (
            not adapter.response_exports_only_for_public_scalars
            or _public_scalar_paths(response_example)
        )
    ):
        response_exports = [row for row in contract.exports if row.source == "response"]
        if not response_exports:
            errors.append("response:missing_exports")
        elif not any(
            (
                _jsonpath_find(row.path, response_example)
                if adapter.tolerant_jsonpath
                else _compiled_jsonpath(row.path).find(response_example)
            )
            for row in response_exports
        ):
            errors.append("response:export_paths_match_no_public_value")
        completed = normalize_mechanical_response_exports(
            [row.model_dump(mode="json", by_alias=True) for row in contract.exports],
            metadata,
        )
        if len(completed) > len(contract.exports):
            errors.append(
                f"response:missing_public_scalar_exports:"
                f"{len(completed) - len(contract.exports)}"
            )
    state_exports = [row for row in contract.exports if row.source == "state_changes"]
    if any(row.path == "$..[*]" for row in state_exports):
        errors.append("state:anonymous_recursive_wildcard")
    if state_exports and not any(
        _schema_can_select_scalar(row.value_schema) for row in state_exports
    ):
        errors.append("state:missing_scalar_export")
    return errors


def _schema_can_select_scalar(schema: Mapping[str, Any]) -> bool:
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared not in {"array", "object"}
    if isinstance(declared, list):
        return any(value not in {"array", "object"} for value in declared)
    for keyword in ("anyOf", "oneOf"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list) and any(
            isinstance(row, Mapping) and _schema_can_select_scalar(row)
            for row in alternatives
        ):
            return True
    return False


def normalize_export_paths(exports: Any) -> Any:
    """Make source-relative JSONPath explicit when a model uses record prefixes."""
    if not isinstance(exports, list):
        return exports
    prefixes = {
        "arguments": "$.arguments",
        "response": "$.output",
        "state_changes": "$.changes",
    }
    normalized: list[Any] = []
    for value in exports:
        if not isinstance(value, Mapping):
            normalized.append(value)
            continue
        row = dict(value)
        if isinstance(row.get("schema"), Mapping):
            row["schema"] = normalize_json_schema(row["schema"])
        prefix = prefixes.get(str(row.get("source")))
        path = row.get("path")
        if prefix and isinstance(path, str):
            if row.get("source") == "state_changes" and path == "$..*":
                row["path"] = "$..[*]"
                path = row["path"]
            if path == prefix:
                row["path"] = "$"
            elif path.startswith(prefix + "."):
                row["path"] = "$" + path[len(prefix):]
            elif path.startswith(prefix + "["):
                row["path"] = "$" + path[len(prefix):]
        normalized.append(row)
    return normalized


def normalize_template_arguments(templates: Any) -> Any:
    """Remove JSON null placeholders from non-literal argument specifications."""
    if not isinstance(templates, list):
        return templates
    normalized: list[Any] = []
    for value in templates:
        if not isinstance(value, Mapping):
            normalized.append(value)
            continue
        template = dict(value)
        arguments = template.get("arguments")
        if isinstance(arguments, Mapping):
            cleaned: dict[str, Any] = {}
            for name, specification in arguments.items():
                if not isinstance(specification, Mapping):
                    cleaned[str(name)] = specification
                    continue
                row = dict(specification)
                if row.get("kind") != "literal" and row.get("value") is None:
                    row.pop("value", None)
                cleaned[str(name)] = row
            template["arguments"] = cleaned
        normalized.append(template)
    return normalized


def _public_scalar_schema(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    return None


def _public_scalar_paths(
    value: Any,
    *,
    path: str = "$",
    leaf_name: str = "response_value",
) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            rows.extend(_public_scalar_paths(
                child,
                path=f"{path}[{json.dumps(name, ensure_ascii=False)}]",
                leaf_name=name,
            ))
        return rows
    if isinstance(value, list):
        for child in value:
            rows.extend(_public_scalar_paths(
                child, path=f"{path}[*]", leaf_name=leaf_name
            ))
        return rows
    schema = _public_scalar_schema(value)
    if schema is not None:
        rows.append((path, leaf_name, schema))
    return rows


def normalize_mechanical_response_exports(
    exports: Any,
    metadata: Mapping[str, Any],
) -> list[Any]:
    """Export scalar leaves explicitly present in the public success shape.

    Paths and primitive schemas are copied mechanically from public metadata;
    no runtime value, tool identity, field category, or domain rule is used.
    """

    rows = list(exports) if isinstance(exports, list | tuple) else []
    normalized = normalize_export_paths(rows)
    if not isinstance(normalized, list):
        normalized = []
    example = (metadata.get("response_schemas") or {}).get("success")
    if example in (None, {}, []):
        return normalized

    covered_paths: set[str] = set()
    for row in normalized:
        if not isinstance(row, Mapping) or row.get("source") != "response":
            continue
        try:
            validator = Draft202012Validator(dict(row.get("schema") or {}))
            for match in _compiled_jsonpath(str(row.get("path") or "")).find(example):
                if _public_scalar_schema(match.value) is not None and validator.is_valid(
                    match.value
                ):
                    covered_paths.add(str(match.full_path))
        except Exception:
            # Normal contract validation will report the malformed declaration.
            continue

    proposed: dict[str, dict[str, Any]] = {}
    for path, name, schema in _public_scalar_paths(example):
        entry = proposed.setdefault(path, {"name": name, "types": set()})
        entry["types"].add(str(schema["type"]))
    existing_names = {
        str(row.get("name"))
        for row in normalized if isinstance(row, Mapping) and row.get("name")
    }
    for path, entry in sorted(proposed.items()):
        concrete = {
            str(match.full_path) for match in _compiled_jsonpath(path).find(example)
            if _public_scalar_schema(match.value) is not None
        }
        if concrete and concrete <= covered_paths:
            continue
        export_name = str(entry["name"] or "response_value")
        if export_name in existing_names:
            export_name = f"response_{export_name}"
        suffix = 2
        while export_name in existing_names:
            export_name = f"response_{entry['name']}_{suffix}"
            suffix += 1
        types = sorted(entry["types"])
        normalized.append({
            "name": export_name,
            "source": "response",
            "path": path,
            "schema": {"type": types[0] if len(types) == 1 else types},
            # No model decision exists for a mechanically recovered field,
            # so it is propagated exhaustively.
            "propagate": True,
        })
        existing_names.add(export_name)
        covered_paths.update(concrete)
    return normalized


def remove_formally_invalid_templates(
    templates: Any,
    metadata: Mapping[str, Any],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> list[Any]:
    """Drop unusable alternatives when another valid success template remains.

    The operation only enforces the public signature and JSON Schema.  It does
    not repair a value, change literal/observation intent, or infer semantics.
    If filtering would remove every proposed success template, the original
    list is retained so normal validation reports the failure.
    """

    rows = list(templates) if isinstance(templates, list | tuple) else []
    parameters = {
        str(row.get("name")): row for row in metadata.get("parameters") or ()
    }
    required = {
        name for name, row in parameters.items() if bool(row.get("required"))
    }
    kept: list[Any] = []
    for template in rows:
        if not isinstance(template, Mapping):
            continue
        arguments = template.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        names = set(map(str, arguments))
        if names - set(parameters) or required - names:
            continue
        valid = True
        for name, specification in arguments.items():
            if not isinstance(specification, Mapping):
                valid = False
                break
            kind = specification.get("kind")
            if (
                not adapter.allow_required_omit
                and kind == "omit"
                and name in required
            ):
                valid = False
                break
            if kind == "literal" and not Draft202012Validator(
                dict(parameters[str(name)].get("schema") or {})
            ).is_valid(specification.get("value")):
                valid = False
                break
        if valid:
            kept.append(template)
    if any(
        isinstance(row, Mapping) and row.get("expected") == "success"
        for row in kept
    ):
        return kept
    return rows


def normalize_mechanical_argument_exports(
    exports: Any,
    templates: Any,
    metadata: Mapping[str, Any],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> list[Any]:
    """Fill argument exports that are uniquely determined by the signature.

    This normalization contains no tool, field, application, or domain rule.
    The model still decides whether an argument is literal, observed, or
    omitted; normalization merely preserves the provenance of any argument it
    chose to supply.
    """

    rows = list(exports) if isinstance(exports, list | tuple) else []
    normalized = normalize_export_paths(rows)
    if not isinstance(normalized, list):
        normalized = []
    supplied = {
        str(name)
        for template in templates if isinstance(template, Mapping)
        for name, specification in (template.get("arguments") or {}).items()
        if not isinstance(specification, Mapping)
        or specification.get("kind") != "omit"
    }
    parameters = {
        str(row.get("name")): row
        for row in metadata.get("parameters") or ()
    }
    existing_paths = {
        str(row.get("path"))
        for row in normalized if isinstance(row, Mapping)
        and row.get("source") == "arguments"
    }
    existing_names = {
        str(row.get("name"))
        for row in normalized if isinstance(row, Mapping) and row.get("name")
    }
    for name in sorted(supplied):
        path = (
            f"$[{json.dumps(name, ensure_ascii=False)}]"
            if adapter.bracket_nonidentifier_argument_paths
            and not name.isidentifier()
            else f"$.{name}"
        )
        if path in existing_paths or name not in parameters:
            continue
        export_name = name
        if export_name in existing_names:
            export_name = f"argument_{name}"
        suffix = 2
        while export_name in existing_names:
            export_name = f"argument_{name}_{suffix}"
            suffix += 1
        normalized.append({
            "name": export_name,
            "source": "arguments",
            "path": path,
            "schema": normalize_json_schema(parameters[name].get("schema") or {}),
            # Argument exports forward their original observation provenance;
            # they are not newly produced low-information values.
            "propagate": True,
        })
        existing_names.add(export_name)
        existing_paths.add(path)
    return normalized


def parse_and_validate_contract(
    raw: str,
    metadata: Mapping[str, Any],
    *,
    revision: int,
    template_policy: TemplatePolicy = MINIMAL_SUCCESS_TEMPLATE_POLICY,
    legacy_export_propagation: bool = False,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> tuple[ToolCallContract | None, list[str]]:
    """Parse one model response and apply domain-free normalizations."""

    if template_policy not in TEMPLATE_POLICIES:
        raise ValueError(f"unknown template policy: {template_policy}")

    tool_id = str(metadata.get("tool") or metadata.get("name") or "")
    try:
        parsed = dict(_extract_object(raw))
        parsed["version"] = CONTRACT_VERSION
        parsed["tool"] = tool_id
        parsed["revision"] = revision
        parsed["templates"] = normalize_template_arguments(parsed.get("templates"))
        if legacy_export_propagation and isinstance(parsed.get("exports"), list):
            for export in parsed["exports"]:
                if isinstance(export, dict):
                    export.setdefault("propagate", True)
        parsed["templates"] = remove_formally_invalid_templates(
            parsed.get("templates"), metadata, adapter=adapter
        )
        parsed["exports"] = normalize_mechanical_argument_exports(
            parsed.get("exports"),
            parsed.get("templates"),
            metadata,
            adapter=adapter,
        )
        parsed["exports"] = normalize_mechanical_response_exports(
            parsed.get("exports"), metadata
        )
        if isinstance(parsed.get("success_schema"), Mapping):
            parsed["success_schema"] = normalize_outcome_schema(
                parsed["success_schema"], expected="success"
            )
        if isinstance(parsed.get("failure_schema"), Mapping):
            parsed["failure_schema"] = normalize_outcome_schema(
                parsed["failure_schema"], expected="failure"
            )
        if (
            adapter.complete_boolean_business_outcomes
            and isinstance(parsed.get("success_schema"), Mapping)
            and isinstance(parsed.get("failure_schema"), Mapping)
        ):
            parsed["failure_schema"] = complete_boolean_business_failure(
                parsed["success_schema"], parsed["failure_schema"]
            )
        contract = ToolCallContract.model_validate(parsed)
        errors = validate_contract_metadata(contract, metadata, adapter=adapter)
        if not errors and template_policy == MINIMAL_SUCCESS_TEMPLATE_POLICY:
            parameters = list(metadata.get("parameters") or ())
            contract = with_minimal_success_baseline(
                contract,
                parameter_names=[str(row.get("name")) for row in parameters],
                required_parameters={
                    str(row.get("name"))
                    for row in parameters
                    if bool(row.get("required"))
                },
            )
            errors = validate_contract_metadata(contract, metadata, adapter=adapter)
        return (None, errors) if errors else (contract, [])
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        return None, [f"{type(exc).__name__}:{str(exc)[:800]}"]


_JSONPATH_PARSER: ExtendedJsonPathParser | None = None
_JSONPATH_PARSER_LOCK = threading.RLock()


@lru_cache(maxsize=None)
def _compiled_jsonpath(path: str) -> Any:
    """Compile with one reusable parser; PLY parser construction is expensive."""

    global _JSONPATH_PARSER
    with _JSONPATH_PARSER_LOCK:
        if _JSONPATH_PARSER is None:
            _JSONPATH_PARSER = ExtendedJsonPathParser()
        return _JSONPATH_PARSER.parse(path)


def _jsonpath_find(path: str, value: Any) -> list[Any]:
    """A valid selector meeting a different shape has no matches."""

    try:
        return list(_compiled_jsonpath(path).find(value))
    except (TypeError, IndexError, KeyError):
        return []


def select_export_values(export: ToolExport, value: Any) -> list[tuple[str, Any]]:
    """Execute one declared export and return concrete match paths and values."""
    validator = Draft202012Validator(export.value_schema)
    if export.path == "$..[*]":
        selected: list[tuple[str, Any]] = []

        def visit(child: Any, path: str) -> None:
            if isinstance(child, Mapping):
                for key, value_at_key in child.items():
                    visit(
                        value_at_key,
                        f"{path}[{json.dumps(str(key), ensure_ascii=False)}]",
                    )
                return
            if isinstance(child, list | tuple):
                for index, value_at_index in enumerate(child):
                    visit(value_at_index, f"{path}[{index}]")
                return
            if validator.is_valid(child):
                selected.append((path, child))

        visit(value, "$")
        return selected
    selected: list[tuple[str, Any]] = []
    matches = _jsonpath_find(export.path, value)
    for match in matches:
        if validator.is_valid(match.value):
            path = str(match.full_path)
            selected.append((f"$.{path}" if path and path != "$" else "$", match.value))
    return selected


@dataclass
class GenerationResult:
    tool: str
    contract: ToolCallContract | None
    audit: dict[str, Any]
    error: str | None = None
    history: tuple[dict[str, Any], ...] = ()


class ContractGenerator:
    def __init__(
        self,
        llm: ContractLLM,
        *,
        feedback_policy: FeedbackPolicy = DEFAULT_FEEDBACK_POLICY,
        template_policy: TemplatePolicy = MINIMAL_SUCCESS_TEMPLATE_POLICY,
        adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
    ) -> None:
        if feedback_policy not in FEEDBACK_POLICIES:
            raise ValueError(f"unknown feedback policy: {feedback_policy}")
        if template_policy not in TEMPLATE_POLICIES:
            raise ValueError(f"unknown template policy: {template_policy}")
        self.llm = llm
        self.feedback_policy = feedback_policy
        self.template_policy = template_policy
        self.adapter = adapter
        self.prompt_version = adapter.prompt_version

    def generate(
        self,
        metadata: Mapping[str, Any],
        *,
        revision: int = 1,
        previous: ToolCallContract | None = None,
        feedback: Sequence[Mapping[str, Any]] = (),
    ) -> GenerationResult:
        tool_id = str(metadata.get("tool") or metadata.get("name") or "")
        messages = self.adapter.build_prompt(
            metadata,
            revision=revision,
            previous=(
                previous.model_dump(mode="json") if previous is not None else None
            ),
            feedback=feedback,
            feedback_policy=self.feedback_policy,
        )
        raw = ""
        usage: Mapping[str, Any] = {}
        errors: list[str] = []
        contract: ToolCallContract | None = None
        try:
            raw, usage = self.llm.complete(messages)
            contract, errors = parse_and_validate_contract(
                raw,
                metadata,
                revision=revision,
                template_policy=self.template_policy,
                adapter=self.adapter,
            )
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            errors = [f"{type(exc).__name__}:{str(exc)[:800]}"]
        complete_runtime_feedback = (
            self.feedback_policy == "complete-same-tool"
            and any("invocation_result" in row for row in feedback)
        )
        audit = {
            "revision": revision,
            "raw_response": "" if complete_runtime_feedback else raw,
            "raw_response_retained": not complete_runtime_feedback,
            "usage": dict(usage),
            "validation_errors": errors,
            "mechanical_normalization": True,
            "prompt_version": self.prompt_version,
            "feedback_policy": self.feedback_policy,
            "template_policy": self.template_policy,
            "generation_kind": "initial" if previous is None else "repair",
            "generation_outcome": (
                "candidate_valid" if contract is not None else "candidate_invalid"
            ),
            "created_at_unix": int(time.time()),
        }
        return GenerationResult(
            tool=tool_id,
            contract=contract,
            audit=audit,
            error=";".join(errors) if errors else None,
            history=(audit,),
        )

    def generate_many(
        self,
        metadata: Mapping[str, Mapping[str, Any]],
        *,
        concurrency: int = 8,
        progress: Any = None,
    ) -> dict[str, GenerationResult]:
        results: dict[str, GenerationResult] = {}

        def generate_with_retries(
            tool_id: str, row: Mapping[str, Any]
        ) -> GenerationResult:
            result = self.generate(row)
            history = list(result.history)
            while (
                result.contract is None
                and len(history) < MAX_LLM_CALLS_PER_TOOL
            ):
                result = self.generate(
                    row,
                    revision=len(history) + 1,
                    feedback=({
                        "event": "contract_validation",
                        "validation_errors": result.audit.get(
                            "validation_errors", []
                        ),
                    },),
                )
                history.extend(result.history)
            result.history = tuple(history)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(generate_with_retries, tool_id, row): tool_id
                for tool_id, row in metadata.items()
            }
            for future in concurrent.futures.as_completed(futures):
                tool_id = futures[future]
                result = future.result()
                results[tool_id] = result
                if progress is not None:
                    progress(result)
        return results


_STATE_EXPORT_REPAIR_EVENTS = {
    "state_observed_without_export",
    "state_export_matched_no_values",
}


def _state_scalar_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def normalize_state_export_repair(
    previous: ToolCallContract,
    candidate: ToolCallContract,
    feedback: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> tuple[ToolCallContract | None, list[str]]:
    """Accept only the field declarations from a state-export JIT repair.

    The same-tool state sample can contain concrete runtime scalars.  Templates
    and outcome schemas therefore stay byte-for-byte equivalent to the prior
    contract.  Only explicit state export names and paths come from the model;
    primitive schemas are reconstructed mechanically from the selected sample
    values, so runtime literals cannot leak into the persisted contract.
    """

    invocation = feedback.get("invocation_result")
    context = (
        invocation.get("changes")
        if isinstance(invocation, Mapping) else None
    )
    if context is None:
        return None, ["state:missing_invocation_changes"]
    errors: list[str] = []
    state_exports: list[dict[str, Any]] = []
    for export in candidate.exports:
        if export.source != "state_changes":
            continue
        if export.path == "$..[*]":
            errors.append(f"state:anonymous_recursive_wildcard:{export.name}")
            continue
        try:
            matches = _compiled_jsonpath(export.path).find(context)
        except Exception:
            matches = []
        types = sorted({
            value_type
            for match in matches
            for value_type in (_state_scalar_type(match.value),)
            if value_type is not None
        })
        if not types:
            errors.append(f"state:path_matches_no_scalar:{export.name}")
            continue
        state_exports.append({
            "name": export.name,
            "source": "state_changes",
            "path": export.path,
            "schema": {"type": types[0] if len(types) == 1 else types},
            "propagate": export.propagate,
        })
    if not state_exports:
        errors.append("state:no_explicit_named_exports")
    if errors:
        return None, sorted(set(errors))

    payload = previous.model_dump(mode="json", by_alias=True)
    payload["revision"] = candidate.revision
    payload["exports"] = [
        row.model_dump(mode="json", by_alias=True)
        for row in previous.exports
        if row.source != "state_changes"
    ] + state_exports
    try:
        merged = ToolCallContract.model_validate(payload)
    except ValidationError as exc:
        return None, [f"state:merge_validation:{str(exc)[:800]}"]
    errors = validate_contract_metadata(merged, metadata, adapter=adapter)
    return (None, errors) if errors else (merged, [])


def normalize_non_state_repair(
    previous: ToolCallContract,
    candidate: ToolCallContract,
    metadata: Mapping[str, Any],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> tuple[ToolCallContract | None, list[str]]:
    """Keep learned state fields across unrelated contract repairs.

    Literal/outcome feedback contains no new state sample that could justify
    deleting or rewriting a previously validated state selector.  The model
    may still repair templates and outcome schemas, while existing state
    exports are copied byte-for-byte from the active contract.
    """

    previous_state = tuple(
        row for row in previous.exports if row.source == "state_changes"
    )
    if not previous_state:
        return candidate, []
    state_names = {row.name for row in previous_state}
    candidate_non_state = tuple(
        row for row in candidate.exports if row.source != "state_changes"
    )
    conflicts = sorted(
        row.name for row in candidate_non_state if row.name in state_names
    )
    if conflicts:
        return None, [
            f"state:preserved_name_conflict:{name}" for name in conflicts
        ]
    payload = candidate.model_dump(mode="json", by_alias=True)
    payload["exports"] = [
        row.model_dump(mode="json", by_alias=True)
        for row in (*candidate_non_state, *previous_state)
    ]
    try:
        merged = ToolCallContract.model_validate(payload)
    except ValidationError as exc:
        return None, [f"state:preserve_validation:{str(exc)[:800]}"]
    errors = validate_contract_metadata(merged, metadata, adapter=adapter)
    return (None, errors) if errors else (merged, [])


@dataclass
class OnlineContractSet:
    """Mutable contracts with at most three current-prompt calls per tool."""

    manifest: ContractManifest
    metadata: Mapping[str, Mapping[str, Any]]
    generator: ContractGenerator | None = None
    count_all_history: bool = False
    max_total_llm_calls: int | None = None
    _feedback_seen: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generator is not None:
            select_template_policy(
                self.manifest,
                self.generator.template_policy,
            )
            self.manifest.feedback_policy = self.generator.feedback_policy

    def get(self, tool: str) -> ToolCallContract | None:
        return self.manifest.contracts.get(tool)

    def llm_calls(self, tool: str) -> int:
        if self.count_all_history:
            return len(self.manifest.audit.get(tool, ()))
        prompt_version = (
            self.generator.prompt_version
            if self.generator is not None
            else self.manifest.prompt_version
        )
        return sum(
            str(row.get("prompt_version") or self.manifest.prompt_version)
            == prompt_version
            for row in self.manifest.audit.get(tool, ())
        )

    def total_llm_calls(self) -> int:
        return sum(self.llm_calls(tool) for tool in self.manifest.audit)

    def _llm_budget_available(self, tool: str) -> bool:
        return (
            self.llm_calls(tool) < MAX_LLM_CALLS_PER_TOOL
            and (
                self.max_total_llm_calls is None
                or self.total_llm_calls() < self.max_total_llm_calls
            )
        )

    def repair(
        self, tool: str, feedback: Mapping[str, Any]
    ) -> ToolCallContract | None:
        current = self.get(tool)
        if self.generator is None or current is None:
            return current
        if not self._llm_budget_available(tool):
            return current
        feedback_policy = self.generator.feedback_policy
        feedback_key = scalar_key(
            redact_runtime_feedback(
                feedback, feedback_policy=feedback_policy
            )
        )
        seen = self._feedback_seen.setdefault(tool, set())
        if feedback_key in seen:
            return current
        seen.add(feedback_key)
        audits = self.manifest.audit.setdefault(tool, [])
        next_feedback: Mapping[str, Any] = feedback
        state_export_repair = str(feedback.get("event") or "") in (
            _STATE_EXPORT_REPAIR_EVENTS
        )
        result: GenerationResult | None = None
        while self._llm_budget_available(tool):
            result = self.generator.generate(
                self.metadata[tool],
                revision=current.revision + 1,
                previous=current,
                feedback=(next_feedback,),
            )
            if state_export_repair and result.contract is not None:
                normalized, state_errors = normalize_state_export_repair(
                    current,
                    result.contract,
                    feedback,
                    self.metadata[tool],
                    adapter=self.generator.adapter,
                )
                if state_errors:
                    combined = sorted(set(
                        list(result.audit.get("validation_errors") or ())
                        + state_errors
                    ))
                    result.audit["validation_errors"] = combined
                    result.error = ";".join(combined)
                    result.contract = None
                else:
                    result.contract = normalized
            elif result.contract is not None:
                normalized, state_errors = normalize_non_state_repair(
                    current,
                    result.contract,
                    self.metadata[tool],
                    adapter=self.generator.adapter,
                )
                if state_errors:
                    combined = sorted(set(
                        list(result.audit.get("validation_errors") or ())
                        + state_errors
                    ))
                    result.audit["validation_errors"] = combined
                    result.error = ";".join(combined)
                    result.contract = None
                else:
                    result.contract = normalized
            if result.contract is not None:
                literal_errors = feedback_derived_literal_errors(
                    current,
                    result.contract,
                    self.metadata[tool],
                    (feedback,),
                    feedback_policy=feedback_policy,
                )
                if literal_errors:
                    combined = sorted(set(
                        list(result.audit.get("validation_errors") or ())
                        + literal_errors
                    ))
                    result.audit["validation_errors"] = combined
                    result.audit["raw_response"] = ""
                    result.audit["raw_response_retained"] = False
                    result.audit["raw_response_redacted_reason"] = (
                        "feedback_derived_literal"
                    )
                    result.error = ";".join(combined)
                    result.contract = None
            result.audit["repair_outcome"] = (
                "accepted" if result.contract is not None else "rejected"
            )
            audits.append(result.audit)
            if result.contract is not None:
                self.manifest.contracts[tool] = result.contract
                self.manifest.unknown.pop(tool, None)
                break
            validation_errors = result.audit.get("validation_errors", [])
            next_feedback = (
                {
                    **dict(feedback),
                    "validation_errors": validation_errors,
                    "conclusion": "return_valid_explicit_named_state_exports",
                }
                if state_export_repair else {
                    "event": "contract_validation",
                    "validation_errors": validation_errors,
                }
            )
        return self.get(tool)


def manifest_from_results(
    results: Mapping[str, GenerationResult],
    *,
    model: str,
    metadata_id: str,
) -> ContractManifest:
    prompt_versions = {
        str(row.get("prompt_version"))
        for result in results.values()
        for row in (result.history or (result.audit,))
        if row.get("prompt_version")
    }
    if len(prompt_versions) > 1:
        raise ValueError(
            "one manifest cannot mix prompt versions: "
            + ",".join(sorted(prompt_versions))
        )
    prompt_version = (
        next(iter(prompt_versions)) if prompt_versions else PROMPT_VERSION
    )
    feedback_policies = {
        str(row.get("feedback_policy"))
        for result in results.values()
        for row in (result.history or (result.audit,))
        if row.get("feedback_policy")
    }
    if len(feedback_policies) > 1:
        raise ValueError(
            "one manifest cannot mix feedback policies: "
            + ",".join(sorted(feedback_policies))
        )
    feedback_policy = (
        next(iter(feedback_policies))
        if feedback_policies else DEFAULT_FEEDBACK_POLICY
    )
    if feedback_policy not in FEEDBACK_POLICIES:
        raise ValueError(f"unknown feedback policy: {feedback_policy}")
    template_policies = {
        str(row.get("template_policy"))
        for result in results.values()
        for row in (result.history or (result.audit,))
        if row.get("template_policy")
    }
    if len(template_policies) > 1:
        raise ValueError(
            "one manifest cannot mix template policies: "
            + ",".join(sorted(template_policies))
        )
    template_policy = (
        next(iter(template_policies))
        if template_policies else MINIMAL_SUCCESS_TEMPLATE_POLICY
    )
    if template_policy not in TEMPLATE_POLICIES:
        raise ValueError(f"unknown template policy: {template_policy}")
    return ContractManifest(
        prompt_version=prompt_version,
        feedback_policy=feedback_policy,
        template_policy=template_policy,
        model=model,
        metadata_id=metadata_id,
        contracts={
            tool: result.contract
            for tool, result in results.items()
            if result.contract is not None
        },
        unknown={
            tool: result.error or "contract_generation_failed"
            for tool, result in results.items()
            if result.contract is None
        },
        audit={
            tool: list(result.history or (result.audit,))
            for tool, result in results.items()
        },
    )


def renormalize_manifest(
    manifest: ContractManifest,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> tuple[ContractManifest, dict[str, Any]]:
    """Reapply mechanical normalization to existing LLM responses.

    This performs zero LLM calls. Unknown tools are recovered only when one of
    their already-recorded raw responses becomes valid after domain-free
    normalization of signature-derived argument exports, public response
    scalar exports, or formally invalid backup templates.
    """

    contracts = dict(manifest.contracts)
    unknown = dict(manifest.unknown)
    source_prompt_version = manifest.prompt_version
    active_prompt_version = adapter.prompt_version
    audit = {
        tool: [
            {
                **dict(row),
                "prompt_version": str(
                    row.get("prompt_version") or source_prompt_version
                ),
                "feedback_policy": str(
                    row.get("feedback_policy") or manifest.feedback_policy
                ),
                "template_policy": str(
                    row.get("template_policy") or manifest.template_policy
                ),
            }
            for row in rows
        ]
        for tool, rows in manifest.audit.items()
    }
    recovered: list[str] = []
    normalized_existing: list[str] = []
    remaining_errors: dict[str, list[str]] = {}
    for tool, row in sorted(metadata.items()):
        current = contracts.get(tool)
        if current is not None:
            payload = current.model_dump(mode="json")
            payload["exports"] = normalize_mechanical_argument_exports(
                payload.get("exports"),
                payload.get("templates"),
                row,
                adapter=adapter,
            )
            payload["exports"] = normalize_mechanical_response_exports(
                payload.get("exports"), row
            )
            payload["success_schema"] = normalize_outcome_schema(
                payload["success_schema"], expected="success"
            )
            payload["failure_schema"] = normalize_outcome_schema(
                payload["failure_schema"], expected="failure"
            )
            if adapter.complete_boolean_business_outcomes:
                payload["failure_schema"] = complete_boolean_business_failure(
                    payload["success_schema"], payload["failure_schema"]
                )
            normalized = ToolCallContract.model_validate(payload)
            errors = validate_contract_metadata(normalized, row, adapter=adapter)
            if errors:
                remaining_errors[tool] = errors
            else:
                contracts[tool] = normalized
                if normalized.model_dump(mode="json") != current.model_dump(mode="json"):
                    normalized_existing.append(tool)
            continue
        errors: list[str] = []
        for audit_row in reversed(audit.get(tool, ())):
            raw = str(audit_row.get("raw_response") or "")
            if not raw:
                continue
            revision = int(audit_row.get("revision") or 1)
            candidate, errors = parse_and_validate_contract(
                raw,
                row,
                revision=revision,
                template_policy=manifest.template_policy,
                legacy_export_propagation=(
                    str(audit_row.get("prompt_version") or source_prompt_version)
                    != adapter.prompt_version
                ),
                adapter=adapter,
            )
            if candidate is not None:
                contracts[tool] = candidate
                unknown.pop(tool, None)
                recovered.append(tool)
                break
        if tool not in contracts:
            remaining_errors[tool] = errors or [unknown.get(tool, "no_raw_response")]
    result = manifest.model_copy(update={
        "contracts": contracts,
        "unknown": unknown,
        "audit": audit,
        "prompt_version": active_prompt_version,
        "generated_at_unix": int(time.time()),
    })
    report = {
        "llm_calls_added": 0,
        "valid_before": len(manifest.contracts),
        "valid_after": len(result.contracts),
        "unknown_before": len(manifest.unknown),
        "unknown_after": len(result.unknown),
        "source_llm_calls": sum(len(rows) for rows in manifest.audit.values()),
        "active_prompt_llm_calls": sum(
            str(row.get("prompt_version") or result.prompt_version)
            == adapter.prompt_version
            for tool, rows in result.audit.items()
            for row in rows
        ),
        "recovered": recovered,
        "normalized_existing": normalized_existing,
        "remaining_error_counts": dict(sorted(
            {
                error: sum(error in values for values in remaining_errors.values())
                for error in {item for values in remaining_errors.values() for item in values}
            }.items()
        )),
    }
    return result, report


def load_active_contract_manifest(
    path: str | Path,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    metadata_id: str,
    adapter: ContractAdapter = DEFAULT_CONTRACT_ADAPTER,
) -> tuple[ContractManifest, dict[str, Any]]:
    """Load working or committed contracts and apply mechanical normalization."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        isinstance(payload, Mapping)
        and "contracts" in payload
        and "metadata_id" in payload
    ):
        manifest = ContractManifest.model_validate(payload)
        if manifest.metadata_id != metadata_id:
            raise ValueError("contract metadata identity does not match")
    else:
        manifest = ContractManifest(
            prompt_version=adapter.prompt_version,
            feedback_policy=DEFAULT_FEEDBACK_POLICY,
            template_policy=MINIMAL_SUCCESS_TEMPLATE_POLICY,
            model=DEFAULT_MODEL,
            metadata_id=metadata_id,
            contracts=load_contracts(path),
        )
    source_prompt_version = manifest.prompt_version
    active, report = renormalize_manifest(
        manifest, metadata, adapter=adapter
    )
    return active, {
        **report,
        "source_prompt_version": source_prompt_version,
        "active_prompt_version": active.prompt_version,
    }
