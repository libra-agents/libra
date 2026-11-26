"""Generate fresh contracts from public tool metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from retrieval.adapters.appworld import load_appworld_tools
from retrieval.adapters.appworld_contracts import APPWORLD_CONTRACT_ADAPTER
from retrieval.adapters.registry_contracts import (
    REGISTRY_CONTRACT_ADAPTER,
    STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER,
)
from retrieval.core.contracts import (
    MAX_LLM_CALLS_PER_TOOL,
    ContractAdapter,
    ContractGenerator,
    DeepSeekFlash,
    ToolCallContract,
    manifest_from_results,
    parse_and_validate_contract,
)
from retrieval.core.registry import load_registry_tools


def _fallback(
    metadata: Mapping[str, Any], adapter: ContractAdapter
) -> ToolCallContract:
    arguments = {
        str(parameter["name"]): {
            "kind": "observation" if parameter.get("required") else "omit"
        }
        for parameter in metadata.get("parameters", [])
    }
    payload = {
        "templates": [{"arguments": arguments, "expected": "success"}],
        "exports": [],
        "success_schema": {
            "type": "object",
            "properties": {"status": {"const": "success"}},
            "required": ["status"],
        },
        "failure_schema": {
            "type": "object",
            "properties": {"status": {"enum": ["rejected", "error"]}},
            "required": ["status"],
        },
    }
    contract, errors = parse_and_validate_contract(
        json.dumps(payload),
        metadata,
        revision=MAX_LLM_CALLS_PER_TOOL,
        adapter=adapter,
    )
    if contract is None:
        raise ValueError("invalid fallback contract: " + ";".join(errors))
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("appworld", "scitool", "bfcl"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.benchmark == "appworld":
        _, metadata, metadata_id, _ = load_appworld_tools(
            args.data_dir / "api_metadata.json"
        )
        adapter = APPWORLD_CONTRACT_ADAPTER
        allow_fallback = False
    else:
        _, metadata, metadata_id = load_registry_tools(
            args.data_dir / "tool_registry.json"
        )
        adapter = (
            STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER
            if args.benchmark == "bfcl"
            else REGISTRY_CONTRACT_ADAPTER
        )
        allow_fallback = True

    llm = DeepSeekFlash(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    )
    generator = ContractGenerator(llm, adapter=adapter)
    completed = 0

    def progress(result: object) -> None:
        nonlocal completed
        completed += 1
        print(
            f"[{completed}/{len(metadata)}] "
            f"{getattr(result, 'tool', 'unknown')}: "
            f"{'valid' if getattr(result, 'contract', None) else 'unknown'}",
            flush=True,
        )

    results = generator.generate_many(metadata, concurrency=16, progress=progress)
    if allow_fallback:
        for tool, result in results.items():
            if result.contract is None:
                result.contract = _fallback(metadata[tool], adapter)
                result.audit["generation_outcome"] = "mechanical_fallback"
                result.audit["fallback_after_error"] = result.error

    manifest = manifest_from_results(
        results,
        model=llm.model,
        metadata_id=metadata_id,
    )
    if allow_fallback:
        contracts = {}
        for tool, contract in manifest.contracts.items():
            payload = contract.model_dump(mode="json", by_alias=True)
            for export in payload["exports"]:
                export["propagate"] = True
            contracts[tool] = ToolCallContract.model_validate(payload)
        manifest = manifest.model_copy(update={"contracts": contracts})
    manifest.write(args.output)
    print(json.dumps({
        "benchmark": args.benchmark,
        "contracts": len(manifest.contracts),
        "unknown": len(manifest.unknown),
        "llm_calls": sum(len(rows) for rows in manifest.audit.values()),
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
