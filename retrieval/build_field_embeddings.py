"""Build one job-local Qwen index for contract fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.adapters.appworld import load_appworld_tools
from retrieval.adapters.appworld_contracts import APPWORLD_CONTRACT_ADAPTER
from retrieval.adapters.registry_contracts import (
    REGISTRY_CONTRACT_ADAPTER,
    STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER,
)
from retrieval.core.contracts import load_active_contract_manifest
from retrieval.core.fields import (
    QwenVllmTextEmbedder,
    SQLiteFieldEmbeddingStore,
    SQLiteFieldPriorityIndex,
)
from retrieval.core.registry import load_registry_tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("appworld", "scitool", "bfcl"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptor", choices=("field", "context"), required=True)
    parser.add_argument("--embedding-base-url", required=True)
    args = parser.parse_args()

    if args.benchmark == "appworld":
        definitions, metadata, metadata_id, _ = load_appworld_tools(
            args.data_dir / "api_metadata.json"
        )
        adapter = APPWORLD_CONTRACT_ADAPTER
    else:
        definitions, metadata, metadata_id = load_registry_tools(
            args.data_dir / "tool_registry.json"
        )
        adapter = (
            STATEFUL_BUSINESS_OUTCOME_REGISTRY_CONTRACT_ADAPTER
            if args.benchmark == "bfcl"
            else REGISTRY_CONTRACT_ADAPTER
        )
    manifest, _ = load_active_contract_manifest(
        args.contracts,
        metadata,
        metadata_id=metadata_id,
        adapter=adapter,
    )
    store = SQLiteFieldEmbeddingStore(
        args.output,
        create=True,
        reset=True,
        base_url=args.embedding_base_url,
        descriptor_mode=args.descriptor,
    )
    try:
        index = SQLiteFieldPriorityIndex(
            store,
            definitions,
            embedder=QwenVllmTextEmbedder(
                base_url=args.embedding_base_url,
                max_concurrent=20,
            ),
            require_complete=False,
        )
        added = index.ensure_manifest(manifest.contracts)
        store.finalize_initial()
        print(json.dumps({
            "benchmark": args.benchmark,
            "descriptor": args.descriptor,
            "descriptors": added,
            "output": str(args.output),
        }))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
