#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI="${1:-podman}"
echo "Building scitool-tool-server image with $CLI..."
$CLI build -t scitool-tool-server:latest -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
echo "Done. Run: $CLI run --rm -p 30100:8100 -v /path/to/ToolsAgent:/app/ToolsAgent:ro scitool-tool-server:latest"
