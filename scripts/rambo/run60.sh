#!/usr/bin/env bash
# Compatibility-named entry point for the pinned Isaac Sim 6 RAMBO runtime.
#
# This dedicated simulator wrapper rejects unsafe visualizer and physics
# override arguments before Python can start.  ``run.sh`` remains the generic
# CUDA-aware Python wrapper for qpth and pytest commands.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CONTRACT="${SCRIPT_DIR}/run60_contract.sh"

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <RAMBO simulator Python script and arguments including exactly one --viz none|kit>" >&2
    exit 2
fi
if [[ ! -r "${CONTRACT}" ]]; then
    echo "Missing RAMBO launch contract: ${CONTRACT}" >&2
    exit 1
fi

# shellcheck source=run60_contract.sh
source "${CONTRACT}"
rambo_validate_launch_contract "$@" || exit $?
exec "${SCRIPT_DIR}/run.sh" "$@"
