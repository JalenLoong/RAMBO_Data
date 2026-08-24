#!/usr/bin/env bash
# Compatibility-named entry point for the pinned Isaac Sim 6 RAMBO runtime.
#
# All production Python launchers invoked through this wrapper enforce the
# explicit ``--viz none|kit`` and actual-PhysX contracts themselves.  Keeping
# the CUDA library setup in ``run.sh`` makes the command usable for both
# simulator processes and qpth-only diagnostic tests.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run.sh" "$@"
