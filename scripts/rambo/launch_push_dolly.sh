#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_WORKSPACE="${WORKSPACE_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)}"
source "${LOCAL_WORKSPACE}/workspace.env"
unset LD_PRELOAD
# Acceptance is scoped to this launch; no user profile or persistent setting.
export OMNI_KIT_ACCEPT_EULA=Y
exec bash "${SCRIPT_DIR}/run.sh" "${SCRIPT_DIR}/teleop_push_dolly.py" --viz kit "$@"
