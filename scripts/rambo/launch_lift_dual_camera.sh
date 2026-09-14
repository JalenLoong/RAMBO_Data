#!/usr/bin/env bash
# Single-environment quadruped teleop with robot-mounted ego/task cameras.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
source "${WORKSPACE_ROOT:-${LOCAL_ROOT}}/workspace.env"
unset LD_PRELOAD
exec env OMNI_KIT_ACCEPT_EULA=Y "${SCRIPT_DIR}/run60.sh" \
    "${SCRIPT_DIR}/teleop_loco_manip.py" \
    --task Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0 \
    --checkpoint "${RAMBO_CHECKPOINT_ROOT}/quadruped/model_2000.pt" \
    --camera-setup robot-dual-v3 --view task --viz kit "$@"
