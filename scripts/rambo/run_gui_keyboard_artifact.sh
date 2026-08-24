#!/usr/bin/env bash
# Run the bounded M8 GUI-keyboard gate and seal evidence after Kit closes.
#
# The Isaac Sim child is launched through run60.sh, preserving the canonical
# PhysX/visualizer contract.  The pure-Python finalizer runs only after that
# child has returned and receives no OMNI_KIT_ACCEPT_EULA environment variable.

set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="${RAMBO_VENV:-/workspace/venvs/rambo60}"
readonly CONTRACT="${SCRIPT_DIR}/run60_contract.sh"
readonly TELEOP_SCRIPT="${SCRIPT_DIR}/teleop_loco_manip.py"
readonly FINALIZER="${SCRIPT_DIR}/finalize_gui_keyboard_artifact.py"

usage() {
    echo "Usage: $0 <teleop_loco_manip.py arguments including exactly one --gui-artifact-dir and --viz kit>" >&2
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 2
fi
if [[ ! -r "${CONTRACT}" ]]; then
    echo "Missing RAMBO launch contract: ${CONTRACT}" >&2
    exit 1
fi
if [[ ! -f "${TELEOP_SCRIPT}" ]]; then
    echo "Missing M8 teleoperation script: ${TELEOP_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${FINALIZER}" ]]; then
    echo "Missing M8 GUI artifact finalizer: ${FINALIZER}" >&2
    exit 1
fi

# Reject unsafe visualizer or physics overrides before any Python work.  The
# canonical validator permits both none and kit generally; this dedicated M8
# runner narrows that to kit below.
# shellcheck source=run60_contract.sh
source "${CONTRACT}"
rambo_validate_launch_contract "$@" || exit $?

artifact_dir=""
viz_value=""
arguments=("$@")
index=0
while (( index < ${#arguments[@]} )); do
    argument="${arguments[index]}"
    case "${argument}" in
        --gui-artifact-dir)
            ((index += 1))
            if (( index >= ${#arguments[@]} )); then
                echo "--gui-artifact-dir requires a value" >&2
                exit 2
            fi
            candidate="${arguments[index]}"
            ;;
        --gui-artifact-dir=*)
            candidate="${argument#--gui-artifact-dir=}"
            ;;
        --viz)
            ((index += 1))
            if (( index >= ${#arguments[@]} )); then
                echo "--viz requires a value" >&2
                exit 2
            fi
            viz_value="${arguments[index]}"
            ((index += 1))
            continue
            ;;
        --viz=*)
            viz_value="${argument#--viz=}"
            ((index += 1))
            continue
            ;;
        *)
            ((index += 1))
            continue
            ;;
    esac
    if [[ -z "${candidate}" ]]; then
        echo "--gui-artifact-dir must not be empty" >&2
        exit 2
    fi
    if [[ -n "${artifact_dir}" ]]; then
        echo "Exactly one --gui-artifact-dir is required" >&2
        exit 2
    fi
    artifact_dir="${candidate}"
    ((index += 1))
done

if [[ -z "${artifact_dir}" ]]; then
    echo "This runner requires exactly one --gui-artifact-dir to record the real post-close exit status" >&2
    exit 2
fi
if [[ "${viz_value}" != "kit" ]]; then
    echo "M8 GUI keyboard artifacts require explicit --viz kit" >&2
    exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "RAMBO Python 3.12 venv was not found: ${VENV_DIR}/bin/python" >&2
    exit 1
fi

set +e
"${SCRIPT_DIR}/run60.sh" "${TELEOP_SCRIPT}" "$@"
child_status=$?
set -e

# EULA acceptance is scoped to the Kit child.  The finalizer is standard
# library only and should never receive that process-level setting.
unset OMNI_KIT_ACCEPT_EULA || true
set +e
"${VENV_DIR}/bin/python" "${FINALIZER}" \
    --gui-artifact-dir "${artifact_dir}" --process-exit-status "${child_status}"
finalizer_status=$?
set -e
if [[ "${finalizer_status}" -ne 0 ]]; then
    echo "M8 GUI artifact finalization failed after child exit status ${child_status}" >&2
    exit "${finalizer_status}"
fi
exit "${child_status}"
