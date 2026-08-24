#!/usr/bin/env bash
# Run the bounded M8 GUI-keyboard gate and seal evidence after Kit closes.
#
# The Isaac Sim child is launched through run60.sh, preserving the canonical
# PhysX/visualizer contract. The pure-Python finalizer, interactive post-close
# recorder, and offline validator run only after that child has returned and
# receive no OMNI_KIT_ACCEPT_EULA environment variable.

set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="${RAMBO_VENV:-/workspace/venvs/rambo60}"
readonly CONTRACT="${SCRIPT_DIR}/run60_contract.sh"
readonly TELEOP_SCRIPT="${SCRIPT_DIR}/teleop_loco_manip.py"
readonly FINALIZER="${SCRIPT_DIR}/finalize_gui_keyboard_artifact.py"
readonly RECORDER="${SCRIPT_DIR}/record_gui_keyboard_attestation.py"
readonly VALIDATOR="${SCRIPT_DIR}/validate_gui_keyboard_artifact.py"

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
for required_script in "${FINALIZER}" "${RECORDER}" "${VALIDATOR}"; do
    if [[ ! -f "${required_script}" ]]; then
        echo "Missing M8 GUI post-close helper: ${required_script}" >&2
        exit 1
    fi
done

# Reject unsafe visualizer or physics overrides before any Python work. The
# canonical validator permits both none and kit generally; this dedicated M8
# runner narrows that to kit below.
# shellcheck source=run60_contract.sh
source "${CONTRACT}"
rambo_validate_launch_contract "$@" || exit $?

artifact_dir=""
operator_name=""
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
            if [[ -z "${candidate}" ]]; then
                echo "--gui-artifact-dir must not be empty" >&2
                exit 2
            fi
            if [[ -n "${artifact_dir}" ]]; then
                echo "Exactly one --gui-artifact-dir is required" >&2
                exit 2
            fi
            artifact_dir="${candidate}"
            ;;
        --gui-artifact-dir=*)
            candidate="${argument#--gui-artifact-dir=}"
            if [[ -z "${candidate}" ]]; then
                echo "--gui-artifact-dir must not be empty" >&2
                exit 2
            fi
            if [[ -n "${artifact_dir}" ]]; then
                echo "Exactly one --gui-artifact-dir is required" >&2
                exit 2
            fi
            artifact_dir="${candidate}"
            ;;
        --operator-name)
            ((index += 1))
            if (( index >= ${#arguments[@]} )); then
                echo "--operator-name requires a value" >&2
                exit 2
            fi
            candidate="${arguments[index]}"
            if [[ -z "${candidate}" ]]; then
                echo "--operator-name must not be empty" >&2
                exit 2
            fi
            if [[ -n "${operator_name}" ]]; then
                echo "Exactly one --operator-name is required" >&2
                exit 2
            fi
            operator_name="${candidate}"
            ;;
        --operator-name=*)
            candidate="${argument#--operator-name=}"
            if [[ -z "${candidate}" ]]; then
                echo "--operator-name must not be empty" >&2
                exit 2
            fi
            if [[ -n "${operator_name}" ]]; then
                echo "Exactly one --operator-name is required" >&2
                exit 2
            fi
            operator_name="${candidate}"
            ;;
        --operator-attestation|--operator-attestation=*)
            echo "--operator-attestation is obsolete; the required declaration is an interactive post-close TTY confirmation" >&2
            exit 2
            ;;
        --viz)
            ((index += 1))
            if (( index >= ${#arguments[@]} )); then
                echo "--viz requires a value" >&2
                exit 2
            fi
            viz_value="${arguments[index]}"
            ;;
        --viz=*)
            viz_value="${argument#--viz=}"
            ;;
    esac
    ((index += 1))
done

if [[ -z "${artifact_dir}" ]]; then
    echo "This runner requires exactly one --gui-artifact-dir to record the real post-close exit status" >&2
    exit 2
fi
if [[ -z "${operator_name}" ]]; then
    echo "This runner requires exactly one --operator-name for the post-close TTY attestation" >&2
    exit 2
fi
if [[ "${viz_value}" != "kit" ]]; then
    echo "M8 GUI keyboard artifacts require explicit --viz kit" >&2
    exit 2
fi
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "Y" ]]; then
    echo "M8 GUI keyboard gate requires the user's process-scoped OMNI_KIT_ACCEPT_EULA=Y prefix" >&2
    exit 2
fi
if [[ -z "${DISPLAY:-}" ]]; then
    echo "M8 GUI keyboard gate requires DISPLAY to point at the visible desktop; do not run this headlessly" >&2
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
if [[ "${child_status}" -ne 0 ]]; then
    echo "M8 GUI child did not close cleanly; no operator attestation was requested." >&2
    exit "${child_status}"
fi

set +e
"${VENV_DIR}/bin/python" "${RECORDER}" \
    --artifact-dir "${artifact_dir}" --operator-name "${operator_name}"
recorder_status=$?
set -e
if [[ "${recorder_status}" -ne 0 ]]; then
    echo "M8 GUI operator attestation was not completed; artifact remains unaccepted." >&2
    exit "${recorder_status}"
fi

set +e
"${VENV_DIR}/bin/python" "${VALIDATOR}" --artifact-dir "${artifact_dir}"
validator_status=$?
set -e
if [[ "${validator_status}" -ne 0 ]]; then
    echo "M8 GUI offline validation failed after post-close attestation." >&2
    exit "${validator_status}"
fi
exit 0
