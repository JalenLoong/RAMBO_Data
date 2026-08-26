#!/usr/bin/env bash
# Run the one permitted M2.3 Kit GUI gate, then require a post-close TTY attestation.
#
# This wrapper deliberately accepts no arbitrary simulator arguments.  It fixes
# the official Direct Cartpole task, one environment, explicit --viz kit, and
# the audited finite dwell protocol.  The generic runtime wrapper seals the
# child's real exit status before this script offers the human acknowledgement.

set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="${RAMBO_VENV:-/workspace/envs/rambo-isaac60-py312}"
readonly ARTIFACT_ROOT="/workspace/runs/audit/isaac60/M2"

usage() {
    echo "Usage: $0 --output-dir <fresh M2 runtime dir> --attestation-dir <fresh M2 sidecar dir> --operator-name <name> [--gui-observation-seconds 45]" >&2
}

fail() {
    echo "M2 Cartpole GUI gate: $1" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || fail "${option} requires a non-empty value"
}

output_dir=""
attestation_dir=""
operator_name=""
observation_seconds="45"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ "$#" -ge 2 ]] || fail "--output-dir requires a value"
            output_dir="$2"
            shift 2
            ;;
        --output-dir=*)
            output_dir="${1#--output-dir=}"
            shift
            ;;
        --attestation-dir)
            [[ "$#" -ge 2 ]] || fail "--attestation-dir requires a value"
            attestation_dir="$2"
            shift 2
            ;;
        --attestation-dir=*)
            attestation_dir="${1#--attestation-dir=}"
            shift
            ;;
        --operator-name)
            [[ "$#" -ge 2 ]] || fail "--operator-name requires a value"
            operator_name="$2"
            shift 2
            ;;
        --operator-name=*)
            operator_name="${1#--operator-name=}"
            shift
            ;;
        --gui-observation-seconds)
            [[ "$#" -ge 2 ]] || fail "--gui-observation-seconds requires a value"
            observation_seconds="$2"
            shift 2
            ;;
        --gui-observation-seconds=*)
            observation_seconds="${1#--gui-observation-seconds=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown or unsafe argument: $1"
            ;;
    esac
done

require_value "--output-dir" "${output_dir}"
require_value "--attestation-dir" "${attestation_dir}"
require_value "--operator-name" "${operator_name}"
[[ "${OMNI_KIT_ACCEPT_EULA:-}" == "Y" ]] || fail "requires the user's process-scoped OMNI_KIT_ACCEPT_EULA=Y prefix"
[[ -n "${DISPLAY:-}" ]] || fail "requires DISPLAY to point at the visible desktop; do not run this headlessly"
[[ -x "${VENV_DIR}/bin/python" ]] || fail "missing pinned Python: ${VENV_DIR}/bin/python"

output_dir="$(realpath -m -- "${output_dir}")"
attestation_dir="$(realpath -m -- "${attestation_dir}")"
case "${output_dir}" in
    "${ARTIFACT_ROOT}"/*) ;;
    *) fail "--output-dir must be below ${ARTIFACT_ROOT}" ;;
esac
case "${attestation_dir}" in
    "${ARTIFACT_ROOT}"/*) ;;
    *) fail "--attestation-dir must be below ${ARTIFACT_ROOT}" ;;
esac
[[ "${output_dir}" != "${attestation_dir}" ]] || fail "runtime and attestation directories must differ"

echo "M2.3 starts a finite visible Kit window. Observe the viewport and manually exercise Play/Stop; do not use UI automation." >&2
set +e
"${SCRIPT_DIR}/run_runtime_artifact.sh" \
    "${SCRIPT_DIR}/official_physx_smoke.py" \
    --scenario cartpole-direct --num-envs 1 --steps 16 \
    --gui-observation-seconds "${observation_seconds}" \
    --output-dir "${output_dir}" --viz kit
runtime_status=$?
set -e
if [[ "${runtime_status}" -ne 0 ]]; then
    echo "M2 Cartpole GUI runtime failed or did not close cleanly; no attestation was requested." >&2
    exit "${runtime_status}"
fi

# The runtime wrapper has already written process_exit.json and its final
# checksums. Never carry EULA consent into either pure-Python post-close stage.
unset OMNI_KIT_ACCEPT_EULA || true
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/record_m2_cartpole_gui_attestation.py" \
    --runtime-artifact-dir "${output_dir}" \
    --attestation-dir "${attestation_dir}" \
    --operator-name "${operator_name}"
"${VENV_DIR}/bin/python" "${SCRIPT_DIR}/validate_m2_cartpole_gui_gate.py" \
    --runtime-artifact-dir "${output_dir}" \
    --attestation-dir "${attestation_dir}"
