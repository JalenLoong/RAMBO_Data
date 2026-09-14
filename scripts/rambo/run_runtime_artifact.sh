#!/usr/bin/env bash
# Run one audited RAMBO runtime command and seal its artifact after Kit exits.
#
# The child receives any process-scoped OMNI_KIT_ACCEPT_EULA=Y prefix.  The
# pure-Python finalizer deliberately receives no EULA variable and cannot
# launch Kit or select a physics backend.

set -uo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"
readonly LOCAL_WORKSPACE_ROOT="${WORKSPACE_ROOT:-${DEFAULT_WORKSPACE_ROOT}}"
readonly VENV_DIR="${RAMBO_VENV:-${LOCAL_WORKSPACE_ROOT}/envs/rambo-isaac60-py312}"
readonly FINALIZER="${SCRIPT_DIR}/finalize_runtime_artifact.py"
readonly CONTRACT="${SCRIPT_DIR}/run60_contract.sh"

usage() {
    echo "Usage: $0 <RAMBO Python script and arguments including exactly one --output-dir>" >&2
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 2
fi
if [[ ! -r "${CONTRACT}" ]]; then
    echo "Missing RAMBO launch contract: ${CONTRACT}" >&2
    exit 1
fi

# Validate before any venv/Python work or artifact finalization.  This keeps a
# rejected launcher override from ever reaching Kit and avoids sealing a
# misleading partial artifact for an invocation that never started.
# shellcheck source=run60_contract.sh
source "${CONTRACT}"
rambo_validate_launch_contract "$@" || exit $?
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "RAMBO Python 3.12 venv was not found: ${VENV_DIR}/bin/python" >&2
    exit 1
fi
if [[ ! -f "${FINALIZER}" ]]; then
    echo "Missing runtime artifact finalizer: ${FINALIZER}" >&2
    exit 1
fi

artifact_dir=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
    argument="${arguments[index]}"
    case "${argument}" in
        --output-dir)
            ((index += 1))
            if [[ "${index}" -ge "${#arguments[@]}" ]]; then
                echo "--output-dir requires a value" >&2
                exit 2
            fi
            candidate="${arguments[index]}"
            ;;
        --output-dir=*)
            candidate="${argument#--output-dir=}"
            ;;
        *)
            continue
            ;;
    esac
    if [[ -z "${candidate}" ]]; then
        echo "--output-dir must not be empty" >&2
        exit 2
    fi
    if [[ -n "${artifact_dir}" ]]; then
        echo "Exactly one --output-dir is required" >&2
        exit 2
    fi
    artifact_dir="${candidate}"
done
if [[ -z "${artifact_dir}" ]]; then
    echo "This runner requires exactly one --output-dir so it can record the real process exit status" >&2
    exit 2
fi

set +e
"${SCRIPT_DIR}/run60.sh" "$@"
child_status=$?
set -e

# Do not carry process-scoped EULA consent into the non-Kit finalizer.
unset OMNI_KIT_ACCEPT_EULA || true
set +e
"${VENV_DIR}/bin/python" "${FINALIZER}" \
    --output-dir "${artifact_dir}" --process-exit-status "${child_status}"
finalizer_status=$?
set -e
if [[ "${finalizer_status}" -ne 0 ]]; then
    echo "Runtime artifact finalization failed after child exit status ${child_status}" >&2
    exit "${finalizer_status}"
fi
exit "${child_status}"
