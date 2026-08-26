#!/usr/bin/env bash
# Install RAMBO's exact Isaac Sim 6.0.1 / Isaac Lab 3.0 Beta 2 Patch 1 stack.
#
# This script intentionally follows the tagged Isaac Lab source layout rather
# than a floating release branch.  Its Newton-related extension installs are
# the bare packages required by the exact official dependency graph only; it
# never requests a Newton extra and RAMBO's runtime configuration remains
# explicitly PhysX-only.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly VENV_DIR="${RAMBO_VENV:-/workspace/envs/rambo-isaac60-py312}"
readonly ISAACLAB_SOURCE="${RAMBO_ISAACLAB_SOURCE:-/workspace/third_party/IsaacLab/3.0.0-beta2.patch1}"
readonly WAM_POLICY_ROOT="${WAM_POLICY_ROOT:-/workspace/repos/WAM-Policy}"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"
readonly REQUIREMENTS_INPUT="${REPO_ROOT}/requirements/isaacsim60.in"
readonly REQUIREMENTS_LOCK="${REPO_ROOT}/requirements/isaacsim60.lock"
readonly VERIFIER="${REPO_ROOT}/scripts/verify_isaacsim60_install.py"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_isaacsim60.sh [--docker-build-metadata-only]

Environment:
  RAMBO_VENV              Override /workspace/envs/rambo-isaac60-py312.
  RAMBO_ISAACLAB_SOURCE   Override the exact tagged Isaac Lab checkout.
  WAM_POLICY_ROOT         Override the canonical WAM checkout.

Options:
  --docker-build-metadata-only
      Run the final package/provenance verification without querying CUDA.
      This option exists only for a Docker *build*, where GPUs are normally
      unavailable.  It is not a GPU, renderer, or PhysX runtime acceptance.

The installer creates/reuses a Python 3.12 venv, installs only the pinned
Isaac Sim/Torch packages, then installs the official Isaac Lab extensions from
the exact v3.0.0-beta2.patch1 commit.  It never accepts the EULA persistently;
pass OMNI_KIT_ACCEPT_EULA=Y only when subsequently launching Isaac Sim.
EOF
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 64
fi

metadata_only=0
case "${1:-}" in
    "") ;;
    --docker-build-metadata-only) metadata_only=1 ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        echo "Unknown setup option: ${1}" >&2
        usage >&2
        exit 64
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create the pinned Isaac Sim 6 environment." >&2
    exit 1
fi
if [[ ! -f "${WAM_POLICY_ROOT}/pyproject.toml" ]]; then
    echo "Missing WAM checkout: ${WAM_POLICY_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${ISAACLAB_SOURCE}/.git" ]]; then
    echo "Missing Isaac Lab source checkout: ${ISAACLAB_SOURCE}" >&2
    exit 1
fi
if [[ "$(git -C "${ISAACLAB_SOURCE}" rev-parse HEAD)" != "${ISAACLAB_COMMIT}" ]]; then
    echo "Isaac Lab checkout is not the required commit ${ISAACLAB_COMMIT}." >&2
    exit 1
fi
for required_file in "${REQUIREMENTS_INPUT}" "${REQUIREMENTS_LOCK}" "${VERIFIER}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing pinned installation input: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    uv venv --python 3.12 "${VENV_DIR}"
fi
readonly PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12, found {sys.version.split()[0]}")
PY

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Keep Isaac Sim and its extension cache transaction isolated from the pinned
# PyTorch override.  Do not add Isaac Lab Newton extras to either transaction.
uv pip install --no-cache --python "${PYTHON}" \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
uv pip uninstall --python "${PYTHON}" torch torchvision torchaudio
uv pip install --no-cache --python "${PYTHON}" \
    "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# The exact tag's official core installation includes these bare source
# extensions.  ``isaaclab_newton`` is an official core node here, not a RAMBO
# backend selection; no [all], [newton], or other optional Newton extras are
# requested.
for extension in \
    isaaclab \
    isaaclab_ppisp \
    isaaclab_contrib \
    isaaclab_assets \
    isaaclab_newton \
    isaaclab_ovphysx \
    isaaclab_physx \
    isaaclab_tasks; do
    uv pip install --no-cache --python "${PYTHON}" --no-deps --editable "${ISAACLAB_SOURCE}/source/${extension}"
done
uv pip install --no-cache --python "${PYTHON}" --no-deps --editable "${ISAACLAB_SOURCE}/source/isaaclab_visualizers[kit]"

# ``isaaclab_tasks`` imports Hydra at runtime without declaring it in the
# minimal source metadata.  qpth is installed without its unconstrained
# resolver dependencies, then supplied with the tested non-Newton runtime.
uv pip install --no-cache --python "${PYTHON}" \
    "numpy==2.3.1" "hydra-core==1.3.2" "omegaconf==2.3.1"
uv pip install --no-cache --python "${PYTHON}" --no-deps "qpth==0.0.18"
uv pip install --no-cache --python "${PYTHON}" "cvxpy==1.6.7" "clarabel==0.11.1" "scs==3.2.11"

uv pip install --no-cache --python "${PYTHON}" --no-deps --editable "${REPO_ROOT}/source/crl2"
uv pip install --no-cache --python "${PYTHON}" --no-deps --editable "${REPO_ROOT}/source/rambo"
uv pip install --no-cache --python "${PYTHON}" --no-deps --editable "${WAM_POLICY_ROOT}"

# The first transaction follows the official tagged install order.  The frozen
# non-editable snapshot then pins every resolved wheel to the validated state
# without invoking a resolver or adding optional Isaac Lab extras.  Isaac Lab,
# CRL2, RAMBO, and WAM editables deliberately remain under the explicit exact-source
# commands above and are checked by direct_url provenance in the verifier.
uv pip install --no-cache --python "${PYTHON}" --no-deps --prerelease=allow \
    --requirements "${REQUIREMENTS_LOCK}"

# This read-only verifier does not launch Kit or any physics backend.  It
# records the exact expected metadata conflicts from the vendor wheel graph
# and rejects unexpected resolver drift rather than treating ``pip check`` as
# a blanket success/failure signal.
verifier_args=(
    --isaaclab-source "${ISAACLAB_SOURCE}"
    --requirements-input "${REQUIREMENTS_INPUT}"
    --requirements-lock "${REQUIREMENTS_LOCK}"
    --installer-script "${SCRIPT_DIR}/setup_isaacsim60.sh"
)
if [[ "${metadata_only}" -eq 1 ]]; then
    verifier_args+=(--metadata-only)
fi
"${PYTHON}" "${VERIFIER}" "${verifier_args[@]}"

echo "Setup complete: ${VENV_DIR}"
echo "Run RAMBO through scripts/rambo/run60.sh and explicitly pass --viz none or --viz kit."
