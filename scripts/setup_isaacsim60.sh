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
readonly VENV_DIR="${RAMBO_VENV:-/workspace/venvs/rambo60}"
readonly ISAACLAB_SOURCE="${RAMBO_ISAACLAB_SOURCE:-/workspace/IsaacLab-3.0.0-beta2.patch1}"
readonly ISAACLAB_COMMIT="ffff603eafc6b74264a5261cc0183d6a65390d78"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_isaacsim60.sh

Environment:
  RAMBO_VENV              Override /workspace/venvs/rambo60.
  RAMBO_ISAACLAB_SOURCE   Override the exact tagged Isaac Lab checkout.

The installer creates/reuses a Python 3.12 venv, installs only the pinned
Isaac Sim/Torch packages, then installs the official Isaac Lab extensions from
the exact v3.0.0-beta2.patch1 commit.  It never accepts the EULA persistently;
pass OMNI_KIT_ACCEPT_EULA=Y only when subsequently launching Isaac Sim.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create the pinned Isaac Sim 6 environment." >&2
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
uv pip install --python "${PYTHON}" \
    "isaacsim[all,extscache]==6.0.1.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
uv pip uninstall --python "${PYTHON}" torch torchvision torchaudio
uv pip install --python "${PYTHON}" \
    "torch==2.10.0" "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu128

# The exact tag requires these source extensions.  ``isaaclab_newton`` and
# ``isaaclab_ovphysx`` are bare transitive graph nodes only: no [all], [newton]
# or other optional Newton extras are requested.
for extension in \
    isaaclab \
    isaaclab_ppisp \
    isaaclab_contrib \
    isaaclab_assets \
    isaaclab_newton \
    isaaclab_ovphysx \
    isaaclab_physx \
    isaaclab_tasks; do
    uv pip install --python "${PYTHON}" --no-deps --editable "${ISAACLAB_SOURCE}/source/${extension}"
done
uv pip install --python "${PYTHON}" --no-deps --editable "${ISAACLAB_SOURCE}/source/isaaclab_visualizers[kit]"

# ``isaaclab_tasks`` imports Hydra at runtime without declaring it in the
# minimal source metadata.  qpth is installed without its unconstrained
# resolver dependencies, then supplied with the tested non-Newton runtime.
uv pip install --python "${PYTHON}" "hydra-core==1.3.2" "omegaconf==2.3.1"
uv pip install --python "${PYTHON}" --no-deps "qpth==0.0.18"
uv pip install --python "${PYTHON}" "cvxpy==1.6.7" "clarabel==0.11.1" "scs==3.2.11"

uv pip install --python "${PYTHON}" --no-deps --editable "${REPO_ROOT}/source/crl2"
uv pip install --python "${PYTHON}" --no-deps --editable "${REPO_ROOT}/source/rambo"

# This read-only verifier does not launch Kit or any physics backend.  It
# records the exact expected metadata conflicts from the vendor wheel graph
# and rejects unexpected resolver drift rather than treating ``pip check`` as
# a blanket success/failure signal.
"${PYTHON}" "${REPO_ROOT}/scripts/verify_isaacsim60_install.py" \
    --isaaclab-source "${ISAACLAB_SOURCE}" \
    --requirements-input "${REPO_ROOT}/requirements/isaacsim60.in"

echo "Setup complete: ${VENV_DIR}"
echo "Run RAMBO through scripts/rambo/run60.sh and explicitly pass --viz none or --viz kit."
