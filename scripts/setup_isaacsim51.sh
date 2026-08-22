#!/usr/bin/env bash
# Install the reproducible RAMBO Isaac Sim 5.1 / Isaac Lab 2.3.2 runtime.
# Invoke with: bash scripts/setup_isaacsim51.sh

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly VENV_DIR="${RAMBO_VENV:-/workspace/venvs/rambo51}"
readonly REQUIREMENTS_FILE="${REPO_ROOT}/requirements/isaacsim51.in"

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_isaacsim51.sh

Environment:
  RAMBO_VENV  Override the default virtual environment path
              (/workspace/venvs/rambo51).

The script reuses an existing Python 3.11 venv and reapplies pinned dependencies.
It does not modify the legacy Isaac Sim 4.5/Python 3.10 environment.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Missing requirements file: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "python3.11 is required. Install Python 3.11 and python3.11-venv first." >&2
        exit 1
    fi
    echo "Creating Python 3.11 virtual environment at ${VENV_DIR}"
    python3.11 -m venv "${VENV_DIR}"
fi

readonly PYTHON="${VENV_DIR}/bin/python"
readonly PIP=("${PYTHON}" -m pip)

"${PYTHON}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Expected Python 3.11 in the RAMBO venv, found {sys.version.split()[0]}. "
        "Remove this venv and rerun the setup script."
    )
PY

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Upgrading pip and applying pre-Isaac-Lab compatibility pins"
"${PIP[@]}" install --upgrade pip
# setuptools 81 removes pkg_resources; flatdict is installed before Isaac Lab so its
# dependency graph is complete before Isaac Sim extensions are resolved.
"${PIP[@]}" install --upgrade "setuptools<81" "flatdict==4.0.1" "wheel==0.45.1"

echo "Installing pinned CUDA 12.8 PyTorch first"
# Keep this as a separate transaction.  Isaac Sim's extension-cache wheels are
# multi-gigabyte downloads; installing Torch first makes a retry resumable and
# prevents pip from leaving a partially-installed CUDA runtime after resolver
# interruption.
"${PIP[@]}" install --upgrade --retries 10 --timeout 120 \
    "numpy<2" "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

echo "Installing official Isaac Lab 2.3.2 with Isaac Sim 5.1"
"${PIP[@]}" install --upgrade --retries 10 --timeout 120 \
    "isaaclab[isaacsim,all]==2.3.2" \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128

# The official Isaac Lab pip wheel ships the asset extension source under its
# own site-packages directory, but does not install ``isaaclab_assets`` as a
# Python distribution.  RAMBO intentionally uses that official Go2 asset
# package rather than this repository's vendored ``source/isaaclab_assets``.
ISAACLAB_PACKAGE_DIR="$("${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("isaaclab")
if spec is None or spec.origin is None:
    raise SystemExit("The official isaaclab package was not installed")
print(Path(spec.origin).resolve().parent)
PY
)"
ISAACLAB_ASSETS_SOURCE="${ISAACLAB_PACKAGE_DIR}/source/isaaclab_assets"
if [[ ! -f "${ISAACLAB_ASSETS_SOURCE}/setup.py" ]]; then
    echo "Official Isaac Lab asset extension was not bundled at ${ISAACLAB_ASSETS_SOURCE}" >&2
    exit 1
fi
echo "Installing official bundled Isaac Lab asset extension"
"${PIP[@]}" install --upgrade --no-deps --no-build-isolation "${ISAACLAB_ASSETS_SOURCE}"

echo "Installing the pinned QP solver"
# Isaac Sim 5.1 requires packaging==23.0 and osqp==0.6.7.post3.  qpth's
# unbounded CVXPY dependency would otherwise upgrade both and leave a broken
# simulator installation, so keep its compatible solver stack explicit.
"${PIP[@]}" install --upgrade --retries 10 --timeout 120 \
    "packaging==23.0" "wheel==0.45.1" "osqp==0.6.7.post3" \
    "cvxpy==1.5.4" "ecos==2.0.14" "qpth==0.0.18"

if [[ ! -d "${REPO_ROOT}/source/crl2" ]]; then
    echo "Missing local CRL2 package: ${REPO_ROOT}/source/crl2" >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/source/rambo" ]]; then
    echo "Missing RAMBO external extension: ${REPO_ROOT}/source/rambo" >&2
    echo "Check out the adaptation implementation before running this installer." >&2
    exit 1
fi

echo "Installing local RAMBO packages in editable mode"
"${PIP[@]}" install --editable "${REPO_ROOT}/source/crl2"
"${PIP[@]}" install --editable "${REPO_ROOT}/source/rambo"

"${PIP[@]}" check

"${PYTHON}" - <<'PY'
import importlib.util
import sys
import torch

asset_spec = importlib.util.find_spec("isaaclab_assets")
if asset_spec is None:
    raise SystemExit("Official bundled isaaclab_assets package was not installed")

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"Compiled architectures: {torch.cuda.get_arch_list()}")
print(f"Isaac Lab assets: {asset_spec.origin}")
PY

echo
echo "Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
echo "Use scripts/rambo/run.sh for standalone CUDA qpth tests from a fresh shell."
echo "First-run check: isaacsim"
