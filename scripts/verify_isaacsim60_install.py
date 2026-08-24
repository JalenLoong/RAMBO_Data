#!/usr/bin/env python3
"""Verify the pinned RAMBO Isaac Sim 6 runtime without launching Kit.

This verifier deliberately only reads Python/package/GPU metadata.  It never
creates a simulation context and therefore never selects a physics backend.
It is used by the installer and can be run independently before a RAMBO launch.
Bare Newton distributions are expected in this exact Isaac Lab dependency graph;
the verifier rejects only optional Newton *extras* requested by RAMBO's pinned
input file, not those transitive distributions.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
EXPECTED_PACKAGES = {
    "isaacsim": "6.0.1.0",
    "torch": "2.10.0+cu128",
    "torchvision": "0.25.0+cu128",
    "numpy": "2.3.1",
    "qpth": "0.0.18",
    "cvxpy": "1.6.7",
    "clarabel": "0.11.1",
    "scs": "3.2.11",
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.1",
}
# The exact pinned official wheel graph conflicts with RAMBO's deliberately
# pinned Torch/Numpy and minimal editable packages.  These lines are recorded
# rather than ignored wholesale: any additional ``pip check`` result fails.
KNOWN_PIP_CHECK_FINDINGS = {
    "isaacsim-core 6.0.1.0 requires torchaudio, which is not installed.",
    "crl2 0.0.1 requires gitpython, which is not installed.",
    "crl2 0.0.1 requires moviepy, which is not installed.",
    "isaacsim-core 6.0.1.0 has requirement torch==2.11.0, but you have torch 2.10.0+cu128.",
    "isaacsim-core 6.0.1.0 has requirement torchvision==0.26.0, but you have torchvision 0.25.0+cu128.",
    "isaacsim-kernel 6.0.1.0 has requirement coverage==7.4.4, but you have coverage 7.6.1.",
    "rambo 0.1.0 has requirement gymnasium==1.2.0, but you have gymnasium 1.2.1.",
    "qpth 0.0.18 has requirement numpy<2,>=1, but you have numpy 2.3.1.",
}


class VerificationError(RuntimeError):
    """Raised when the reproducible target environment does not match."""


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Cannot resolve Isaac Lab git revision at {path}") from error


def _pip_check() -> list[str]:
    process = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if process.returncode not in (0, 1):
        raise VerificationError(f"pip check failed to execute: {process.stderr.strip()}")
    return [line.strip() for line in process.stdout.splitlines() if line.strip()]


def _validate_no_optional_newton_extras(requirements_input: Path) -> None:
    try:
        text = requirements_input.read_text(encoding="utf-8").lower()
    except OSError as error:
        raise VerificationError(f"Cannot read pinned requirements input: {requirements_input}") from error
    forbidden = ("isaaclab_newton[", "isaaclab_physx[newton", "isaaclab_physx[all")
    found = [fragment for fragment in forbidden if fragment in text]
    if found:
        raise VerificationError(
            "Pinned input requests optional Isaac Lab Newton extras: " + ", ".join(found)
        )


def verify(*, isaaclab_source: Path, requirements_input: Path) -> dict[str, Any]:
    """Return auditable pinned-runtime facts or fail before any Kit launch."""

    if sys.version_info[:2] != (3, 12):
        raise VerificationError(f"Expected Python 3.12, found {sys.version.split()[0]}")
    package_versions: dict[str, str] = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise VerificationError(f"Missing required package: {package}") from error
        package_versions[package] = actual
        if actual != expected:
            raise VerificationError(f"Unexpected {package}: expected {expected}, found {actual}")
    if _git_head(isaaclab_source) != ISAACLAB_COMMIT:
        raise VerificationError(f"Isaac Lab is not the required commit {ISAACLAB_COMMIT}")
    _validate_no_optional_newton_extras(requirements_input)

    import torch

    if not torch.cuda.is_available():
        raise VerificationError("Pinned CUDA PyTorch cannot see a GPU")
    findings = _pip_check()
    unexpected_findings = sorted(set(findings) - KNOWN_PIP_CHECK_FINDINGS)
    missing_findings = sorted(KNOWN_PIP_CHECK_FINDINGS - set(findings))
    if unexpected_findings or missing_findings:
        raise VerificationError(
            "pip check findings differ from the reviewed pinned-stack allowlist; "
            f"unexpected={unexpected_findings!r}, missing={missing_findings!r}"
        )
    device_index = torch.cuda.current_device()
    return {
        "verified": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "isaaclab": {"tag": "v3.0.0-beta2.patch1", "commit": ISAACLAB_COMMIT},
        "packages": package_versions,
        "gpu": {
            "index": int(device_index),
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "torch_cuda": torch.version.cuda,
        },
        "physics_policy": {
            "required_runtime_backend": "PhysX",
            "bare_transitive_newton_packages_allowed": True,
            "optional_isaaclab_newton_extras_requested": False,
        },
        "pip_check_allowlisted_findings": sorted(findings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaaclab-source", type=Path, default=Path("/workspace/IsaacLab-3.0.0-beta2.patch1")
    )
    parser.add_argument(
        "--requirements-input", type=Path, default=Path("requirements/isaacsim60.in")
    )
    args = parser.parse_args()
    try:
        result = verify(
            isaaclab_source=args.isaaclab_source.expanduser().resolve(),
            requirements_input=args.requirements_input.expanduser().resolve(),
        )
    except VerificationError as error:
        print(f"ISAACSIM60_INSTALL_VERIFICATION_FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("ISAACSIM60_INSTALL_VERIFIED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
