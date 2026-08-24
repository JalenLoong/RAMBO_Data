#!/usr/bin/env python3
"""Verify the pinned RAMBO Isaac Sim 6 runtime without launching Kit.

This verifier deliberately only reads Python/package/GPU metadata.  It never
creates a simulation context and therefore never selects a physics backend.
It is used by the installer and can be run independently before a RAMBO launch.
Bare Newton distributions are expected in this exact Isaac Lab dependency graph;
the verifier rejects only optional Newton *extras* requested by RAMBO's pinned
direct input or canonical setup script, not those transitive distributions.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse


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
ISAACLAB_EDITABLE_EXTENSIONS = (
    "isaaclab",
    "isaaclab_ppisp",
    "isaaclab_contrib",
    "isaaclab_assets",
    "isaaclab_newton",
    "isaaclab_ovphysx",
    "isaaclab_physx",
    "isaaclab_tasks",
    "isaaclab_visualizers",
)
_LOCK_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)$")
_DIRECT_INPUT_PIN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_.,-]+\])?)==([^\s;]+)$"
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")
_OPTIONAL_NEWTON_EXTRA_PATTERNS = (
    re.compile(r"isaaclab_newton\s*\[", re.IGNORECASE),
    re.compile(r"isaaclab_physx\s*\[[^\]]*\bnewton\b", re.IGNORECASE),
    re.compile(r"isaaclab_physx\s*\[[^\]]*\ball\b", re.IGNORECASE),
)


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


def _read_required_text(path: Path, *, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise VerificationError(f"Cannot read {label}: {path}") from error


def _validate_no_optional_newton_extras(*sources: tuple[str, Path]) -> None:
    findings: list[str] = []
    for label, path in sources:
        text = _read_required_text(path, label=label)
        for pattern in _OPTIONAL_NEWTON_EXTRA_PATTERNS:
            if pattern.search(text):
                findings.append(f"{label}:{pattern.pattern}")
    if findings:
        raise VerificationError(
            "Pinned installation input requests optional Isaac Lab Newton extras: "
            + ", ".join(findings)
        )


def _normalize_distribution_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name).lower()


def _load_lock_pins(requirements_lock: Path) -> dict[str, tuple[str, str]]:
    """Read the non-editable frozen wheel pins from the canonical lock."""

    text = _read_required_text(requirements_lock, label="pinned requirements lock")
    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-"):
            raise VerificationError(
                f"The frozen lock must not contain editable or nested input at line {line_number}: {line}"
            )
        match = _LOCK_PIN.fullmatch(line)
        if match is None:
            raise VerificationError(f"Unrecognized frozen lock entry at line {line_number}: {line}")
        package, version = match.groups()
        normalized = _normalize_distribution_name(package)
        previous = pins.get(normalized)
        if previous is not None and previous[1] != version:
            raise VerificationError(
                f"Frozen lock has conflicting versions for {package}: {previous[1]} and {version}"
            )
        pins[normalized] = (package, version)
    if not pins:
        raise VerificationError("Pinned requirements lock contains no wheel pins")
    return pins


def _validate_lock_versions(requirements_lock: Path) -> dict[str, Any]:
    pins = _load_lock_pins(requirements_lock)
    mismatches: list[str] = []
    for package, expected in pins.values():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{package}: missing (expected {expected})")
            continue
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual}")
    if mismatches:
        raise VerificationError(
            "Installed non-editable packages differ from the frozen lock: " + "; ".join(mismatches)
        )
    return {"path": str(requirements_lock), "wheel_pins_verified": len(pins)}


def _validate_direct_input_against_lock(
    requirements_input: Path, requirements_lock: Path
) -> dict[str, Any]:
    """Ensure direct intent is represented by the immutable wheel snapshot."""

    lock_pins = _load_lock_pins(requirements_lock)
    direct_pins: dict[str, tuple[str, str]] = {}
    text = _read_required_text(requirements_input, label="pinned requirements input")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _DIRECT_INPUT_PIN.fullmatch(line)
        if match is None:
            raise VerificationError(f"Unrecognized direct requirements input at line {line_number}: {line}")
        raw_name, expected = match.groups()
        package = raw_name.split("[", maxsplit=1)[0]
        normalized = _normalize_distribution_name(package)
        previous = direct_pins.get(normalized)
        if previous is not None and previous[1] != expected:
            raise VerificationError(
                f"Pinned input has conflicting direct versions for {package}: {previous[1]} and {expected}"
            )
        direct_pins[normalized] = (package, expected)
    if not direct_pins:
        raise VerificationError("Pinned requirements input contains no direct package pins")
    mismatches: list[str] = []
    for normalized, (package, expected) in direct_pins.items():
        locked = lock_pins.get(normalized)
        if locked is None:
            mismatches.append(f"{package}: absent from frozen lock")
        elif locked[1] != expected:
            mismatches.append(f"{package}: input {expected}, lock {locked[1]}")
    if mismatches:
        raise VerificationError(
            "Pinned direct input and frozen lock disagree: " + "; ".join(mismatches)
        )
    return {"direct_input_pins_verified": len(direct_pins)}


def _file_url_to_path(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise VerificationError(f"Expected local editable direct_url, found {url!r}")
    return Path(unquote(parsed.path)).resolve()


def _validate_isaaclab_editable_provenance(isaaclab_source: Path) -> dict[str, str]:
    """Require all selected Isaac Lab extensions to resolve to the exact checkout."""

    resolved_source = isaaclab_source.resolve()
    resolved_extensions: dict[str, str] = {}
    for package in ISAACLAB_EDITABLE_EXTENSIONS:
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError as error:
            raise VerificationError(f"Missing required Isaac Lab extension: {package}") from error
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            raise VerificationError(f"{package} is not installed as an editable exact-source extension")
        try:
            direct_url = json.loads(direct_url_text)
            installed_path = _file_url_to_path(str(direct_url["url"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise VerificationError(f"Invalid direct_url provenance for {package}") from error
        expected_path = (resolved_source / "source" / package).resolve()
        if direct_url.get("dir_info", {}).get("editable") is not True or installed_path != expected_path:
            raise VerificationError(
                f"{package} provenance is {installed_path}, expected editable {expected_path}"
            )
        resolved_extensions[package] = str(installed_path)
    return resolved_extensions


def verify(
    *,
    isaaclab_source: Path,
    requirements_input: Path,
    requirements_lock: Path,
    installer_script: Path,
    require_cuda: bool = True,
) -> dict[str, Any]:
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
    _validate_no_optional_newton_extras(
        ("pinned requirements input", requirements_input),
        ("canonical setup script", installer_script),
    )
    lock_record = _validate_lock_versions(requirements_lock)
    lock_record.update(_validate_direct_input_against_lock(requirements_input, requirements_lock))
    isaaclab_extensions = _validate_isaaclab_editable_provenance(isaaclab_source)

    findings = _pip_check()
    unexpected_findings = sorted(set(findings) - KNOWN_PIP_CHECK_FINDINGS)
    missing_findings = sorted(KNOWN_PIP_CHECK_FINDINGS - set(findings))
    if unexpected_findings or missing_findings:
        raise VerificationError(
            "pip check findings differ from the reviewed pinned-stack allowlist; "
            f"unexpected={unexpected_findings!r}, missing={missing_findings!r}"
        )
    gpu: dict[str, Any] = {
        "checked": require_cuda,
        "required_before_any_simulator_runtime_claim": True,
    }
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise VerificationError("Pinned CUDA PyTorch cannot see a GPU")
        device_index = torch.cuda.current_device()
        gpu.update(
            {
                "index": int(device_index),
                "name": torch.cuda.get_device_name(device_index),
                "compute_capability": list(torch.cuda.get_device_capability(device_index)),
                "torch_cuda": torch.version.cuda,
            }
        )
    else:
        gpu["reason"] = "metadata-only mode intentionally skips the CUDA device query"
    return {
        "verified": True,
        "verification_mode": "gpu-required" if require_cuda else "metadata-only",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "isaaclab": {"tag": "v3.0.0-beta2.patch1", "commit": ISAACLAB_COMMIT},
        "packages": package_versions,
        "frozen_lock": lock_record,
        "isaaclab_editable_extensions": isaaclab_extensions,
        "gpu": gpu,
        "physics_policy": {
            "required_runtime_backend": "PhysX",
            "bare_transitive_newton_packages_allowed": True,
            "pinned_input_requests_no_optional_isaaclab_newton_extras": True,
            "canonical_setup_requests_no_optional_isaaclab_newton_extras": True,
            "metadata_verifier_does_not_select_or_execute_a_physics_backend": True,
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
    parser.add_argument(
        "--requirements-lock", type=Path, default=Path("requirements/isaacsim60.lock")
    )
    parser.add_argument(
        "--installer-script", type=Path, default=Path("scripts/setup_isaacsim60.sh")
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip CUDA device inspection for a Docker build; this is not a runtime acceptance.",
    )
    args = parser.parse_args()
    try:
        result = verify(
            isaaclab_source=args.isaaclab_source.expanduser().resolve(),
            requirements_input=args.requirements_input.expanduser().resolve(),
            requirements_lock=args.requirements_lock.expanduser().resolve(),
            installer_script=args.installer_script.expanduser().resolve(),
            require_cuda=not args.metadata_only,
        )
    except VerificationError as error:
        print(f"ISAACSIM60_INSTALL_VERIFICATION_FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("ISAACSIM60_INSTALL_VERIFIED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
