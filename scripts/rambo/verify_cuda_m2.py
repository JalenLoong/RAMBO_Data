#!/usr/bin/env python3
"""Write the finite, non-simulator CUDA evidence required by M2.1.

This utility deliberately does not import Isaac Sim, Isaac Lab, or RAMBO.  It
only verifies the pinned Torch CUDA runtime and records a checksum-covered
artifact, so it never selects any physics backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any


ARTIFACT_ROOT = Path("/workspace/runs/audit/isaac60/M2")
EXPECTED_CAPABILITY = (8, 9)
REQUIRED_ARCH = "sm_89"
# NVIDIA documents that Ampere-native SASS (including sm_86) is forward
# compatible with Ada.  The exact pinned Torch 2.10.0+cu128 wheel currently
# exposes sm_86 but not a native sm_89 entry, so preserve that distinction in
# the artifact rather than rejecting a working, vendor-compatible wheel.
ADA_COMPATIBLE_ARCHS = (REQUIRED_ARCH, "sm_86", "sm_80")


def _write_artifact(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (output_dir / "checksums.sha256").write_text(f"{digest}  summary.json\n", encoding="utf-8")


def collect_cuda_evidence(torch: Any) -> dict[str, Any]:
    """Fail closed unless the accepted host CUDA/Torch contract is live."""

    if not bool(torch.cuda.is_available()):
        raise RuntimeError("torch.cuda.is_available() is false")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != EXPECTED_CAPABILITY:
        raise RuntimeError(f"CUDA capability is {capability}, expected {EXPECTED_CAPABILITY}")
    architectures = list(torch.cuda.get_arch_list())
    compatible_arch = next((arch for arch in ADA_COMPATIBLE_ARCHS if arch in architectures), None)
    if compatible_arch is None:
        raise RuntimeError(
            "Torch architecture list lacks native Ada or an Ada-forward-compatible "
            f"Ampere target {ADA_COMPATIBLE_ARCHS}: {architectures}"
        )
    values = torch.arange(1024, device="cuda", dtype=torch.float32)
    squared_sum = float(values.square().sum().item())
    if not squared_sum > 0.0:
        raise RuntimeError(f"CUDA tensor arithmetic returned an invalid sum: {squared_sum}")
    torch.cuda.synchronize()
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": True,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "capability": list(capability),
        "arch_list": architectures,
        "native_sm89_compiled": REQUIRED_ARCH in architectures,
        "ada_compatible_compiled_arch": compatible_arch,
        "arange_square_sum": squared_sum,
        "synchronized": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(ARTIFACT_ROOT.resolve())
    except ValueError as error:
        parser.error(f"--output-dir must be below {ARTIFACT_ROOT}: {output_dir}")
        raise AssertionError("argparse.error must not return") from error
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    summary: dict[str, Any] = {
        "artifact_kind": "rambo_m2_cuda_contract",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "passed": False,
        "physics_backend_selected": False,
    }
    exit_code = 0
    try:
        import torch

        summary["cuda"] = collect_cuda_evidence(torch)
    except BaseException as error:
        exit_code = 1
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["traceback"] = traceback.format_exc()
        traceback.print_exception(type(error), error, error.__traceback__)
    finally:
        summary["passed"] = exit_code == 0
        _write_artifact(output_dir, summary)
    print("M2_CUDA_CONTRACT_SUCCESS" if exit_code == 0 else "M2_CUDA_CONTRACT_FAILURE", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
