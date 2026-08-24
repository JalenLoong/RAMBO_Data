#!/usr/bin/env python3
"""Validate a sealed M2.3 Cartpole Kit runtime plus its human-attestation sidecar.

This command is standard-library-only and offline. It validates checksums,
post-close exit status, explicit PhysX evidence, Kit-live RTX/GPU-memory
samples, and the operator declaration without launching Isaac Sim or choosing
a physics backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from m2_cartpole_gui_artifact import M2CartpoleGuiArtifactError, validate_m2_cartpole_gui_gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-artifact-dir", type=Path, required=True)
    parser.add_argument("--attestation-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = validate_m2_cartpole_gui_gate(
            runtime_artifact_dir=args.runtime_artifact_dir,
            attestation_dir=args.attestation_dir,
        )
    except M2CartpoleGuiArtifactError as error:
        print(f"M2_CARTPOLE_GUI_GATE_INVALID: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("M2_CARTPOLE_GUI_GATE_VALID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
