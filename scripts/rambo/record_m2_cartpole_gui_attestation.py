#!/usr/bin/env python3
"""Interactively attest a completed M2.3 Cartpole Kit GUI observation.

Run this only after ``run_runtime_artifact.sh`` has recorded the Kit child's
real exit status.  It refuses redirected stdin and requires the named operator
to type the exact acknowledgement after the GUI has closed; no command-line
``--yes`` or pre-run attestation option exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from m2_cartpole_gui_artifact import (
    M2CartpoleGuiArtifactError,
    OPERATOR_ATTESTATION_STATEMENT,
    TERMINAL_CONFIRMATION_PHRASE,
    write_operator_attestation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-artifact-dir", type=Path, required=True)
    parser.add_argument("--attestation-dir", type=Path, required=True)
    parser.add_argument("--operator-name", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not sys.stdin.isatty():
        print(
            "M2_GUI_ATTESTATION_FAILURE: an interactive TTY is required; redirected acknowledgement is refused",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print("The sealed Kit runtime has returned. Do not attest unless you personally observed all of this:", flush=True)
    print(OPERATOR_ATTESTATION_STATEMENT, flush=True)
    try:
        response = input(f"Type exactly {TERMINAL_CONFIRMATION_PHRASE!r} to attest: ")
    except (EOFError, KeyboardInterrupt):
        print("\nM2_GUI_ATTESTATION_FAILURE: no interactive acknowledgement was recorded", file=sys.stderr, flush=True)
        return 2
    if response != TERMINAL_CONFIRMATION_PHRASE:
        print("M2_GUI_ATTESTATION_FAILURE: acknowledgement phrase did not match", file=sys.stderr, flush=True)
        return 2
    try:
        result = write_operator_attestation(
            runtime_artifact_dir=args.runtime_artifact_dir,
            attestation_dir=args.attestation_dir,
            operator_name=args.operator_name,
        )
    except M2CartpoleGuiArtifactError as error:
        print(f"M2_GUI_ATTESTATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("M2_GUI_ATTESTATION_RECORDED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
