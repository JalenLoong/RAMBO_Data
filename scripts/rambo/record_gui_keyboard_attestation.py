#!/usr/bin/env python3
"""Interactively attest a completed M8 physical-keyboard GUI observation.

This standard-library-only post-close step never launches Kit, selects a
physics backend, or creates input events.  It first verifies that the dedicated
parent runner sealed a successful child exit and completed workload evidence,
then requires a named operator at an interactive TTY to type an exact phrase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rambo.validation.gui_keyboard_artifact import (
    GuiKeyboardArtifactError,
    OPERATOR_ATTESTATION_STATEMENT,
    TERMINAL_CONFIRMATION_PHRASE,
    validate_pre_attestation_artifact,
    write_post_close_operator_attestation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--operator-name", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        # Do this before inviting a declaration, and repeat it inside the write
        # helper after the exact phrase to fail closed on a race or mutation.
        preflight = validate_pre_attestation_artifact(args.artifact_dir)
    except GuiKeyboardArtifactError as error:
        print(f"M8_GUI_ATTESTATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    if not sys.stdin.isatty():
        print(
            "M8_GUI_ATTESTATION_FAILURE: an interactive TTY is required; redirected acknowledgement is refused",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print("The sealed Kit child has returned. Do not attest unless you personally observed all of this:", flush=True)
    print(OPERATOR_ATTESTATION_STATEMENT, flush=True)
    try:
        response = input(f"Type exactly {TERMINAL_CONFIRMATION_PHRASE!r} to attest: ")
    except (EOFError, KeyboardInterrupt):
        print("\nM8_GUI_ATTESTATION_FAILURE: no interactive acknowledgement was recorded", file=sys.stderr, flush=True)
        return 2
    if response != TERMINAL_CONFIRMATION_PHRASE:
        print("M8_GUI_ATTESTATION_FAILURE: acknowledgement phrase did not match", file=sys.stderr, flush=True)
        return 2
    try:
        result = write_post_close_operator_attestation(
            args.artifact_dir,
            operator_name=args.operator_name,
            confirmation_phrase=response,
        )
    except GuiKeyboardArtifactError as error:
        print(f"M8_GUI_ATTESTATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(preflight | {"attestation": result}, indent=2, sort_keys=True), flush=True)
    print("M8_GUI_ATTESTATION_RECORDED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
