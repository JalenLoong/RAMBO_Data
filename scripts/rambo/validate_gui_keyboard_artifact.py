#!/usr/bin/env python3
"""Validate a RAMBO GUI-keyboard artifact without launching Kit or physics.

The validator reads the immutable evidence directory only.  A successful
result verifies callback/state provenance, explicit PhysX evidence, base/FL
commands, an unhandled SPACE callback, button press/success/rebound evidence,
exact checksums, the operator declaration, and the dedicated parent-observed
zero exit written after Kit closed. The declaration must be the recorded
post-close interactive-TTY acknowledgement. It does not independently prove
that the callback originated from a physical keyboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rambo.validation.gui_keyboard_artifact import (
    GuiKeyboardArtifactError,
    validate_gui_keyboard_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Completed GUI-keyboard artifact directory")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = validate_gui_keyboard_artifact(args.artifact_dir)
    except GuiKeyboardArtifactError as error:
        print(f"RAMBO_GUI_KEYBOARD_ARTIFACT_INVALID: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("RAMBO_GUI_KEYBOARD_ARTIFACT_VALID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
