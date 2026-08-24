#!/usr/bin/env python3
"""Offline acceptance check for a sealed M7/M9 final third-person artifact.

This command imports only RAMBO's pure-Python artifact schema.  It never
creates AppLauncher, Isaac Sim, or a physics backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rambo.validation.third_person_diagnostic import ThirdPersonDiagnosticError, validate_final_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_final_artifact(args.artifact_dir)
    except ThirdPersonDiagnosticError as error:
        print(f"FINAL_THIRD_PERSON_ARTIFACT_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
