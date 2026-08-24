#!/usr/bin/env python3
"""Validate a recorded RAMBO Button PhysX episode without launching Kit.

This tool is deliberately offline: it reads the immutable evidence directory,
recomputes trace cadence, terminal/success consistency, checksums, and M10's
100-step memory gate.  It never creates a simulator application and therefore
cannot select any physics backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rambo.validation.button_episode_artifact import (
    ButtonEpisodeArtifactError,
    validate_button_episode_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Existing M10 artifact directory")
    parser.add_argument(
        "--require-acceptance",
        action="store_true",
        help="Reject explicitly labelled short-smoke artifacts; require the 3000-step / 375-RGB evidence",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = validate_button_episode_artifact(
            args.artifact_dir,
            require_acceptance=bool(args.require_acceptance),
        )
    except ButtonEpisodeArtifactError as error:
        print(f"RAMBO_BUTTON_PHYSX_EPISODE_INVALID: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("RAMBO_BUTTON_PHYSX_EPISODE_VALID", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
