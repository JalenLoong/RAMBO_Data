#!/usr/bin/env python3
"""Pure-offline validator for one LingBot-VA -> RAMBO Dataset V1 episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rambo.validation.lingbot_dataset_v1 import DatasetValidationError, validate_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = validate_episode(args.episode_dir, verify_checksums=True, verify_images=True)
    except DatasetValidationError as error:
        print(f"LINGBOT_RAMBO_DATASET_V1_VALIDATION_FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("LINGBOT_RAMBO_DATASET_V1_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
