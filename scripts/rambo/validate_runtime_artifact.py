#!/usr/bin/env python3
"""Offline acceptance check for a sealed generic RAMBO runtime artifact.

This command is standard-library-only through its sibling finalizer and never
launches Isaac Sim or chooses a physics backend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _finalizer_module():
    path = Path(__file__).with_name("finalize_runtime_artifact.py")
    specification = importlib.util.spec_from_file_location("rambo_runtime_artifact_finalizer", path)
    if specification is None or specification.loader is None:  # pragma: no cover - filesystem guard.
        raise RuntimeError(f"Cannot load runtime artifact finalizer: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    module = _finalizer_module()
    try:
        result = module.verify_runtime_artifact(args.artifact_dir)
    except module.RuntimeArtifactFinalizationError as error:
        print(f"RUNTIME_ARTIFACT_VALIDATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
