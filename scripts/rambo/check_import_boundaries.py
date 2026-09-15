#!/usr/bin/env python3
"""Fail if simulator-owned RAMBO code imports WAM packages."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "rambo" / "rambo"
FORBIDDEN = {"wam_data", "wam_policy", "wam_rambo", "wam_contracts", "wan_va", "diffusers", "transformers"}


def main() -> int:
    failures: list[str] = []
    for path in list(SOURCE.rglob("*.py")) + list((ROOT / "scripts" / "rambo").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        found = sorted(roots & FORBIDDEN)
        if found:
            failures.append(f"{path.relative_to(ROOT)}: forbidden WAM imports {found}")
    if failures:
        print("RAMBO import-boundary check failed:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("RAMBO import boundaries: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
