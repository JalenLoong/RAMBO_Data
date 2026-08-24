#!/usr/bin/env python3
"""Compatibility entry point for RAMBO's finite, task-level PhysX smoke.

This is intentionally a thin alias.  The implementation lives in
``physx_task_smoke.py`` so every task smoke uses one fail-closed launcher,
explicit ``PhysxCfg``, and active-manager verification.
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("physx_task_smoke.py")), run_name="__main__")
