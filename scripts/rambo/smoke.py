#!/usr/bin/env python3
"""Compatibility entry point for the audited Isaac Sim 6 PhysX smoke gate.

The former script embedded legacy simulator APIs and accepted unrestricted
launcher modes. The replacement delegates every invocation to
``official_physx_smoke.py`` so it can only use explicit ``PhysxCfg``, disable
Newton actuators, reject unaudited visualizers, and record the active manager.

For RAMBO task-level smoke tests use ``physx_task_smoke.py``. For the official
layer-separation gate use this command with the same arguments as
``official_physx_smoke.py`` (``--scenario``, ``--output-dir``, and
``--viz none`` or ``--viz kit``).
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("official_physx_smoke.py")), run_name="__main__")
