"""Small runtime boundary for optional Omniverse/Isaac Lab imports.

Importing the installed :mod:`isaaclab` package outside an Omniverse Kit
process bootstraps Isaac Sim (and may prompt for the EULA).  RAMBO's package
metadata, validation helpers, and pure-Python tests must stay usable in a
regular interpreter, so simulator-dependent modules use this explicit guard
before importing Isaac Lab.
"""

from __future__ import annotations

import sys


def kit_runtime_ready() -> bool:
    """Return whether an Omniverse Kit application is already running.

    ``AppLauncher`` loads ``omni.kit.app`` before user task modules are
    imported.  Looking only in :data:`sys.modules` is intentionally passive:
    it never imports ``omni`` or ``isaaclab`` as a side effect.
    """

    return sys.modules.get("omni.kit.app") is not None


__all__ = ["kit_runtime_ready"]
