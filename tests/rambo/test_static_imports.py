"""Ensure metadata tooling never boots Isaac Sim merely by importing RAMBO."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_static_rambo_import_does_not_import_isaaclab_or_kit() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    package_paths = [
        str(repository_root / "source" / "rambo"),
        str(repository_root / "source" / "crl2"),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        package_paths + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    code = "\n".join(
        (
            "import sys",
            "import rambo",
            "from rambo.actuators import DelayedDCMotorCfg",
            "from rambo.utils.markers import GREEN_SPHERE_MARKER_CFG",
            "assert rambo.register_tasks() is False",
            "assert DelayedDCMotorCfg().max_delay == 0",
            "assert GREEN_SPHERE_MARKER_CFG is None",
            "assert 'isaaclab' not in sys.modules",
            "assert 'omni.kit.app' not in sys.modules",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
