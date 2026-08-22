"""Static-runtime tests for the Isaac Sim 5.1 loco-manip teleop entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_teleop_module():
    script = Path(__file__).resolve().parents[2] / "scripts/rambo/teleop_loco_manip.py"
    spec = importlib.util.spec_from_file_location("rambo_teleop_loco_manip", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keyboard_event_name_accepts_native_and_synthetic_isaac_sim_inputs() -> None:
    teleop = _load_teleop_module()

    assert teleop._keyboard_event_name(SimpleNamespace(input=SimpleNamespace(name="R"))) == "R"
    assert teleop._keyboard_event_name(SimpleNamespace(input="R")) == "R"
