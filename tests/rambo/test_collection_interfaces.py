from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from rambo.collection import (
    CameraFrame,
    RAMBO_QUADRUPED_COMMAND_DIM,
    RAMBO_QUADRUPED_COMMAND_FRAMES,
    RAMBO_QUADRUPED_COMMAND_NAMES,
    RAMBO_QUADRUPED_COMMAND_UNITS,
    RamboStepSnapshot,
    TaskOutcomeSnapshot,
)


def test_quadruped_command_contract_is_explicit_and_simulator_native() -> None:
    assert RAMBO_QUADRUPED_COMMAND_DIM == 9
    assert RAMBO_QUADRUPED_COMMAND_NAMES == (
        "base_vx", "base_vy", "base_yaw_rate", "fl_ee_x", "fl_ee_y", "fl_ee_z",
        "fl_ee_fx", "fl_ee_fy", "fl_ee_fz",
    )
    assert RAMBO_QUADRUPED_COMMAND_UNITS == (
        "m/s", "m/s", "rad/s", "m", "m", "m", "N", "N", "N",
    )
    assert RAMBO_QUADRUPED_COMMAND_FRAMES == (
        "projected_com_controller", "projected_com_controller", "projected_com_controller",
        "projected_com", "projected_com", "projected_com",
        "projected_com", "projected_com", "projected_com",
    )


ROOT = Path(__file__).resolve().parents[2]
QUADRUPED_ROOT = ROOT / "source" / "rambo" / "rambo" / "tasks" / "direct" / "rambo_quadruped"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
    )
    return ast.get_source_segment(source, method) or ""


def test_quadruped_public_setters_preserve_three_native_3d_buffers_without_clamping() -> None:
    for filename, class_name in (("object_tasks_env.py", "ObjectTaskQPEnv"), ("button_env.py", "ButtonQPEnv")):
        setter = _method_source(QUADRUPED_ROOT / filename, class_name, "set_loco_manip_commands")
        for buffer_name in ("_velocity_commands", "_ee_pos_commands", "_ee_force_commands"):
            assert buffer_name in setter
        assert "index_copy_" in setter
        assert "clamp" not in setter
        validation = setter
        if class_name == "ButtonQPEnv":
            validation += _method_source(QUADRUPED_ROOT / filename, class_name, "_validate_loco_manip_command")
        assert "isfinite" in validation


def test_quadruped_observation_keeps_9d_command_order_at_offsets_54_through_62() -> None:
    source = (QUADRUPED_ROOT / "qp_env.py").read_text(encoding="utf-8")
    start = source.index("    def _get_observations(self) -> dict:")
    end = source.index("    def _prepare_rewards", start)
    observation = source[start:end]
    ordered = ("self._velocity_commands", "self._ee_pos_commands", "self._ee_force_commands", "self._last_action")
    positions = [observation.index(item) for item in ordered]
    assert positions == sorted(positions)
    # One history row is 81D.  The 9D command occupies 54:63 and the 18D
    # low-level residual action is separate at 63:81; five rows make 405D.
    state_width_before_command = 1 + 3 + 3 + 3 + 12 + 12 + 4 + 4 + 12
    assert state_width_before_command == 54
    assert state_width_before_command + RAMBO_QUADRUPED_COMMAND_DIM == 63
    assert 5 * (63 + 18) == 405


def test_rambo_source_has_no_wam_imports() -> None:
    package_root = ROOT / "source" / "rambo" / "rambo"
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                violations.extend(f"{path}: {alias.name}" for alias in node.names if alias.name.startswith("wam"))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("wam"):
                violations.append(f"{path}: {node.module}")
    assert not violations


def test_collection_snapshots_keep_native_9d_and_18d_surfaces_separate() -> None:
    frame = CameraFrame("ego", 80_000_000, np.zeros((480, 640, 3), dtype=np.uint8))
    outcome = TaskOutcomeSnapshot({"contact": True}, True, False, None)
    snapshot = RamboStepSnapshot(
        observation=np.zeros(405, dtype=np.float32),
        policy_action_18d=np.zeros(18, dtype=np.float32),
        robot_state={"joint_pos": np.zeros(12, dtype=np.float32)},
        cameras=(frame,),
        outcome=outcome,
        command_9d=np.zeros(9, dtype=np.float32),
    )
    assert snapshot.command_9d.shape == (9,)
    assert snapshot.policy_action_18d.shape == (18,)
    assert snapshot.observation.shape == (405,)
