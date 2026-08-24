"""Static-runtime tests for the Isaac Sim 5.1 loco-manip teleop entry point."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def _load_loco_manip_command_methods():
    """Compile only the public command methods without importing Isaac Sim."""

    import torch

    source = (
        Path(__file__).resolve().parents[2]
        / "source/rambo/rambo/tasks/direct/rambo_quadruped/button_env.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    button_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ButtonQPEnv"
    )
    names = {
        "_resolve_loco_manip_env_ids",
        "_validate_loco_manip_command",
        "set_loco_manip_commands",
    }
    methods = [node for node in button_cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ImportFrom(module="collections.abc", names=[ast.alias(name="Sequence")], level=0),
            ast.Import(names=[ast.alias(name="torch")]),
            *methods,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(module), "<button-loco-manip-api>", "exec"), namespace)

    class FakeButtonEnv:
        _resolve_loco_manip_env_ids = namespace["_resolve_loco_manip_env_ids"]
        _validate_loco_manip_command = namespace["_validate_loco_manip_command"]
        set_loco_manip_commands = namespace["set_loco_manip_commands"]

        def __init__(self) -> None:
            self.num_envs = 3
            self.device = torch.device("cpu")
            self._velocity_commands = torch.full((3, 3), -1.0)
            self._ee_pos_commands = torch.full((3, 3), -2.0)
            self._ee_force_commands = torch.full((3, 3), -3.0)

    return FakeButtonEnv, torch


def test_public_loco_manip_command_api_validates_and_updates_only_requested_envs() -> None:
    fake_env_type, torch = _load_loco_manip_command_methods()
    env = fake_env_type()
    env_ids = torch.tensor([2, 0], dtype=torch.int64)
    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    position = base + 10.0
    force = base + 20.0

    env.set_loco_manip_commands(base, position, force, env_ids=env_ids)

    torch.testing.assert_close(env._velocity_commands[env_ids], base)
    torch.testing.assert_close(env._ee_pos_commands[env_ids], position)
    torch.testing.assert_close(env._ee_force_commands[env_ids], force)
    torch.testing.assert_close(env._velocity_commands[1], torch.full((3,), -1.0))

    with pytest.raises(ValueError, match="shape"):
        env.set_loco_manip_commands(base[:1], position[:1], force[:1], env_ids=env_ids)
    with pytest.raises(ValueError, match="must be on"):
        env.set_loco_manip_commands(
            torch.empty((1, 3), device="meta"), position[:1], force[:1], env_ids=[0]
        )
    with pytest.raises(TypeError, match="torch.float32"):
        env.set_loco_manip_commands(base[:1].double(), position[:1], force[:1], env_ids=[0])
    with pytest.raises(TypeError, match="int32 or int64"):
        env.set_loco_manip_commands(base, position, force, env_ids=torch.tensor([0.0, 1.0]))
    with pytest.raises(IndexError, match=r"\[0, 3\)"):
        env.set_loco_manip_commands(base[:1], position[:1], force[:1], env_ids=[3])
    with pytest.raises(ValueError, match="duplicates"):
        env.set_loco_manip_commands(base, position, force, env_ids=[1, 1])
    with pytest.raises(ValueError, match="finite"):
        env.set_loco_manip_commands(
            torch.tensor([[float("nan"), 0.0, 0.0]]), position[:1], force[:1], env_ids=[0]
        )


def test_teleop_uses_public_command_api_and_preserves_space_fallthrough() -> None:
    teleop = _load_teleop_module()

    valid = SimpleNamespace(
        set_loco_manip_commands=lambda *args, **kwargs: None,
        button_displacement=None,
        button_success=None,
        button_released=None,
        manipulator_ready=None,
        clear_button_success=None,
    )
    teleop._validate_button_environment(valid)
    with pytest.raises(RuntimeError, match="set_loco_manip_commands"):
        teleop._validate_button_environment(SimpleNamespace())

    script_source = (
        Path(__file__).resolve().parents[2] / "scripts/rambo/teleop_loco_manip.py"
    ).read_text(encoding="utf-8")
    assert "base_env.set_loco_manip_commands(" in script_source
    assert "writer_env_ids = env_ids.to(dtype=torch.int32)" in script_source
    assert script_source.count("env_ids=writer_env_ids") == 4
    assert "base_env._velocity_commands.copy_(" not in script_source
    assert "base_env._ee_pos_commands.copy_(" not in script_source
    assert '"SPACE"' not in script_source


def test_button_source_uses_physx_checked_proxy_and_index_writer_apis() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "source/rambo/rambo/tasks/direct/rambo_quadruped/button_env.py"
    ).read_text(encoding="utf-8")

    assert "assert_physx_environment(self)" in source
    assert "default_root_pose.torch" in source
    assert "default_root_vel.torch" in source
    assert "root_link_pos_w.torch" in source
    assert "write_root_pose_to_sim_index" in source
    assert "write_root_velocity_to_sim_index" in source
    assert "writer_env_ids = env_ids.to(dtype=torch.int32)" in source
    assert source.count("env_ids=writer_env_ids") == 2
    assert "write_root_state_to_sim(" not in source
    assert "default_root_state" not in source
    assert "button_press_threshold_m = 0.012" in source
    assert "button_hold_steps = 5" in source
    assert "button_release_threshold_m = 0.002" in source
