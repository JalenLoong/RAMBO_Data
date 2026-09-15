"""Static registry contracts for the standalone RAMBO external extension."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

gym = pytest.importorskip("gymnasium")

from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rambo.validation.checkpoints import contract_for_task  # noqa: E402


def test_registry_helper_resolves_rambo_yaml_for_each_task_without_isaaclab_tasks() -> None:
    """The quadruped locomotion and loco-manip tasks use RAMBO-owned entry points."""

    import rambo.tasks  # noqa: F401 - registration is the behavior under test.

    expected = {
        "Isaac-RAMBO-Quadruped-Go2-v0": ("rambo_quadruped.qp_env:QPEnv", 405, 18),
        "Isaac-RAMBO-Quadruped-Button-Go2-v0": (
            "rambo_quadruped.button_env:ButtonQPEnv",
            405,
            18,
        ),
    }
    for task, (entry_point, observation_dim, action_dim) in expected.items():
        spec = gym.spec(task)
        assert spec.entry_point.endswith(entry_point)
        assert spec.kwargs["env_cfg_entry_point"].startswith("rambo.tasks.direct.")
        cfg = load_cfg_from_registry(task, "crl2_cfg_entry_point")
        assert isinstance(cfg, dict)
        assert cfg["general"]["num_envs"] > 0
        assert cfg["network"]["policy_hidden"] == [512, 256, 128]
        contract = contract_for_task(task)
        # Dimensions remain task-owned checkpoint contracts rather than a side
        # effect of importing a vendored Isaac Lab task package.
        assert (contract.observation_dim, contract.action_dim) == (observation_dim, action_dim)


def test_registry_helper_applies_environment_overrides_without_isaaclab_tasks() -> None:
    """The only runner-side config behavior matches the official helper."""

    task = "RamboRegistryHelper-Test-v0"
    gym.registry.pop(task, None)

    class FakeEnvCfg:
        def __init__(self) -> None:
            self.sim = SimpleNamespace(device="cpu", use_fabric=True)
            self.scene = SimpleNamespace(num_envs=4096)

    gym.register(task, entry_point="builtins:object", kwargs={"env_cfg_entry_point": FakeEnvCfg})
    try:
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, use_fabric=False)
    finally:
        gym.registry.pop(task, None)

    assert cfg.sim.device == "cuda:0"
    assert cfg.sim.use_fabric is False
    assert cfg.scene.num_envs == 1


def test_runtime_scripts_and_tasks_do_not_depend_on_vendored_isaaclab_tasks() -> None:
    """External-extension runners retain only the official asset dependency."""

    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "scripts/rambo/play.py",
        "scripts/rambo/validate.py",
        "scripts/rambo/teleop_loco_manip.py",
    ):
        assert "isaaclab_tasks" not in (repository_root / relative_path).read_text(encoding="utf-8")
    for relative_path in (
        "source/rambo/rambo/tasks/direct/rambo_quadruped/qp_env.py",
    ):
        contents = (repository_root / relative_path).read_text(encoding="utf-8")
        assert "from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG" in contents
        assert "from isaaclab_assets import UNITREE_GO2_CFG" not in contents
    button_contents = (
        repository_root / "source/rambo/rambo/tasks/direct/rambo_quadruped/button_env.py"
    ).read_text(encoding="utf-8")
    assert "isaaclab_tasks" not in button_contents
