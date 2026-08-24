"""Simulator-free tests for RAMBO's fail-closed PhysX contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rambo.utils.physx import assert_physx_runtime, validate_rambo_visualizer_args


def _named_type(module: str, name: str):
    return type(name, (), {"__module__": module})


def _sim(*, cfg_name: str = "PhysxCfg", manager_name: str = "PhysxManager", use_newton_actuators: bool = False):
    cfg_type = _named_type("isaaclab_physx.physics.physx_manager_cfg", cfg_name)
    manager_type = _named_type("isaaclab_physx.physics.physx_manager", manager_name)
    configured_manager = _named_type("isaaclab_physx.physics.physx_manager", "PhysxManager")
    cfg = cfg_type()
    cfg.class_type = configured_manager
    return SimpleNamespace(
        cfg=SimpleNamespace(physics=cfg, use_newton_actuators=use_newton_actuators),
        physics_manager=manager_type,
        device="cuda:0",
        get_physics_dt=lambda: 0.002,
    )


def test_physx_runtime_contract_records_exact_backend_evidence() -> None:
    evidence = assert_physx_runtime(_sim())
    assert evidence == {
        "requested_cfg": "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg",
        "actual_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
        "configured_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
        "use_newton_actuators": False,
        "physics_dt_s": 0.002,
        "device": "cuda:0",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"cfg_name": "OtherCfg"}, "physics config"),
        ({"manager_name": "OtherManager"}, "active physics manager"),
        ({"use_newton_actuators": True}, "use_newton_actuators"),
    ),
)
def test_physx_runtime_contract_fails_closed(kwargs, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        assert_physx_runtime(_sim(**kwargs))


def test_visualizer_contract_accepts_only_explicit_none_or_kit() -> None:
    class Parser:
        def error(self, message: str) -> None:
            raise ValueError(message)

    none_args = SimpleNamespace(
        visualizer=["none"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False
    )
    assert validate_rambo_visualizer_args(Parser(), none_args, ["--visualizer", "none"]) == ["none"]
    assert none_args.fast_shutdown is True
    assert none_args.livestream == 0

    kit_args = SimpleNamespace(visualizer=["kit"], experience=None, kit_args=None, livestream=0, fast_shutdown=False)
    assert validate_rambo_visualizer_args(Parser(), kit_args, ["--viz", "kit"]) == ["kit"]

    rejected_args = SimpleNamespace(
        visualizer=["newton"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False
    )
    with pytest.raises(ValueError, match="only --viz none and --viz kit"):
        validate_rambo_visualizer_args(Parser(), rejected_args, ["--viz", "newton"])
