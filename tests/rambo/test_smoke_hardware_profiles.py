"""Static contracts for migrated, PhysX-only smoke launchers."""

from __future__ import annotations

from pathlib import Path


def test_legacy_smoke_aliases_the_audited_physx_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "smoke.py").read_text(encoding="utf-8")
    assert "official_physx_smoke.py" in contents
    assert "Isaac Sim 5.1" not in contents
    assert "skip_cleanup" not in contents


def test_physx_smoke_launchers_fail_closed_before_starting_kit() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("official_physx_smoke.py", "physx_task_smoke.py"):
        contents = (root / "scripts" / "rambo" / name).read_text(encoding="utf-8")
        assert "use_newton_actuators" in contents
        assert "--viz none" in contents
        assert "--experience" in contents
        assert "--kit_args" in contents
        assert "skip_cleanup" not in contents
        assert "os._exit" not in contents
    assert "PhysxCfg" in (root / "scripts" / "rambo" / "official_physx_smoke.py").read_text(encoding="utf-8")
    official_smoke = (root / "scripts" / "rambo" / "official_physx_smoke.py").read_text(encoding="utf-8")
    assert '"cartpole-direct"' in official_smoke
    assert '"Isaac-Cartpole-Direct-v0"' in official_smoke
    assert "--num-envs" in official_smoke
    assert "num_envs=args_cli.num_envs" in official_smoke
    assert 'env_cfg.seed = 42' in official_smoke
    assert '"checksums.sha256"' in official_smoke
    assert '"backend_before": backend_before' in official_smoke
    assert '"backend_after": backend_after_steps' in official_smoke
    assert "external_exit_code_required" in official_smoke
    assert "assert_physx_environment" in (root / "scripts" / "rambo" / "physx_task_smoke.py").read_text(
        encoding="utf-8"
    )


def test_isaacsim60_compatibility_entry_points_delegate_to_audited_launchers() -> None:
    root = Path(__file__).resolve().parents[2]
    run60 = (root / "scripts" / "rambo" / "run60.sh").read_text(encoding="utf-8")
    smoke60 = (root / "scripts" / "rambo" / "smoke60.py").read_text(encoding="utf-8")
    assert '"${SCRIPT_DIR}/run.sh"' in run60
    assert "physx_task_smoke.py" in smoke60


def test_checkpoint_entry_points_use_the_shared_physx_launcher_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("play.py", "validate.py"):
        contents = (root / "scripts" / "rambo" / name).read_text(encoding="utf-8")
        assert "validate_rambo_visualizer_args" in contents
        assert "assert_physx_environment" in contents
        assert "configure_physx" in contents
        assert "skip_cleanup" not in contents
        assert "os._exit" not in contents

    task_smoke = (root / "scripts" / "rambo" / "physx_task_smoke.py").read_text(encoding="utf-8")
    assert "--viz none" in task_smoke
    assert "assert_physx_environment" in task_smoke
    assert "configure_physx" in task_smoke


def test_rgbd_smoke_disables_nested_debug_callbacks_before_kit_teardown() -> None:
    """Camera-only runs must not retain controller marker callbacks after close."""

    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "physx_rgbd_smoke.py").read_text(encoding="utf-8")
    assert "joint_position_controller_debug_vis" in contents
    assert "contact_generator_debug_vis" in contents
    assert "qp_debug_vis" in contents
    assert "use_newton_actuators = False" in contents
