"""Simulator-free contracts for the finite released-biped PhysX smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_smoke_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "rambo" / "physx_biped_policy_smoke.py"
    specification = importlib.util.spec_from_file_location("rambo_physx_biped_policy_smoke", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_biped_policy_smoke_is_importable_without_starting_kit() -> None:
    module = _load_smoke_module()

    assert module.TASK == "Isaac-RAMBO-Biped-Go2-v0"
    assert module.DEFAULT_STEPS == 16
    assert module.BIPED_VALIDATION_PHASE_OFFSET_S == 19.6


def test_biped_policy_smoke_accepts_batched_released_policy_dimensions() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()

    class BatchedEnv:
        num_actions = 18

        def get_observations(self):
            return torch.zeros((16, 435), dtype=torch.float32), {}

    observations = module._validate_policy_environment(
        BatchedEnv(),
        SimpleNamespace(observation_dim=435, action_dim=18),
        torch,
        16,
    )
    assert tuple(observations.shape) == (16, 435)


def test_biped_policy_smoke_passes_the_fixed_phase_to_validation_configuration() -> None:
    module = _load_smoke_module()
    calls = []
    env_cfg = SimpleNamespace(enable_rgb_camera=True, seed=None)

    def configure_validation_cfg(config, **kwargs):
        calls.append(kwargs)
        config.episode_length_s = kwargs["duration_s"]
        config.contact_phase_offset_s = kwargs["contact_phase_offset_s"]
        config.events = None
        config.randomize_initial_state = False
        config.randomize_episode_progress = False
        config.obs_noise = False
        config.contact_generator_config = {
            "contact_sequence": {
                "FL": [["swing", kwargs["duration_s"] + kwargs["contact_phase_offset_s"]]],
            }
        }

    evidence = module._configure_biped_smoke(
        env_cfg,
        configure_validation_cfg=configure_validation_cfg,
        seed=42,
        steps=3000,
    )

    assert calls == [{
        "duration_s": 31.0,
        "enable_rgb_camera": False,
        "contact_phase_offset_s": 19.6,
    }]
    assert evidence["phase_offset_s"] == 19.6
    assert evidence["required_contact_duration_s"] == pytest.approx(50.6)
    assert env_cfg.enable_rgb_camera is False


def test_biped_policy_smoke_tensor_metrics_fail_closed_on_nonfinite_data() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()

    metrics = module._tensor_metrics(torch.tensor([[1.0, -2.0]], dtype=torch.float32), "actions")
    assert metrics["shape"] == [1, 2]
    assert metrics["abs_max"] == 2.0
    with pytest.raises(module.PolicySmokeError, match="NaN or Inf"):
        module._tensor_metrics(torch.tensor([float("nan")]), "actions")


def test_biped_policy_smoke_memory_analysis_records_long_run_threshold_evidence() -> None:
    module = _load_smoke_module()
    samples = [
        {"step": step, "gpu_allocated_bytes": 2 * module.MIB, "rss_bytes": 4 * module.MIB}
        for step in range(0, 3001, 100)
    ]

    evidence = module._memory_analysis(samples, 3000)

    assert evidence["evaluated"] is True
    assert evidence["sample_interval_steps"] == 100
    assert evidence["gpu"]["slope_mib_per_100_steps"] == 0.0
    assert evidence["rss"]["median_delta_mib"] == 0.0
    assert evidence["first_window_steps"] == [0, 100, 200, 300, 400, 500]
    assert evidence["last_window_steps"] == [2600, 2700, 2800, 2900, 3000]


def test_biped_policy_smoke_memory_analysis_fails_closed_when_growth_exceeds_limit() -> None:
    module = _load_smoke_module()
    samples = [
        {"step": step, "gpu_allocated_bytes": step * 2 * module.MIB, "rss_bytes": 0}
        for step in range(0, 3001, 100)
    ]

    with pytest.raises(module.PolicySmokeError, match="gpu allocation slope"):
        module._memory_analysis(samples, 3000)


def test_biped_policy_smoke_uses_the_shared_fail_closed_physx_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "physx_biped_policy_smoke.py").read_text(encoding="utf-8")

    assert "validate_rambo_visualizer_args" in contents
    assert "configure_physx" in contents
    assert "assert_physx_environment" in contents
    assert "use_newton_actuators=False" in contents
    assert '"--num-envs"' in contents
    assert "isaaclab_newton" not in contents
    assert "import newton" not in contents
    assert "os._exit" not in contents


def test_biped_source_keeps_independent_front_leg_commands_and_qp_overrides() -> None:
    """Guard the dual-FL/FR manipulation contract without importing Kit."""

    root = Path(__file__).resolve().parents[2]
    env_source = (root / "source" / "rambo" / "rambo" / "tasks" / "direct" / "rambo_biped" / "qp_env.py").read_text(
        encoding="utf-8"
    )
    qp_source = (
        root
        / "source"
        / "rambo"
        / "rambo"
        / "tasks"
        / "direct"
        / "rambo_biped"
        / "modules"
        / "qp_torque_optimizer.py"
    ).read_text(encoding="utf-8")

    for command_name in (
        "_ee_pos_fl_commands",
        "_ee_pos_fr_commands",
        "_ee_force_fl_commands",
        "_ee_force_fr_commands",
    ):
        assert command_name in env_source
    assert "torch.cat((external_force_b_fl, external_force_b_fr), dim=1)" in env_source
    assert "torch.cat((body_id_fl, body_id_fr), dim=0).to(dtype=torch.int32)" in env_source
    for index in (0, 1, 4, 5, 8, 9):
        assert f"contact_state_expanded[:, {index}] = 1.0" in env_source
    assert "desired_ee_force_fl_com" in qp_source
    assert "desired_ee_force_fr_com" in qp_source
