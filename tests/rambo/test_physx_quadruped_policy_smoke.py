"""Simulator-free contracts for the finite released-policy PhysX smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_smoke_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "rambo" / "physx_quadruped_policy_smoke.py"
    specification = importlib.util.spec_from_file_location("rambo_physx_policy_smoke", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_policy_smoke_is_importable_without_starting_kit() -> None:
    module = _load_smoke_module()

    assert module.TASK == "Isaac-RAMBO-Quadruped-Go2-v0"
    assert module.DEFAULT_STEPS == 16


def test_policy_smoke_tensor_metrics_fail_closed_on_nonfinite_data() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()

    metrics = module._tensor_metrics(torch.tensor([[1.0, -2.0]], dtype=torch.float32), "actions")
    assert metrics == {
        "shape": [1, 2],
        "min": -2.0,
        "max": 1.0,
        "mean": -0.5,
        "abs_max": 2.0,
    }
    with pytest.raises(module.PolicySmokeError, match="NaN or Inf"):
        module._tensor_metrics(torch.tensor([float("nan")]), "actions")


def test_policy_smoke_accepts_batched_released_policy_dimensions() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()

    class BatchedEnv:
        num_actions = 18

        def get_observations(self):
            return torch.zeros((16, 405), dtype=torch.float32), {}

    observations = module._validate_policy_environment(
        BatchedEnv(),
        SimpleNamespace(observation_dim=405, action_dim=18),
        torch,
        16,
    )
    assert tuple(observations.shape) == (16, 405)


def test_policy_smoke_reads_required_state_from_public_articulation_data() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()

    class Proxy:
        def __init__(self, value):
            self.torch = value

    data = SimpleNamespace(
        root_link_pos_w=Proxy(torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32)),
        projected_gravity_b=Proxy(torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)),
        joint_vel=Proxy(torch.zeros((1, 12), dtype=torch.float32)),
        joint_vel_limits=Proxy(torch.full((1, 12), 30.0, dtype=torch.float32)),
        joint_names=[f"joint_{index}" for index in range(12)],
    )
    base_env = SimpleNamespace(scene=SimpleNamespace(articulations={"robot": SimpleNamespace(data=data)}))
    record, extrema = module._collect_public_robot_state(
        base_env,
        num_envs=1,
        step=1,
        contract=SimpleNamespace(
            gravity_target=(0.0, 0.0, -1.0), min_base_height=0.1, max_orientation_error=0.75
        ),
        torch=torch,
    )

    assert record["base_height_m"] == [pytest.approx(0.3)]
    assert record["orientation_error"] == [0.0]
    assert record["joint_velocity_rad_s"] == [[0.0] * 12]
    assert extrema["max_joint_velocity_ratio"] == 0.0


def test_policy_smoke_memory_analysis_enforces_both_approved_thresholds() -> None:
    module = _load_smoke_module()
    stable_samples = [
        {"step": step, "gpu_allocated_bytes": 32 * module.MIB, "rss_bytes": 1024 * module.MIB}
        for step in range(0, 3001, 100)
    ]

    stable = module._memory_analysis(stable_samples, 3000)
    assert stable["evaluated"] is True
    assert stable["gpu"]["slope_mib_per_100_steps"] == 0.0
    assert stable["rss"]["median_delta_mib"] == 0.0

    leaking_gpu_samples = [
        {
            "step": step,
            "gpu_allocated_bytes": (32 + 2 * (step // 100)) * module.MIB,
            "rss_bytes": 1024 * module.MIB,
        }
        for step in range(0, 3001, 100)
    ]
    with pytest.raises(module.PolicySmokeError, match="gpu allocation slope"):
        module._memory_analysis(leaking_gpu_samples, 3000)


def test_long_smoke_extends_a_copied_contact_schedule_to_its_required_span() -> None:
    module = _load_smoke_module()
    original_contact_config = {
        "contact_generator_debug_vis": False,
        "contact_sequence": {
            "FL": [["stance", 1.0, 0.0], ["swing", 9.0, 0.0]],
            "FR": [["stance", 1.0, 0.0], ["phase", 9.0, 0.0]],
        },
    }
    cfg = SimpleNamespace(
        seed=None,
        events=object(),
        randomize_initial_state=True,
        obs_noise=True,
        randomize_episode_progress=True,
        enable_sampled_velocity_commands=True,
        enable_sampled_pos_commands=True,
        enable_sampled_force_commands=True,
        velocity_debug_vis=True,
        pos_debug_vis=True,
        force_debug_vis=True,
        enable_rgb_camera=True,
        sim=SimpleNamespace(dt=0.002),
        decimation=5,
        episode_length_s=10.0,
        contact_generator_config=original_contact_config,
    )

    module._configure_deterministic_smoke(cfg, seed=42, steps=3000)

    assert cfg.episode_length_s == 31.0
    assert cfg.contact_generator_config is not original_contact_config
    assert original_contact_config["contact_sequence"]["FL"][-1][1] == 9.0
    assert cfg.contact_generator_config["contact_sequence"]["FL"][-1][1] == 30.0
    assert sum(segment[1] for segment in cfg.contact_generator_config["contact_sequence"]["FR"]) == 31.0


def test_policy_smoke_uses_the_shared_fail_closed_physx_gate() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "physx_quadruped_policy_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "validate_rambo_visualizer_args" in contents
    assert "configure_physx" in contents
    assert "assert_physx_environment" in contents
    assert "use_newton_actuators=False" in contents
    assert '"--num-envs"' in contents
    assert "os._exit" not in contents
