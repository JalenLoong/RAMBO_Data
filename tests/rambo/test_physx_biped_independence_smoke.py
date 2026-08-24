"""Pure-Python contracts for the focused M9 biped front-leg independence smoke."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rambo.validation import biped_independence as schema


def _load_smoke_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "rambo" / "physx_biped_independence_smoke.py"
    specification = importlib.util.spec_from_file_location("rambo_physx_biped_independence_smoke", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_physx_contract_module():
    root = Path(__file__).resolve().parents[2]
    source_path = root / "source" / "rambo" / "rambo" / "utils" / "physx.py"
    specification = importlib.util.spec_from_file_location("rambo_physx_contract_for_independence_test", source_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tensor(values, *, dtype: str = "torch.float32") -> dict[str, object]:
    def shape(value):
        if not isinstance(value, list):
            return []
        return [len(value), *shape(value[0])]

    return {"shape": shape(values), "dtype": dtype, "values": values}


def _trace(step: int = 1) -> dict[str, object]:
    return {
        "step": step,
        "contact_source": schema.CONTACT_SOURCE_SCHEDULE,
        "source_contact_state": _tensor([[False, False, True, True]], dtype="torch.bool"),
        "contact_schedule_mode": _tensor([[-1.0, -1.0, 1.0, 1.0]]),
        "contact_schedule_phase": _tensor([[0.0, 0.0, 1.0, 1.0]]),
        "front_leg_logical_joint_slots": list(schema.FRONT_LEG_LOGICAL_JOINT_SLOTS),
        "front_leg_override_applied": True,
        "use_actual_contact": False,
        "contact_state_expanded": _tensor(
            [[True, True, False, False, True, True, False, False, True, True, False, False]],
            dtype="torch.bool",
        ),
        "grf": _tensor([[0.0] * schema.NUM_LOGICAL_JOINTS]),
        "desired_joint_torque": _tensor([[0.0] * schema.NUM_LOGICAL_JOINTS]),
        "desired_tor": _tensor([[0.0] * schema.NUM_LOGICAL_JOINTS]),
    }


def _wrench_output() -> dict[str, object]:
    return {
        "api": schema.PERMANENT_WRENCH_OUTPUT_API,
        "active": True,
        "shape": [1, 16, 3],
        "front_feet": {
            "fl": {
                "body_name": "FL_foot",
                "body_id": 13,
                "force_body": _tensor([[0.0, 0.0, 0.0]]),
            },
            "fr": {
                "body_name": "FR_foot",
                "body_id": 14,
                "force_body": _tensor([[0.0, -1.0, 0.0]]),
            },
        },
    }


def _command_buffers() -> dict[str, object]:
    return {
        "fl_position": _tensor([[0.22, 0.10, 0.48]]),
        "fr_position": _tensor([[0.24, -0.10, 0.50]]),
        "fl_force": _tensor([[0.0, 0.0, 0.0]]),
        "fr_force": _tensor([[0.0, 1.0, 0.0]]),
    }


def _observation_stats() -> dict[str, object]:
    return {
        "shape": [1, 435],
        "dtype": "torch.float32",
        "finite": True,
        "min": -1.0,
        "max": 1.0,
        "abs_max": 1.0,
    }


def _summary() -> dict[str, object]:
    physics = {
        "requested_cfg": schema.PHYSX_CFG_FQN,
        "actual_manager": schema.PHYSX_MANAGER_FQN,
        "configured_manager": schema.PHYSX_MANAGER_FQN,
        "use_newton_actuators": False,
    }
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_kind": schema.ARTIFACT_KIND,
        "passed": True,
        "provenance": {
            "schema_version": 1,
            "collection_phase": "pre_app_launcher_source",
            "module_path": "/workspace/rambo60/source/rambo/rambo/__init__.py",
            "git_root": "/workspace/rambo60",
            "git_head": "a" * 40,
            "pre_run_worktree_clean": True,
            "pre_run_status_porcelain_v1": [],
        },
        "task": schema.TASK_ID,
        "requested_steps": schema.REQUIRED_STEPS,
        "steps_completed": schema.REQUIRED_STEPS,
        "configured_physics": {"cfg": schema.PHYSX_CFG_FQN, "use_newton_actuators": False},
        "backend_before": physics,
        "backend_after": physics.copy(),
        "determinism": {"baseline_termination_flags": schema.BIPED_BASELINE_TERMINATION_FLAGS},
        "command_buffers_before_steps": _command_buffers(),
        "command_buffers_after_steps": _command_buffers(),
        "front_foot_body_ids": {
            "fl": {"body_name": "FL_foot", "body_id": 13},
            "fr": {"body_name": "FR_foot", "body_id": 14},
        },
        "zero_action": _tensor([[0.0] * 18]),
        "steps": [
            {
                "step": step,
                "observation": _observation_stats(),
                "qp_contact_trace": _trace(step),
                "permanent_wrench_output": _wrench_output(),
            }
            for step in range(1, schema.REQUIRED_STEPS + 1)
        ],
    }


def test_independence_smoke_import_and_cli_fail_closed_without_kit() -> None:
    module = _load_smoke_module()
    physx_contract = _load_physx_contract_module()

    class Parser:
        def error(self, message: str) -> None:
            raise ValueError(message)

    allowed = SimpleNamespace(
        visualizer=["none"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False, seed=42
    )
    assert module._validate_cli_contract(
        Parser(), allowed, ["--viz", "none"], validate_visualizer_args=physx_contract.validate_rambo_visualizer_args
    ) == ["none"]
    assert allowed.fast_shutdown is True
    assert allowed.livestream == 0

    missing_viz = SimpleNamespace(
        visualizer=["none"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False, seed=42
    )
    with pytest.raises(ValueError, match="--viz must be explicitly set"):
        module._validate_cli_contract(
            Parser(), missing_viz, [], validate_visualizer_args=physx_contract.validate_rambo_visualizer_args
        )

    rejected_viz = SimpleNamespace(
        visualizer=["rerun"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False, seed=42
    )
    with pytest.raises(ValueError, match="only --viz none and --viz kit"):
        module._validate_cli_contract(
            Parser(),
            rejected_viz,
            ["--viz", "rerun"],
            validate_visualizer_args=physx_contract.validate_rambo_visualizer_args,
        )

    mutable_seed = SimpleNamespace(
        visualizer=["none"], experience=None, kit_args=None, livestream=-1, fast_shutdown=False, seed=7
    )
    with pytest.raises(ValueError, match="seed is fixed"):
        module._validate_cli_contract(
            Parser(),
            mutable_seed,
            ["--viz", "none", "--seed", "7"],
            validate_visualizer_args=physx_contract.validate_rambo_visualizer_args,
        )


def test_independence_smoke_source_cannot_select_an_unaudited_physics_path() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "rambo" / "physx_biped_independence_smoke.py").read_text(encoding="utf-8")
    env_source = (
        root / "source" / "rambo" / "rambo" / "tasks" / "direct" / "rambo_biped" / "qp_env.py"
    ).read_text(encoding="utf-8")

    assert "validate_rambo_visualizer_args" in script
    assert "configure_physx" in script
    assert "assert_physx_environment" in script
    assert "use_newton_actuators" in script
    assert "isaaclab_newton" not in script
    assert "import newton" not in script
    assert "physics=Newton" not in script
    assert "set_runtime_qp_diagnostic_sink" in env_source
    assert "contact_state_expanded" in env_source
    assert "desired_joint_torque" in env_source
    assert "if self._runtime_qp_diagnostic_sink is not None" in env_source
    assert '_set_required_attr(env_cfg, "terminate_on_' not in script
    assert "_collect_clean_source_provenance" in script


def test_independence_configuration_disables_events_randomization_and_debug_visuals() -> None:
    module = _load_smoke_module()
    physx_type = type("PhysxCfg", (), {"__module__": "isaaclab_physx.physics.physx_manager_cfg"})
    cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=None, use_newton_actuators=True, use_fabric=True),
        scene=SimpleNamespace(num_envs=64),
        terrain=SimpleNamespace(debug_vis=True),
        seed=None,
        events=object(),
        randomize_initial_state=True,
        randomize_episode_progress=True,
        contact_phase_offset_s=3.0,
        obs_noise=True,
        action_noise_model=object(),
        observation_noise_model=object(),
        enable_rgb_camera=True,
        rerender_on_reset=True,
        num_rerenders_on_reset=1,
        wait_for_textures=True,
        enable_sampled_velocity_commands=True,
        enable_sampled_pos_commands=True,
        enable_sampled_force_commands=True,
        velocity_debug_vis=True,
        pos_debug_vis=True,
        force_debug_vis=True,
        use_actual_contact=True,
        add_feedforward_torque=False,
        terminate_on_undesired_foot_contact=False,
        terminate_on_low_base_height=True,
        terminate_on_large_orientation_error=True,
        terminate_on_body_contact=True,
        terminate_on_limb_contact=True,
        contact_generator_config={"contact_generator_debug_vis": True},
        joint_position_controller_config={"joint_position_controller_debug_vis": True},
        qp_torque_optimizer_config={"qp_debug_vis": True},
        fl_pos_x=[0.15, 0.30],
        fl_pos_y=[0.0, 0.20],
        fl_pos_z=[0.30, 0.90],
        fr_pos_x=[0.15, 0.30],
        fr_pos_y=[-0.20, 0.0],
        fr_pos_z=[0.30, 0.90],
        fl_force_x=[-20.0, 20.0],
        fl_force_y=[-20.0, 20.0],
        fl_force_z=[-20.0, 20.0],
        fr_force_x=[-20.0, 20.0],
        fr_force_y=[-20.0, 20.0],
        fr_force_z=[-20.0, 20.0],
    )

    def configure_physx(value):
        value.sim.physics = physx_type()
        value.sim.use_newton_actuators = False
        return value

    evidence = module._configure_independence_env_cfg(cfg, configure_physx=configure_physx, seed=42)

    assert evidence["cfg"] == schema.PHYSX_CFG_FQN
    assert evidence["use_newton_actuators"] is False
    determinism = module._determinism_evidence(cfg)
    assert determinism["events_disabled"] is True
    assert all(value is False for value in determinism["sampled_commands"].values())
    assert all(value is False for value in determinism["debug_visualization"].values())
    assert determinism["baseline_termination_flags"] == module.BIPED_BASELINE_TERMINATION_FLAGS
    assert set(module._validate_safe_command_ranges(cfg)) == {
        "fl_position",
        "fr_position",
        "fl_force",
        "fr_force",
    }


def test_independence_zero_action_requires_the_batched_isaaclab_space_shape() -> None:
    torch = pytest.importorskip("torch")
    module = _load_smoke_module()
    base_env = SimpleNamespace(
        action_space=SimpleNamespace(shape=(1, 18)),
        num_envs=1,
        device=torch.device("cpu"),
    )

    action = module._zero_action(base_env, torch)

    assert tuple(action.shape) == (1, 18)
    assert bool(torch.all(action == 0.0))
    base_env.action_space = SimpleNamespace(shape=(18,))
    with pytest.raises(module.BipedIndependenceSmokeError, match="must have shape"):
        module._zero_action(base_env, torch)


def test_trace_schema_proves_contact_override_and_front_wrench_independence() -> None:
    trace = _trace()
    commands = _command_buffers()
    wrench = _wrench_output()

    schema.validate_qp_contact_trace(trace)
    schema.validate_command_buffers(commands)
    schema.validate_permanent_wrench_output(wrench)
    schema.validate_biped_independence_summary(_summary())

    dirty = _summary()
    dirty["provenance"]["pre_run_status_porcelain_v1"] = [" M qp_env.py"]
    with pytest.raises(schema.BipedIndependenceError, match="empty pre-run porcelain"):
        schema.validate_biped_independence_summary(dirty)

    trace["source_contact_state"]["values"][0][0] = True
    with pytest.raises(schema.BipedIndependenceError, match="raw FL/FR desired contacts"):
        schema.validate_qp_contact_trace(trace)

    wrench["front_feet"]["fr"]["body_id"] = wrench["front_feet"]["fl"]["body_id"]
    with pytest.raises(schema.BipedIndependenceError, match="physical body ids"):
        schema.validate_permanent_wrench_output(wrench)


def test_independence_artifact_writes_a_summary_checksum(tmp_path: Path) -> None:
    module = _load_smoke_module()
    summary = _summary()
    module._assert_json_finite(summary)
    module._write_summary(tmp_path, summary)

    summary_bytes = (tmp_path / "summary.json").read_bytes()
    assert (tmp_path / "checksums.sha256").read_text(encoding="utf-8") == (
        f"{hashlib.sha256(summary_bytes).hexdigest()}  summary.json\n"
    )
