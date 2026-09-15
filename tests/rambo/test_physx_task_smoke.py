"""Simulator-free contracts for the opt-in M6 task-level PhysX evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_task_smoke_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "rambo" / "physx_task_smoke.py"
    specification = importlib.util.spec_from_file_location("rambo_physx_task_smoke", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _m6_cfg():
    return SimpleNamespace(
        seed=None,
        events=object(),
        randomize_initial_state=True,
        obs_noise=True,
        randomize_episode_progress=True,
        enable_sampled_velocity_commands=True,
        enable_sampled_pos_commands=True,
        enable_sampled_force_commands=True,
        enable_rgb_camera=True,
        velocity_debug_vis=True,
        pos_debug_vis=True,
        force_debug_vis=True,
        action_noise_model=object(),
        observation_noise_model=object(),
        sim=SimpleNamespace(dt=0.002),
        decimation=5,
        episode_length_s=10.0,
        action_scale=[5.0] * 6 + [0.15] * 12,
        terminate_on_undesired_foot_contact=False,
        terminate_on_low_base_height=True,
        terminate_on_large_orientation_error=True,
        terminate_on_body_contact=True,
        terminate_on_limb_contact=True,
        contact_generator_config={
            "contact_generator_debug_vis": True,
            "contact_sequence": {
                "FL": [["stance", 1.0, 0.0, 0.0, 0.0], ["swing", 9.0, 0.0, 0.0, 0.0]],
                "FR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.0, 0.0, 0.0]],
                "RL": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.0, 0.0, 0.0]],
                "RR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 9.0, 0.0, 0.0, 0.0]],
            },
        },
        joint_position_controller_config={"joint_position_controller_debug_vis": True},
        qp_torque_optimizer_config={"qp_debug_vis": True},
    )


def test_task_smoke_is_importable_without_starting_kit() -> None:
    module = _load_task_smoke_module()

    assert module.TASKS["quadruped"] == ("Isaac-RAMBO-Quadruped-Go2-v0", 405)
    assert module.M6_ZERO_ACTION_STEPS == (100, 1000)


def test_m6_task_script_keeps_the_opt_in_physx_and_public_data_contracts_explicit() -> None:
    root = Path(__file__).resolve().parents[2]
    contents = (root / "scripts" / "rambo" / "physx_task_smoke.py").read_text(encoding="utf-8")

    assert "--m6-zero-action-contract" in contents
    assert "M6_ZERO_ACTION_STEPS = (100, 1000)" in contents
    assert contents.index("configure_physx(env_cfg)") < contents.rindex("_configure_m6_zero_action_scene(")
    assert "actual_physx_manager_before" in contents
    assert "actual_physx_manager_after" in contents
    assert "use_newton_actuators=False" in contents
    assert "scene.articulations['robot']" in contents
    assert "scene.sensors['contact_sensor']" in contents
    assert "ordered_jacobians" in contents
    assert "M6_TRACE_FILENAME" in contents


def test_m6_zero_action_contract_rejects_nonquadruped_or_unapproved_length() -> None:
    module = _load_task_smoke_module()

    module._validate_m6_zero_action_request("quadruped", 100)
    module._validate_m6_zero_action_request("quadruped", 1000)
    with pytest.raises(module.TaskSmokeError, match="only permits --task quadruped"):
        module._validate_m6_zero_action_request("unsupported", 100)
    with pytest.raises(module.TaskSmokeError, match="one of 100, 1000"):
        module._validate_m6_zero_action_request("quadruped", 10)


def test_m6_zero_action_scene_is_event_free_deterministic_and_extends_only_its_copy() -> None:
    module = _load_task_smoke_module()
    cfg = _m6_cfg()
    original_contact_config = cfg.contact_generator_config

    summary = module._configure_m6_zero_action_scene(cfg, seed=42, steps=1000)

    assert cfg.seed == 42
    assert cfg.events is None
    assert cfg.randomize_initial_state is False
    assert cfg.obs_noise is False
    assert cfg.randomize_episode_progress is False
    assert cfg.enable_rgb_camera is False
    assert cfg.action_noise_model is None
    assert cfg.observation_noise_model is None
    assert cfg.episode_length_s == pytest.approx(11.0)
    assert cfg.contact_generator_config is not original_contact_config
    assert original_contact_config["contact_sequence"]["FL"][-1][1] == 9.0
    assert cfg.contact_generator_config["contact_sequence"] == {
        foot_name: [["stance", pytest.approx(11.0), 0.0, 0.0, 0.0]]
        for foot_name in module.M6_ALL_STANCE_FEET
    }
    assert summary["contact_schedule_profile"] == "m6_diagnostic_all_stance"
    assert summary["production_contact_schedule_mutated"] is False
    assert len(summary["production_contact_sequence_sha256"]) == 64
    assert summary["production_contact_sequence"] == original_contact_config["contact_sequence"]
    assert summary["effective_contact_sequence"] == cfg.contact_generator_config["contact_sequence"]
    assert summary["contact_sequence_total_duration_s"]["FR"] == pytest.approx(11.0)
    assert summary["action_scale"] == [5.0] * 6 + [0.15] * 12
    assert summary["safety_termination_flags"] == module.M6_BASELINE_TERMINATION_FLAGS
    assert summary["debug_visualization_disabled"] == {
        "contact_generator": True,
        "joint_position_controller": True,
        "qp_torque_optimizer": True,
    }


def test_m6_public_scene_contract_records_ordering_and_finite_state_without_private_fields() -> None:
    torch = pytest.importorskip("torch")
    module = _load_task_smoke_module()

    class Proxy:
        def __init__(self, value):
            self.torch = value

    body_order = ("base", "leg")
    joint_order = ("joint_a", "joint_b")
    data = SimpleNamespace(
        body_names=["base", "leg", "Head_link"],
        joint_names=list(joint_order),
        body_com_jacobian_w=Proxy(torch.zeros((1, 2, 6, 8), dtype=torch.float32)),
        root_link_pos_w=Proxy(torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32)),
        root_link_quat_w=Proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)),
        root_link_lin_vel_w=Proxy(torch.zeros((1, 3), dtype=torch.float32)),
        root_link_ang_vel_w=Proxy(torch.zeros((1, 3), dtype=torch.float32)),
        joint_pos=Proxy(torch.tensor([[0.1, -0.2]], dtype=torch.float32)),
        joint_vel=Proxy(torch.zeros((1, 2), dtype=torch.float32)),
    )
    robot = SimpleNamespace(data=data)

    class ContactSensor:
        data = SimpleNamespace(net_forces_w_history=Proxy(torch.zeros((1, 2, 3, 3), dtype=torch.float32)))

        def find_sensors(self, names, preserve_order=False):
            if names == "Head_.*":
                return [2], ["Head_link"]
            assert preserve_order is True
            assert names == list(body_order)
            return [0, 1], list(body_order)

    sensor = ContactSensor()
    base_env = SimpleNamespace(
        scene=SimpleNamespace(articulations={"robot": robot}, sensors={"contact_sensor": sensor}),
        contact_generator=SimpleNamespace(
            desired_contact_mode=torch.ones((1, 4), dtype=torch.float32),
            desired_contact_state=torch.ones((1, 4), dtype=torch.bool),
        ),
    )
    indices = SimpleNamespace(
        body_ids=torch.tensor([0, 1]),
        joint_ids=torch.tensor([0, 1]),
        foot_body_ids=torch.tensor([1]),
        head_body_ids=torch.tensor([2]),
    )
    contract = module._resolve_m6_scene_contract(
        base_env,
        torch=torch,
        body_order=body_order,
        joint_order=joint_order,
        resolve_go2_indices=lambda resolved_robot: indices,
        ordered_jacobians=lambda resolved_robot, resolved_indices: resolved_robot.data.body_com_jacobian_w.torch,
    )
    state = module._collect_m6_public_state(contract, step=1, num_envs=1, torch=torch)

    assert contract["summary"]["articulation"]["joint_names"] == list(joint_order)
    assert contract["summary"]["contact_sensor"]["body_ids"] == [0, 1]
    assert contract["summary"]["jacobian_ordering"]["raw_omits_root_row"] is True
    assert state["state_finite"] is True
    assert state["metrics"]["joint_position_rad"]["shape"] == [1, 2]
    assert state["sample"]["root_position_w_m_env0"] == [0.0, 0.0, pytest.approx(0.3)]
    assert state["metrics"]["root_quaternion_xyzw"]["shape"] == [1, 4]
    assert state["sample"]["root_quaternion_xyzw_env0"] == [0.0, 0.0, 0.0, 1.0]
    assert state["m6_all_stance_scheduler"]["all_four_feet_stance"] is True


def test_m6_zero_action_dones_fail_closed_before_a_reset_can_be_hidden() -> None:
    torch = pytest.importorskip("torch")
    module = _load_task_smoke_module()

    transition = ({}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {})
    assert module._zero_action_dones(transition, num_envs=1, torch=torch)["all_dones_false"] is True
    terminal = ({}, torch.zeros(1), torch.tensor([True]), torch.tensor([False]), {})
    with pytest.raises(module.TaskSmokeError, match="terminated=True"):
        module._zero_action_dones(terminal, num_envs=1, torch=torch)


def test_m6_all_stance_scheduler_and_episode_counter_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    module = _load_task_smoke_module()

    scheduler = SimpleNamespace(
        desired_contact_mode=torch.ones((1, 4), dtype=torch.float32),
        desired_contact_state=torch.ones((1, 4), dtype=torch.bool),
    )
    record = module._m6_all_stance_scheduler_record(scheduler, num_envs=1, torch=torch)
    assert record["foot_order"] == ["FL", "FR", "RL", "RR"]
    assert record["all_four_feet_contact"] is True
    scheduler.desired_contact_mode[0, 0] = 0.0
    with pytest.raises(module.TaskSmokeError, match="non-stance contact mode"):
        module._m6_all_stance_scheduler_record(scheduler, num_envs=1, torch=torch)

    base_env = SimpleNamespace(episode_length_buf=torch.tensor([7], dtype=torch.long))
    assert module._m6_episode_counter(base_env, expected_step=7, num_envs=1, torch=torch) == [7]
    with pytest.raises(module.TaskSmokeError, match="did not advance monotonically"):
        module._m6_episode_counter(base_env, expected_step=8, num_envs=1, torch=torch)


def test_m6_state_trace_is_included_with_summary_in_complete_checksums(tmp_path: Path) -> None:
    module = _load_task_smoke_module()
    trace = tmp_path / module.M6_TRACE_FILENAME
    trace.write_text('{"step":0,"state_finite":true}\n', encoding="utf-8")

    module._write_summary_and_checksums(tmp_path, {"passed": True}, (trace,))

    checksummed_files = {
        line.split(maxsplit=1)[1]
        for line in (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    }
    assert checksummed_files == {"summary.json", module.M6_TRACE_FILENAME}
