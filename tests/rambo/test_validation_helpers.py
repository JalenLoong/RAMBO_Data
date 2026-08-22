"""Simulator-free acceptance helpers for deterministic RAMBO validation."""

from __future__ import annotations

import copy
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rambo.actuators.delayed_dc_motor import _batch_count  # noqa: E402
from rambo.actuators.go2 import (  # noqa: E402
    GO2_ACTUATOR_DELAY_STEPS,
    make_go2_delayed_dc_motor_cfgs,
)
from rambo.validation.rollout import RgbFrameRecorder, configure_validation_cfg  # noqa: E402
from rambo.tasks.common.camera import (  # noqa: E402
    advance_camera_cadence,
    camera_update_interval_steps,
    make_front_rgb_camera_cfg,
)


def test_batch_count_accepts_none_slices_sequences_and_tensor_ids() -> None:
    assert _batch_count(None, total=5) == 5
    assert _batch_count(slice(None), total=5) == 5
    assert _batch_count(slice(1, 5, 2), total=5) == 2
    assert _batch_count([0, 3, 4], total=5) == 3
    assert _batch_count((2,), total=5) == 1
    assert _batch_count(torch.tensor([0, 2, 4]), total=5) == 3
    assert _batch_count(torch.empty(0, dtype=torch.long), total=5) == 0


def test_both_modes_share_the_calibrated_go2_delay_and_saturation_contract() -> None:
    """Lock the checkpoint-compatible motor contract outside Kit runtime tests."""

    motors = make_go2_delayed_dc_motor_cfgs()
    assert set(motors) == {"calf", "hip_thigh"}
    calf = motors["calf"]
    hip_thigh = motors["hip_thigh"]

    assert calf.joint_names_expr == [".*_calf_joint"]
    assert (calf.effort_limit, calf.saturation_effort, calf.velocity_limit) == (40.887, 40.887, 15.70)
    assert hip_thigh.joint_names_expr == [".*_hip_joint", ".*_thigh_joint"]
    assert (hip_thigh.effort_limit, hip_thigh.saturation_effort, hip_thigh.velocity_limit) == (
        21.33,
        21.33,
        30.1,
    )
    for motor in motors.values():
        assert (motor.min_delay, motor.max_delay) == (0, GO2_ACTUATOR_DELAY_STEPS)
        assert (motor.stiffness, motor.damping) == (40.0, 1.0)
        assert (motor.friction, motor.dynamic_friction, motor.viscous_friction) == (0.0, 0.0, 0.0)


def test_front_camera_integer_cadence_is_exact_for_the_full_acceptance_rollout() -> None:
    """30 s at 500 Hz must cross precisely 375 80 ms RGB deadlines."""

    interval_steps = camera_update_interval_steps(0.08, 0.002)
    assert interval_steps == 40

    elapsed_physics_steps = 0
    due_physics_steps = []
    for _ in range(3000):
        # RAMBO's 100 Hz policy action is decimated into five 2 ms physics steps.
        elapsed_physics_steps, due = advance_camera_cadence(
            elapsed_physics_steps,
            elapsed_step_count=5,
            interval_steps=interval_steps,
        )
        if due:
            due_physics_steps.append(elapsed_physics_steps)

    assert elapsed_physics_steps == 15_000
    assert len(due_physics_steps) == 375
    assert due_physics_steps[:3] == [40, 80, 120]
    assert due_physics_steps[-1] == 15_000


def test_validation_config_removes_all_randomization_without_mutating_source_sequence() -> None:
    source_contact_config = {
        "contact_sequence": {
            "FL_foot": [[1, 4.0], [0, 3.0]],
            "FR_foot": [[1, 2.0], [0, 5.0]],
        }
    }
    original_contact_config = copy.deepcopy(source_contact_config)
    cfg = SimpleNamespace(
        episode_length_s=20.0,
        scene=SimpleNamespace(lazy_sensor_update=True),
        events=object(),
        randomize_initial_state=True,
        obs_noise=True,
        randomize_episode_progress=True,
        enable_sampled_velocity_commands=True,
        enable_sampled_pos_commands=True,
        enable_sampled_force_commands=True,
        enable_rgb_camera=False,
        contact_generator_config=source_contact_config,
        joint_position_controller_config={"joint_position_controller_debug_vis": True},
        qp_torque_optimizer_config={"qp_debug_vis": True},
    )

    configured = configure_validation_cfg(cfg, duration_s=31.0, enable_rgb_camera=True)

    assert configured is cfg
    assert cfg.episode_length_s == 31.0
    assert cfg.events is None
    assert cfg.enable_rgb_camera is True
    assert cfg.scene.lazy_sensor_update is True
    for name in (
        "randomize_initial_state",
        "obs_noise",
        "randomize_episode_progress",
        "enable_sampled_velocity_commands",
        "enable_sampled_pos_commands",
        "enable_sampled_force_commands",
    ):
        assert getattr(cfg, name) is False
    assert source_contact_config == original_contact_config
    assert cfg.contact_generator_config is not source_contact_config
    for sequence in cfg.contact_generator_config["contact_sequence"].values():
        assert sum(segment[1] for segment in sequence) == 31.0
    assert cfg.joint_position_controller_config["joint_position_controller_debug_vis"] is False
    assert cfg.qp_torque_optimizer_config["qp_debug_vis"] is False


def test_validation_config_supports_a_fixed_biped_phase_without_using_episode_progress() -> None:
    source_contact_config = {"contact_sequence": {"RL": [["stance", 1.0], ["phase", 2.0]]}}
    cfg = SimpleNamespace(
        episode_length_s=20.0,
        randomize_episode_progress=True,
        contact_phase_offset_s=0.0,
        validation_contact_phase_offset_s=2.5,
        contact_generator_config=source_contact_config,
    )

    configure_validation_cfg(cfg, duration_s=31.0, enable_rgb_camera=False)

    assert cfg.randomize_episode_progress is False
    assert cfg.contact_phase_offset_s == 2.5
    assert cfg.episode_length_s == 31.0
    assert sum(segment[1] for segment in cfg.contact_generator_config["contact_sequence"]["RL"]) == 33.5
    assert source_contact_config["contact_sequence"]["RL"][-1][1] == 2.0


def _install_minimal_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep frame freshness tests independent of the optional PNG encoder."""

    try:
        import imageio.v2  # noqa: F401
    except ModuleNotFoundError:
        imageio_module = types.ModuleType("imageio")
        v2_module = types.ModuleType("imageio.v2")

        def imwrite(path, _frame) -> None:
            # Recorder behavior, not the encoder, is under test here.
            Path(path).write_bytes(b"rambo-test-image")

        from pathlib import Path

        v2_module.imwrite = imwrite
        imageio_module.v2 = v2_module
        monkeypatch.setitem(sys.modules, "imageio", imageio_module)
        monkeypatch.setitem(sys.modules, "imageio.v2", v2_module)


def _rgb_frame(offset: int) -> np.ndarray:
    y, x = np.indices((480, 640), dtype=np.uint16)
    frame = np.empty((1, 480, 640, 3), dtype=np.uint8)
    frame[..., 0] = (x + offset) % 256
    frame[..., 1] = (y + 2 * offset) % 256
    frame[..., 2] = (x // 3 + y // 4 + 7 * offset) % 256
    return frame


def test_rgb_recorder_consumes_only_consecutive_fresh_frames(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_minimal_imageio(monkeypatch)
    camera = SimpleNamespace(
        frame=torch.tensor([40]),
        data=SimpleNamespace(output={"rgb": _rgb_frame(0)}),
    )
    recorder = RgbFrameRecorder(tmp_path / "validation", expected_count=2)

    recorder.prime(camera)
    assert recorder.capture_if_new(camera, 0.00) is False

    camera.frame = torch.tensor([41])
    camera.data.output["rgb"] = _rgb_frame(20)
    assert recorder.capture_if_new(camera, 0.08) is True
    assert recorder.capture_if_new(camera, 0.09) is False

    camera.frame = torch.tensor([42])
    camera.data.output["rgb"] = _rgb_frame(40)
    assert recorder.capture_if_new(camera, 0.16) is True

    summary = recorder.finalize()
    assert summary["passed"] is True
    assert summary["frame_count"] == 2
    assert np.load(tmp_path / "validation" / "rgb_frame_ids.npy").tolist() == [41, 42]
    assert np.load(tmp_path / "validation" / "rgb_timestamps.npy").tolist() == [0.08, 0.16]
    assert len(list((tmp_path / "validation" / "rgb").glob("*.png"))) == 2


def test_rgb_recorder_reads_lazy_camera_data_before_frame(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isaac Lab cameras increment their frame only when ``data`` is read."""

    _install_minimal_imageio(monkeypatch)

    class LazyCamera:
        def __init__(self) -> None:
            self._frame = 0
            self.outdated = True
            self._data = SimpleNamespace(output={"rgb": _rgb_frame(30)})

        @property
        def data(self):
            if self.outdated:
                self._frame += 1
                self.outdated = False
            return self._data

        @property
        def frame(self):
            return torch.tensor([self._frame])

    camera = LazyCamera()
    recorder = RgbFrameRecorder(tmp_path / "lazy-camera", expected_count=1)
    recorder.prime(camera)
    assert camera.frame.item() == 1

    camera.outdated = True
    assert recorder.capture_if_new(camera, 0.08) is True
    recorder.finalize()
    assert np.load(tmp_path / "lazy-camera" / "rgb_frame_ids.npy").tolist() == [2]
