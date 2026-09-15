"""Coordinate and opt-in isolation tests for the robot-mounted camera pair."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rambo.tasks.common import lift_camera_rig as rig


def test_mount_pitch_and_rigid_motion_and_renderer_crosscheck():
    root = np.array([.25, .12, .30, 0., 0., math.sin(.35), math.cos(.35)])
    position, rotation = rig.camera_world_transform(root, "task")
    assert rotation[2, 0] == pytest.approx(-math.sin(math.radians(65)))
    np.testing.assert_allclose(position, root[:3] + rig.rotation_matrix(root[3:]) @ [.30, 0, .34])
    world_gl = np.eye(4)
    world_gl[:3, :3] = np.stack((-rotation[:, 1], rotation[:, 2], -rotation[:, 0]))
    world_gl[3, :3] = position
    assert rig.validate_renderer_transform(root, "task", np.linalg.inv(world_gl))["position_max_error_m"] < 1e-12
    stale = root.copy()
    stale[0] -= .1
    with pytest.raises(ValueError, match="RTX/mount mismatch"):
        rig.validate_renderer_transform(stale, "task", np.linalg.inv(world_gl))


def test_pair_config_preserves_native_ego_and_has_no_external_views(monkeypatch):
    from rambo.tasks.common import camera
    monkeypatch.setattr(camera, "make_front_rgb_camera_cfg", lambda **kw:
                        SimpleNamespace(**kw, spawn=SimpleNamespace()))
    cfg = SimpleNamespace(task_kind="lift_basket", enable_rgb_camera=False)
    rig.configure(cfg)
    assert set(rig.CAMERAS) == {"ego", "task"}
    assert cfg.front_camera.offset_rot == (0., 0., 0., 1.)
    assert cfg.front_camera.spawn.focal_length == rig.EGO_FOCAL_MM
    assert cfg.task_camera.spawn.focal_length == 1.88
    assert cfg.task_camera.prim_path.startswith(rig.BASE_PATH + "/")
    with pytest.raises(ValueError, match="Lift-basket only"):
        rig.configure(SimpleNamespace(task_kind="press_button"))


def test_nominal_ego_projection_is_explicit(monkeypatch):
    from rambo.tasks.common import camera
    monkeypatch.setattr(camera, "make_front_rgb_camera_cfg", lambda **kw:
                        SimpleNamespace(**kw, spawn=SimpleNamespace()))
    cfg = SimpleNamespace(task_kind="lift_basket")
    rig.configure(cfg)
    assert (cfg.front_camera.width, cfg.front_camera.height) == (1280, 720)
    assert cfg.front_camera.offset_pos == (.32715, -.00003, .04297)
    diagonal = math.hypot(cfg.front_camera.spawn.horizontal_aperture, cfg.front_camera.spawn.vertical_aperture)
    assert math.degrees(2 * math.atan(diagonal / (2 * cfg.front_camera.spawn.focal_length))) == pytest.approx(120)


def test_v3_task_camera_matches_d435i_rgb_fov_and_preserves_mount(monkeypatch):
    from rambo.tasks.common import camera
    monkeypatch.setattr(camera, "make_front_rgb_camera_cfg", lambda **kw:
                        SimpleNamespace(**kw, spawn=SimpleNamespace()))
    cfg = SimpleNamespace(task_kind="lift_basket")
    rig.configure(cfg)
    task = rig.CAMERAS["task"]
    assert (cfg.task_camera.width, cfg.task_camera.height) == (1280, 720)
    assert cfg.task_camera.offset_pos == (.30, 0., .34)
    assert cfg.task_camera.offset_rot == (0., math.sin(math.radians(32.5)), 0., math.cos(math.radians(32.5)))
    assert cfg.task_camera.spawn.focal_length == 1.88
    horizontal = math.degrees(2 * math.atan(
        cfg.task_camera.spawn.horizontal_aperture / (2 * cfg.task_camera.spawn.focal_length)))
    vertical = math.degrees(2 * math.atan(
        cfg.task_camera.spawn.vertical_aperture / (2 * cfg.task_camera.spawn.focal_length)))
    assert horizontal == pytest.approx(69.4)
    assert vertical == pytest.approx(rig.D435I_RGB_VERTICAL_FOV_SIMULATED_DEG)
    assert abs(vertical - rig.D435I_RGB_VERTICAL_FOV_PUBLISHED_DEG) < .1
    assert cfg.task_camera.width / cfg.task_camera.spawn.horizontal_aperture == pytest.approx(
        cfg.task_camera.height / cfg.task_camera.spawn.vertical_aperture)
