"""Opt-in robot-mounted Lift-basket cameras; imports Isaac only at runtime."""

from __future__ import annotations

import math
import copy

import numpy as np

SETUP_ID = "robot-dual-v3"
SETUP_IDS = ("robot-dual-v1", "robot-dual-v2", SETUP_ID)
BASE_PATH = "/World/envs/env_0/Robot/base"
RIG_PATH = BASE_PATH + "/task_camera_rig"
CAMERAS_V1 = {
    "ego": {"path": BASE_PATH + "/front_camera", "position": (0.30, 0.0, 0.08),
            "rotation_xyzw": (0.0, 0.0, 0.0, 1.0), "focal_length_mm": 18.0},
    "task": {"path": RIG_PATH + "/task_camera", "position": (0.30, 0.0, 0.34),
             "rotation_xyzw": (0.0, math.sin(math.radians(32.5)), 0.0, math.cos(math.radians(32.5))),
             "focal_length_mm": 9.0},
}
CAMERAS_V2 = copy.deepcopy(CAMERAS_V1)
APERTURE_MM = 20.955
EGO_DIAGONAL_FOV_DEG = 120.0  # Brochure does not identify the axis: explicit assumption.
EGO_FOCAL_MM = math.hypot(APERTURE_MM, APERTURE_MM * 720 / 1280) / (2 * math.tan(math.radians(60)))
CAMERAS_V2["ego"].update({
    "position": (0.32715, -0.00003, 0.04297), "focal_length_mm": EGO_FOCAL_MM,
    "width": 1280, "height": 720, "horizontal_aperture_mm": APERTURE_MM,
    "vertical_aperture_mm": APERTURE_MM * 720 / 1280,
    "calibration_status": "nominal_specification_approximation_not_device_calibration",
    "fov_axis_assumption": "diagonal", "diagonal_fov_deg": 120.,
    "projection": "centered_rectilinear_pinhole_no_distortion",
    "mount_source": "https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf",
    "fov_source": "https://static.generation-robots.com/media/brochure-unitree-go2-en.pdf",
})
CAMERAS = copy.deepcopy(CAMERAS_V2)
D435I_RGB_FOCAL_LENGTH_MM = 1.88
D435I_RGB_HORIZONTAL_FOV_DEG = 69.4
D435I_RGB_VERTICAL_FOV_PUBLISHED_DEG = 42.5
D435I_RGB_HORIZONTAL_APERTURE_MM = 2 * D435I_RGB_FOCAL_LENGTH_MM * math.tan(
    math.radians(D435I_RGB_HORIZONTAL_FOV_DEG / 2)
)
D435I_RGB_VERTICAL_APERTURE_MM = D435I_RGB_HORIZONTAL_APERTURE_MM * 720 / 1280
D435I_RGB_VERTICAL_FOV_SIMULATED_DEG = math.degrees(
    2 * math.atan(D435I_RGB_VERTICAL_APERTURE_MM / (2 * D435I_RGB_FOCAL_LENGTH_MM))
)
CAMERAS["task"].update({
    "focal_length_mm": D435I_RGB_FOCAL_LENGTH_MM,
    "width": 1280,
    "height": 720,
    "horizontal_aperture_mm": D435I_RGB_HORIZONTAL_APERTURE_MM,
    "vertical_aperture_mm": D435I_RGB_VERTICAL_APERTURE_MM,
    "sensor_model": "Intel RealSense D435i RGB / OmniVision OV2740 nominal specification",
    "horizontal_fov_deg": D435I_RGB_HORIZONTAL_FOV_DEG,
    "vertical_fov_published_deg": D435I_RGB_VERTICAL_FOV_PUBLISHED_DEG,
    "vertical_fov_simulated_deg": D435I_RGB_VERTICAL_FOV_SIMULATED_DEG,
    "pixel_aspect_ratio": 1.0,
    "projection": "centered_rectilinear_pinhole_no_device_distortion",
    "specification_source": "https://www.realsenseai.com/wp-content/uploads/2022/03/Intel-RealSense-D400-Series-Datasheet-March-2022.pdf",
})


def cameras_for_setup(setup_id):
    try:
        return {
            "robot-dual-v1": CAMERAS_V1,
            "robot-dual-v2": CAMERAS_V2,
            "robot-dual-v3": CAMERAS,
        }[setup_id]
    except KeyError as error:
        raise ValueError(f"Unknown camera setup: {setup_id}") from error


def rotation_matrix(q):
    """Quaternion XYZW to local-to-parent rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
        raise ValueError("Expected finite unit XYZW quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def camera_world_transform(root_xyzw, view):
    """World XYZ and world-convention axes (+X forward, +Y left, +Z up)."""
    root = np.asarray(root_xyzw, dtype=np.float64)
    if root.shape != (7,) or not np.isfinite(root).all():
        raise ValueError("Expected finite XYZ/XYZW root pose")
    mount = CAMERAS[view]
    rotation = rotation_matrix(root[3:])
    return root[:3] + rotation @ mount["position"], rotation @ rotation_matrix(mount["rotation_xyzw"])


def validate_renderer_transform(root_xyzw, view, view_matrix):
    """Check native RTX row-vector OpenGL view matrix against moving mount."""
    view_matrix = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(view_matrix).all():
        raise ValueError("Nonfinite RTX matrix")
    world = np.linalg.inv(view_matrix)
    position, rotation = camera_world_transform(root_xyzw, view)
    gl_axes = np.stack((-rotation[:, 1], rotation[:, 2], -rotation[:, 0]))
    pos_error = float(np.max(np.abs(world[3, :3] - position)))
    axis_error = float(np.max(np.abs(world[:3, :3] - gl_axes)))
    if max(pos_error, axis_error) > 1e-5:
        raise ValueError(f"{view} RTX/mount mismatch: position={pos_error}, axes={axis_error}")
    return {"position_max_error_m": pos_error, "axes_max_error": axis_error}


def configure(env_cfg, setup_id=SETUP_ID):
    """Configure two real sensors without changing legacy task defaults."""
    from .camera import make_front_rgb_camera_cfg

    if env_cfg.task_kind != "lift_basket":
        raise ValueError("robot-dual camera setup is Lift-basket only")
    if setup_id not in SETUP_IDS:
        raise ValueError(f"Unknown camera setup: {setup_id}")
    env_cfg.lift_camera_setup = setup_id
    env_cfg.enable_rgb_camera = True
    for view, field in (("ego", "front_camera"), ("task", "task_camera")):
        mount = cameras_for_setup(setup_id)[view]
        cfg = make_front_rgb_camera_cfg(prim_path=mount["path"],
                                       width=mount.get("width", 640), height=mount.get("height", 480),
                                       offset_pos=mount["position"], offset_rot=mount["rotation_xyzw"])
        cfg.spawn.focal_length = mount["focal_length_mm"]
        if "vertical_aperture_mm" in mount:
            cfg.spawn.horizontal_aperture = mount["horizontal_aperture_mm"]
            cfg.spawn.vertical_aperture = mount["vertical_aperture_mm"]
        cfg.spawn.lock_camera = True
        setattr(env_cfg, field, cfg)


def spawn(env):
    """Parent a visual bracket/housing and task sensor to the existing base."""
    import isaaclab.sim as sim
    import omni.usd
    from pxr import UsdGeom
    from .camera import create_front_rgb_camera

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(RIG_PATH).IsValid():
        raise ValueError(f"Refusing existing camera rig: {RIG_PATH}")
    UsdGeom.Xform.Define(stage, RIG_PATH)

    def box(name, position, size, rotation=(0., 0., 0., 1.), color=(.10, .12, .15)):
        cfg = sim.CuboidCfg(size=size, visual_material=sim.PreviewSurfaceCfg(diffuse_color=color))
        cfg.func(RIG_PATH + "/" + name, cfg, translation=position, orientation=rotation)

    box("mount_plate", (.12, 0., .095), (.10, .10, .025))
    # A thin upright and a short forward boom, all local to the robot base.
    box("support", (.15, 0., .205), (.025, .035, .22))
    box("boom", (.215, 0., .315), (.155, .035, .025))
    rotation = CAMERAS["task"]["rotation_xyzw"]
    forward = rotation_matrix(rotation)[:, 0]
    center = np.array(CAMERAS["task"]["position"]) - .035 * forward
    box("camera_housing", tuple(center), (.07, .085, .045), rotation)
    # Lens face remains behind the optical center and outside its frustum.
    box("lens_face", tuple(center + .032 * forward), (.004, .028, .028), rotation, (.015, .035, .05))
    env.task_camera = create_front_rgb_camera(env.cfg.task_camera)
    env.scene.sensors["task_camera"] = env.task_camera
