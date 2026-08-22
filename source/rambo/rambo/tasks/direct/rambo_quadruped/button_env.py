"""RAMBO Go2 quadruped with a physical spring-loaded wall button.

The released policy remains the original 405-observation/18-action quadruped
checkpoint. Button state is task-side telemetry for the loco-manip teleop
runner and never changes the policy observation or network architecture.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import omni.usd
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .qp_env import QPEnv, QPEnvCfg


@configclass
class ButtonQPEnvCfg(QPEnvCfg):
    """Deterministic single-FL loco-manipulation configuration."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=10.0,
        replicate_physics=True,
    )
    episode_length_s = 300.0
    randomize_episode_progress = False
    randomize_initial_state = False
    enable_sampled_velocity_commands = False
    enable_sampled_pos_commands = False
    enable_sampled_force_commands = False
    obs_noise = False
    events = None

    qp_torque_optimizer_config = {
        "qp_debug_vis": False,
        "base_position_kp": np.array([50.0, 50.0, 50.0]),
        "base_position_kd": np.array([10.0, 10.0, 10.0]),
        "base_orientation_kp": np.array([50.0, 50.0, 50.0]),
        "base_orientation_kd": np.array([10.0, 10.0, 10.0]),
        "qp_weight_ddq": np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "qp_weight_grf": 1e-4,
        "qp_weight_ee_force": 1.0,
        "qp_foot_friction_coef": 0.6,
    }
    velocity_debug_vis = False
    pos_debug_vis = False
    force_debug_vis = False

    # FL becomes the manipulator after one second; the remaining three legs
    # retain the checkpoint-compatible walking contact phases.
    contact_generator_config = {
        "contact_generator_debug_vis": False,
        "contact_sequence": {
            "FL": [["stance", 1.0, 0.0, 0.0, 0.0], ["swing", 299.0, 0.0, 0.0, 0.0]],
            "FR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 299.0, 0.0, 0.7, 0.8]],
            "RL": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 299.0, 0.33, 0.7, 0.8]],
            "RR": [["stance", 1.0, 0.0, 0.0, 0.0], ["phase", 299.0, 0.67, 0.7, 0.8]],
        },
    }

    # The panel front is x=1.02 m. The cap moves along +X through 20 mm.
    button_root_path = "/World/envs/env_0/Button"
    button_cap_rest_pos = (0.99, 0.142, 0.30)
    button_panel_center = (1.06, 0.0, 0.50)
    button_stroke_m = 0.02
    button_press_threshold_m = 0.012
    button_release_threshold_m = 0.002
    button_hold_steps = 5
    button_contact_guard_m = 0.002
    button_indicator_pos = (0.99, 0.142, 0.46)
    ee_default_command = (0.1934, 0.142, 0.05)


class ButtonQPEnv(QPEnv):
    """QP RAMBO environment with a resettable physical USD spring button."""

    cfg: ButtonQPEnvCfg

    def _setup_task_assets(self) -> None:
        if self.scene.cfg.num_envs != 1:
            raise ValueError("The physical Button task supports exactly one environment")
        # Adding an articulation before cloning can disturb Go2's render
        # binding. The native USD button is intentionally authored afterward.
        return None

    def _setup_post_clone_task_assets(self) -> None:
        root_path = self.cfg.button_root_path
        panel_path = f"{root_path}/Panel"
        cap_path = f"{root_path}/Cap"
        joint_path = f"{root_path}/SlideJoint"

        panel_cfg = sim_utils.CuboidCfg(
            size=(0.08, 1.20, 1.00),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        panel_cfg.func(panel_path, panel_cfg, translation=self.cfg.button_panel_center)
        cap_cfg = sim_utils.CuboidCfg(
            size=(0.02, 0.12, 0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=2.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.80, 0.04, 0.04), emissive_color=(0.08, 0.0, 0.0)
            ),
        )
        cap_cfg.func(cap_path, cap_cfg, translation=self.cfg.button_cap_rest_pos)

        stage = omni.usd.get_context().get_stage()
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("X")
        joint.CreateLowerLimitAttr().Set(0.0)
        joint.CreateUpperLimitAttr().Set(self.cfg.button_stroke_m)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(cap_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*self.cfg.button_cap_rest_pos))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        sim_utils.modify_joint_drive_properties(
            joint_path,
            sim_utils.JointDrivePropertiesCfg(
                drive_type="force",
                max_effort=40.0,
                max_velocity=0.5,
                stiffness=800.0,
                damping=25.0,
            ),
            stage=stage,
        )

        self._button_cap = RigidObject(
            RigidObjectCfg(
                prim_path=cap_path,
                init_state=RigidObjectCfg.InitialStateCfg(pos=self.cfg.button_cap_rest_pos),
            )
        )
        self.scene.rigid_objects["button_cap"] = self._button_cap
        indicator_path = "/Visuals/RAMBOButtonStatus"
        indicator_cfg = sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
        indicator_cfg.func(
            indicator_path,
            indicator_cfg,
            translation=self.cfg.button_indicator_pos,
        )
        self._button_status_marker = UsdGeom.Imageable(stage.GetPrimAtPath(indicator_path))

    def __init__(self, cfg: ButtonQPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._button_rest_root_state = self._button_cap.data.default_root_state.clone()
        self._button_pressed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._button_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._update_status_marker()

    def step(self, action: torch.Tensor):
        result = super().step(action)
        self._update_button_state()
        return result

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        super()._reset_idx(env_ids)
        # DirectRLEnv may reset while ButtonQPEnv.__init__ is still inside its
        # parent constructor. The subsequent public reset initializes telemetry.
        if not hasattr(self, "_button_rest_root_state"):
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        self._button_cap.reset(env_ids)
        self._button_cap.write_root_state_to_sim(self._button_rest_root_state[env_ids], env_ids=env_ids)
        self._velocity_commands[env_ids] = 0.0
        self._ee_pos_commands[env_ids] = torch.as_tensor(
            self.cfg.ee_default_command, device=self.device, dtype=self._ee_pos_commands.dtype
        )
        self._ee_force_commands[env_ids] = 0.0
        self.desired_pos[env_ids] = self.default_joint_pos[env_ids]
        self.desired_vel[env_ids] = 0.0
        self.desired_tor[env_ids] = 0.0
        self._last_action[env_ids] = 0.0
        self._button_pressed_steps[env_ids] = 0
        self._button_success[env_ids] = False
        if hasattr(self, "_button_status_marker"):
            self._update_status_marker()

    @property
    def button_displacement(self) -> torch.Tensor:
        displacement = self._button_cap.data.root_pos_w[:, 0] - self._button_rest_root_state[:, 0]
        return displacement.clamp(0.0, self.cfg.button_stroke_m)

    @property
    def button_success(self) -> torch.Tensor:
        return self._button_success.clone()

    @property
    def button_released(self) -> torch.Tensor:
        return self.button_displacement <= self.cfg.button_release_threshold_m

    @property
    def manipulator_ready(self) -> torch.Tensor:
        return self.contact_generator.desired_contact_mode[:, 0] <= -0.5

    def clear_button_success(self) -> torch.Tensor:
        cleared = torch.logical_and(self._button_success, self.button_released)
        self._button_success[cleared] = False
        self._button_pressed_steps[cleared] = 0
        self._update_status_marker()
        return cleared

    def _update_button_state(self) -> None:
        pressed = self.button_displacement >= self.cfg.button_press_threshold_m
        self._button_pressed_steps = torch.where(
            pressed,
            self._button_pressed_steps + 1,
            torch.zeros_like(self._button_pressed_steps),
        )
        self._button_success |= self._button_pressed_steps >= self.cfg.button_hold_steps
        self._update_status_marker()
        self.extras.setdefault("log", {})["Button/displacement_m"] = self.button_displacement.mean()
        self.extras["log"]["Button/success"] = self._button_success.float().mean()

    def _update_status_marker(self) -> None:
        visible = bool(torch.any(self._button_success))
        if visible:
            self._button_status_marker.MakeVisible()
        else:
            self._button_status_marker.MakeInvisible()
