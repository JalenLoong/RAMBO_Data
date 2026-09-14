"""Procedural PhysX loco-manipulation tasks for LingBot-RAMBO Dataset V1.

The released 405D/18D RAMBO checkpoint remains untouched.  These tasks only
add rigid scene objects and telemetry; all robot control still enters through
the public 9D high-level command interface inherited from :class:`QPEnv`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from rambo.assets import spawn_lift_basket_asset
from rambo.utils.physx import assert_physx_environment

from .button_env import ButtonQPEnvCfg
from .qp_env import QPEnv


@configclass
class ObjectTaskQPEnvCfg(ButtonQPEnvCfg):
    """Shared deterministic FL-manipulation configuration."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0, replicate_physics=True)
    task_kind = "lift_basket"
    primary_position = (0.75, 0.15, 0.11)
    basket_center = (0.48, 0.14, 0.11)
    table_center = (0.90, 0.0, 0.45)
    goal_center = (1.55, 0.14, 0.28)
    ball_radius_m = 0.10
    ee_default_command = (0.1934, 0.142, 0.05)


@configclass
class LiftBasketQPEnvCfg(ObjectTaskQPEnvCfg):
    task_kind = "lift_basket"
    # Opt-in UI/preview layout; legacy collection keeps its original cameras.
    lift_camera_setup = "legacy"
    # The source USD rigid-body origin is on the physical bottom surface.
    primary_position = (0.75, 0.15, 0.02)
    basket_clearance_m = 0.06
    basket_hold_steps = 25
    basket_max_tilt_deg = 35.0


@configclass
class PullObjectQPEnvCfg(ObjectTaskQPEnvCfg):
    task_kind = "pull_object_into_basket"
    # object center on the fixed tabletop, z is recomputed when randomized.
    primary_position = (0.84, 0.14, 0.51)
    basket_center = (0.48, 0.14, 0.11)
    table_center = (0.90, 0.0, 0.45)
    object_half_extent_m = 0.035
    containment_hold_steps = 25
    settle_speed_m_s = 0.15


@configclass
class ShootBallQPEnvCfg(ObjectTaskQPEnvCfg):
    task_kind = "shoot_ball_into_goal"
    primary_position = (0.73, 0.14, 0.10)
    goal_center = (1.55, 0.14, 0.28)
    ball_radius_m = 0.10
    goal_hold_steps = 10


def _rigid_cfg(*, size: tuple[float, float, float], color: tuple[float, float, float], mass: float, gravity: bool = True):
    return sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=not gravity,
            # Tables, goals, and the receiving basket are scene geometry, not
            # extremely-heavy dynamic props.  Making them kinematic removes a
            # source of seed-dependent impulses while retaining PhysX contact
            # collisions for the robot and manipulated object.
            kinematic_enabled=not gravity,
            max_depenetration_velocity=3.0,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


class ObjectTaskQPEnv(QPEnv):
    """One of lift-basket, pull-object-into-basket, or shoot-ball tasks."""

    cfg: ObjectTaskQPEnvCfg

    def _setup_task_assets(self) -> None:
        if self.scene.cfg.num_envs != 1:
            raise ValueError("LingBot object tasks support exactly one environment")

    def _resolve_loco_manip_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor:
        """Normalize public command target indices without using QP internals."""

        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        value = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if value.ndim != 1 or value.numel() != torch.unique(value).numel():
            raise ValueError("env_ids must be a one-dimensional unique index vector")
        if bool(torch.any(value < 0)) or bool(torch.any(value >= self.num_envs)):
            raise IndexError("env_ids are outside the active environment range")
        return value

    def set_loco_manip_commands(
        self,
        base_velocity: torch.Tensor,
        fl_position: torch.Tensor,
        fl_force: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set the unchanged public 9D command interface for one or more envs."""

        indices = self._resolve_loco_manip_env_ids(env_ids)
        expected = (indices.numel(), 3)
        for name, command, storage in (
            ("base_velocity", base_velocity, self._velocity_commands),
            ("fl_position", fl_position, self._ee_pos_commands),
            ("fl_force", fl_force, self._ee_force_commands),
        ):
            if not isinstance(command, torch.Tensor) or tuple(command.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if command.device != storage.device or command.dtype != storage.dtype:
                raise ValueError(f"{name} must use {storage.dtype} on {storage.device}")
            if not bool(torch.isfinite(command).all()):
                raise ValueError(f"{name} contains NaN or Inf")
            storage.index_copy_(0, indices, command)

    @property
    def manipulator_ready(self) -> torch.Tensor:
        """Report when the FL contact schedule has entered swing/manipulation."""

        return self.contact_generator.desired_contact_mode[:, 0] <= -0.5

    def _setup_post_clone_task_assets(self) -> None:
        self._root = "/World/envs/env_0/LingBotTask"
        kind = self.cfg.task_kind
        if kind == "lift_basket":
            self._spawn_lift_basket()
        elif kind == "pull_object_into_basket":
            self._spawn_pull_task()
        elif kind == "shoot_ball_into_goal":
            self._spawn_shoot_task()
        else:  # pragma: no cover - configuration error
            raise ValueError(f"unsupported task kind: {kind}")
        if getattr(self.cfg, "lift_camera_setup", "legacy") in ("robot-dual-v1", "robot-dual-v2", "robot-dual-v3"):
            from rambo.tasks.common.lift_camera_rig import spawn
            spawn(self)

    def _add_rigid(self, name: str, cfg: object, position: tuple[float, float, float]) -> RigidObject:
        path = f"{self._root}/{name}"
        cfg.func(path, cfg, translation=position)
        asset = RigidObject(RigidObjectCfg(prim_path=path, init_state=RigidObjectCfg.InitialStateCfg(pos=position)))
        self.scene.rigid_objects[name] = asset
        return asset

    def _spawn_lift_basket(self) -> None:
        center = tuple(self.cfg.primary_position)
        self._primary = spawn_lift_basket_asset(f"{self._root}/basket", center)
        self.scene.rigid_objects["basket"] = self._primary

    def _spawn_open_basket(self, center: tuple[float, float, float]) -> None:
        # Fixed compound basket: floor plus three walls, opening faces robot (-X).
        floor = _rigid_cfg(size=(0.30, 0.26, 0.03), color=(0.08, 0.28, 0.24), mass=100.0, gravity=False)
        self._add_rigid("receiver_floor", floor, center)
        for name, size, offset in (
            ("receiver_back", (0.03, 0.26, 0.20), (0.135, 0.0, 0.10)),
            ("receiver_left", (0.30, 0.03, 0.20), (0.0, 0.115, 0.10)),
            ("receiver_right", (0.30, 0.03, 0.20), (0.0, -0.115, 0.10)),
        ):
            position = tuple(center[index] + offset[index] for index in range(3))
            self._add_rigid(name, _rigid_cfg(size=size, color=(0.08, 0.28, 0.24), mass=100.0, gravity=False), position)

    def _spawn_pull_task(self) -> None:
        table = tuple(self.cfg.table_center)
        self._add_rigid("table_top", _rigid_cfg(size=(0.62, 0.52, 0.04), color=(0.88, 0.88, 0.88), mass=100.0, gravity=False), table)
        for sx in (-0.25, 0.25):
            for sy in (-0.20, 0.20):
                self._add_rigid(
                    f"table_leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                    _rigid_cfg(size=(0.04, 0.04, 0.42), color=(0.35, 0.35, 0.35), mass=100.0, gravity=False),
                    (table[0] + sx, table[1] + sy, 0.21),
                )
        self._spawn_open_basket(tuple(self.cfg.basket_center))
        self._primary = self._add_rigid(
            "table_object",
            _rigid_cfg(size=(0.07, 0.07, 0.07), color=(0.75, 0.10, 0.12), mass=0.08),
            tuple(self.cfg.primary_position),
        )

    def _spawn_shoot_task(self) -> None:
        center = tuple(self.cfg.goal_center)
        # Frame and collision backstop form a physical catch volume behind x=goal_center.x.
        for name, size, offset in (
            ("goal_left", (0.04, 0.04, 0.56), (0.0, 0.33, 0.0)),
            ("goal_right", (0.04, 0.04, 0.56), (0.0, -0.33, 0.0)),
            ("goal_crossbar", (0.04, 0.70, 0.04), (0.0, 0.0, 0.28)),
            ("goal_back", (0.04, 0.70, 0.56), (0.30, 0.0, 0.0)),
        ):
            self._add_rigid(
                name,
                _rigid_cfg(size=size, color=(0.72, 0.22, 0.36), mass=100.0, gravity=False),
                tuple(center[index] + offset[index] for index in range(3)),
            )
        ball_cfg = sim_utils.SphereCfg(
            radius=self.cfg.ball_radius_m,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=12, solver_velocity_iteration_count=2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.78, 0.80)),
        )
        self._primary = self._add_rigid("ball", ball_cfg, tuple(self.cfg.primary_position))

    def __init__(self, cfg: ObjectTaskQPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._physx_evidence = assert_physx_environment(self)
        # Capture the initialized actor-frame pose reported by PhysX. Imported
        # USDs may have a computed actor-frame offset from their authored Xform
        # origin, so their config-space default pose is not necessarily the
        # runtime root-link pose used by reset and task displacement metrics.
        self._primary_rest_pose = self.primary_pose_w.clone()
        self._primary_rest_vel = self._primary.data.default_root_vel.torch.clone()
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fl_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._left_table_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def primary_pose_w(self) -> torch.Tensor:
        return torch.cat(
            (
                self._primary.data.root_link_pos_w.torch,
                self._primary.data.root_link_quat_w.torch,
            ),
            dim=-1,
        ).clone()

    @property
    def primary_velocity_w(self) -> torch.Tensor:
        return torch.cat(
            (
                self._primary.data.root_link_lin_vel_w.torch,
                self._primary.data.root_link_ang_vel_w.torch,
            ),
            dim=-1,
        ).clone()

    @property
    def task_success(self) -> torch.Tensor:
        return self._success.clone()

    @property
    def fl_contact_seen(self) -> torch.Tensor:
        return self._fl_contact_seen.clone()

    @property
    def task_metrics(self) -> dict[str, torch.Tensor]:
        pos = self.primary_pose_w[:, :3]
        vel = self.primary_velocity_w[:, :3]
        foot = self._robot.data.body_link_pos_w.torch[:, self.feet_ids[0], :]
        distance = torch.linalg.vector_norm(pos - foot, dim=-1)
        if self.cfg.task_kind == "lift_basket":
            # PhysX reports the source actor at its computed rigid-body frame,
            # not at the USD's bottom-surface Xform origin. Measure vertical
            # lift relative to the reset pose and retain the configured 20-mm
            # initial bottom clearance.
            clearance = (
                float(self.cfg.primary_position[2])
                + pos[:, 2]
                - self._primary_rest_pose[:, 2]
            )
            quat = self.primary_pose_w[:, 3:7]
            # Stage up is Y, but the source root authors an X rotation: the
            # rigid-local physical up axis is -Z. Use -R[2, 2] for XYZW.
            local_up_world_z = 2.0 * (quat[:, 0].square() + quat[:, 1].square()) - 1.0
            tilt = torch.acos(torch.clamp(local_up_world_z, -1.0, 1.0))
            return {"clearance_m": clearance, "tilt_rad": tilt, "fl_distance_m": distance, "hold_steps": self._hold.float()}
        if self.cfg.task_kind == "pull_object_into_basket":
            basket = torch.as_tensor(self.cfg.basket_center, device=self.device, dtype=pos.dtype)
            inside = (torch.abs(pos[:, 0] - basket[0]) <= 0.105) & (torch.abs(pos[:, 1] - basket[1]) <= 0.085) & (pos[:, 2] <= basket[2] + 0.16)
            speed = torch.linalg.vector_norm(vel, dim=-1)
            left_table = pos[:, 2] < self.cfg.table_center[2] + 0.01
            return {"inside_basket": inside.float(), "speed_m_s": speed, "left_table": left_table.float(), "fl_distance_m": distance, "hold_steps": self._hold.float()}
        goal = torch.as_tensor(self.cfg.goal_center, device=self.device, dtype=pos.dtype)
        r = float(self.cfg.ball_radius_m)
        scored = (pos[:, 0] >= goal[0] + r) & (torch.abs(pos[:, 1] - goal[1]) <= 0.33 - r) & (pos[:, 2] >= r) & (pos[:, 2] <= goal[2] + 0.28 - r)
        return {"scored": scored.float(), "behind_goal_m": pos[:, 0] - goal[0], "fl_distance_m": distance, "hold_steps": self._hold.float()}

    def step(self, action: torch.Tensor):
        result = super().step(action)
        self._update_task_state()
        return result

    def _update_task_state(self) -> None:
        metrics = self.task_metrics
        # Proximity plus nonzero robot contact force is evidence of a real FL manipulation event.
        # The public sensor records a short force history.  A peak over that
        # history avoids aliasing a genuine foot/object impact between policy
        # samples and does not depend on private PhysX contact-pair APIs.
        history = self._contact_sensor.data.net_forces_w_history.torch
        foot_force = torch.linalg.vector_norm(
            history[:, :, self._contact_feet_ids[0], :], dim=-1
        ).amax(dim=1)
        interaction_radius = 0.30 if self.cfg.task_kind == "lift_basket" else 0.17
        self._fl_contact_seen |= (metrics["fl_distance_m"] <= interaction_radius) & (foot_force >= 1.0)
        if self.cfg.task_kind == "lift_basket":
            valid = (metrics["clearance_m"] >= self.cfg.basket_clearance_m) & (metrics["tilt_rad"] <= np.deg2rad(self.cfg.basket_max_tilt_deg)) & self._fl_contact_seen
            need = self.cfg.basket_hold_steps
        elif self.cfg.task_kind == "pull_object_into_basket":
            self._left_table_seen |= metrics["left_table"].bool()
            valid = metrics["inside_basket"].bool() & (metrics["speed_m_s"] <= self.cfg.settle_speed_m_s) & self._fl_contact_seen & self._left_table_seen
            need = self.cfg.containment_hold_steps
        else:
            valid = metrics["scored"].bool() & self._fl_contact_seen
            need = self.cfg.goal_hold_steps
        self._hold = torch.where(valid, self._hold + 1, torch.zeros_like(self._hold))
        self._success |= self._hold >= int(need)
        self.extras.setdefault("log", {})["LingBot/task_success"] = self._success.float().mean()

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_primary_rest_pose"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        sim_env_ids = env_ids.to(dtype=torch.int32)
        self._primary.reset(env_ids=env_ids)
        self._primary.write_root_pose_to_sim_index(
            root_pose=self._primary_rest_pose.index_select(0, env_ids), env_ids=sim_env_ids
        )
        self._primary.write_root_velocity_to_sim_index(
            root_velocity=self._primary_rest_vel.index_select(0, env_ids), env_ids=sim_env_ids
        )
        self._success[env_ids] = False
        self._hold[env_ids] = 0
        self._fl_contact_seen[env_ids] = False
        self._left_table_seen[env_ids] = False
        default = torch.as_tensor(self.cfg.ee_default_command, dtype=self._ee_pos_commands.dtype, device=self.device).expand(env_ids.numel(), -1)
        zeros = torch.zeros((env_ids.numel(), 3), dtype=self._velocity_commands.dtype, device=self.device)
        self.set_loco_manip_commands(zeros, default, zeros, env_ids=env_ids)
