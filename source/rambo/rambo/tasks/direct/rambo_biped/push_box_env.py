"""Independent, single-environment biped two-front-foot box pushing scene."""
from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .qp_env import QPEnv, QPEnvCfg

TASK_ID = "Isaac-RAMBO-Biped-Push-Box-Go2-v0"


@configclass
class PushBoxCfg(QPEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10., replicate_physics=True)
    box_size = (0.50, 0.36, 0.50)
    box_position = (0.75, 0., 0.255)
    box_mass = 1.0
    box_friction = 0.25
    box_color = (0.035, 0.25, 0.65)
    wall_position = (1.40, 0., 0.50)
    wall_size = (0.10, 2., 1.)


class PushBoxEnv(QPEnv):
    cfg: PushBoxCfg

    def _setup_scene(self):
        if self.num_envs != 1:
            raise ValueError("Push-box collection currently supports exactly one environment")
        super()._setup_scene()
        box_path = "/World/envs/env_0/PushBox"
        wall_path = "/World/envs/env_0/PushWall"
        material = sim_utils.RigidBodyMaterialCfg(static_friction=self.cfg.box_friction,
            dynamic_friction=self.cfg.box_friction, restitution=0., friction_combine_mode="min")
        box = sim_utils.CuboidCfg(size=self.cfg.box_size, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.box_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.003, rest_offset=0.),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.box_color, roughness=.85))
        self.box = RigidObject(RigidObjectCfg(prim_path=box_path, spawn=box,
            init_state=RigidObjectCfg.InitialStateCfg(pos=self.cfg.box_position)))
        self.scene.rigid_objects["push_box"] = self.box
        wall = sim_utils.CuboidCfg(size=self.cfg.wall_size, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.003, rest_offset=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.55,.56,.59), roughness=.95))
        self.wall = RigidObject(RigidObjectCfg(prim_path=wall_path, spawn=wall,
            init_state=RigidObjectCfg.InitialStateCfg(pos=self.cfg.wall_position)))
        self.scene.rigid_objects["push_wall"] = self.wall
        self.push_contacts = {}
        for name, path, partner in (("fl_box", "/World/envs/env_0/Robot/FL_foot", box_path),
                                    ("fr_box", "/World/envs/env_0/Robot/FR_foot", box_path),
                                    ("box_wall", box_path, wall_path)):
            sensor = ContactSensor(ContactSensorCfg(prim_path=path, filter_prim_paths_expr=[partner],
                update_period=0., history_length=5, max_contact_data_count_per_prim=64))
            self.push_contacts[name] = sensor
            self.scene.sensors["push_"+name] = sensor

    def set_biped_commands(self, command: torch.Tensor):
        if command.shape != (1,15) or command.dtype != torch.float32 or command.device != self._velocity_commands.device:
            raise ValueError("Native biped command must be float32 [1,15] on the environment device")
        if not bool(torch.isfinite(command).all()):
            raise ValueError("Nonfinite biped command")
        for i, name in enumerate(("_velocity_commands", "_ee_pos_fl_commands", "_ee_pos_fr_commands",
                                  "_ee_force_fl_commands", "_ee_force_fr_commands")):
            getattr(self, name).copy_(command[:,3*i:3*i+3])

    def push_pair_forces(self):
        # Each sensor has a single sensing body and single partner: actual
        # object-pair normal forces, not proximity or all-contact surrogates.
        result = {}
        for name, sensor in self.push_contacts.items():
            forces = sensor.data.force_matrix_w_history.torch
            result[name] = forces.reshape(1,-1,3).norm(dim=-1).amax(dim=-1)
        return result

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "box"):
            return
        indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int32) if env_ids is None else env_ids.to(torch.int32)
        for asset in (self.box, self.wall):
            asset.write_root_pose_to_sim_index(root_pose=asset.data.default_root_state.torch[indices.long(),:7], env_ids=indices)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(indices),6),device=self.device), env_ids=indices)
            asset.reset(indices.long())
