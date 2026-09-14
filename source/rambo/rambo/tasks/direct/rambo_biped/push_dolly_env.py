"""Go2 biped with the independently authored, unpowered NVIDIA Dolly."""
import os
from pathlib import Path
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .qp_env import QPEnv, QPEnvCfg

TASK_ID="Isaac-RAMBO-Biped-Push-Dolly-Go2-v0"


def asset_path():
    workspace=Path(os.environ.get("WORKSPACE_ROOT",Path(__file__).resolve().parents[8]))
    return Path(os.environ.get("RAMBO_DOLLY_USD",workspace/"assets/dolly/variants/dolly_passive_v1.usda"))


@configclass
class PushDollyCfg(QPEnvCfg):
    scene: InteractiveSceneCfg=InteractiveSceneCfg(num_envs=1,env_spacing=10.,replicate_physics=True)
    dolly_position=(.95,0.,.34589605)


class PushDollyEnv(QPEnv):
    cfg: PushDollyCfg

    def _setup_scene(self):
        if self.num_envs!=1:raise ValueError("Dolly teleoperation supports one environment")
        super()._setup_scene()
        path=asset_path()
        if not path.is_file():raise FileNotFoundError(f"Dolly physics asset missing: {path}")
        self.dolly=Articulation(ArticulationCfg(prim_path="/World/envs/env_0/Dolly",
            spawn=sim_utils.UsdFileCfg(usd_path=str(path),activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                    solver_position_iteration_count=16,solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=self.cfg.dolly_position),actuators={}))
        self.scene.articulations["dolly"]=self.dolly

    def set_biped_commands(self,command):
        if command.shape!=(1,15) or command.dtype!=torch.float32 or command.device!=self._velocity_commands.device:
            raise ValueError("Expected float32 [1,15] biped command on environment device")
        if not bool(torch.isfinite(command).all()):raise ValueError("Nonfinite command")
        for i,name in enumerate(("_velocity_commands","_ee_pos_fl_commands","_ee_pos_fr_commands","_ee_force_fl_commands","_ee_force_fr_commands")):
            getattr(self,name).copy_(command[:,3*i:3*i+3])

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self,"dolly"):return
        ids=torch.arange(self.num_envs,device=self.device,dtype=torch.int32) if env_ids is None else env_ids.to(torch.int32)
        a=self.dolly
        a.write_root_pose_to_sim_index(root_pose=a.data.default_root_pose.torch[ids.long()],env_ids=ids)
        a.write_root_velocity_to_sim_index(root_velocity=a.data.default_root_vel.torch[ids.long()],env_ids=ids)
        a.write_joint_state_to_sim_index(position=a.data.default_joint_pos.torch[ids.long()],velocity=a.data.default_joint_vel.torch[ids.long()],env_ids=ids)
        a.reset(ids.long())
