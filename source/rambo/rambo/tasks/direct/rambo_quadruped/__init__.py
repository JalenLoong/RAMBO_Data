"""RAMBO quadruped task registration."""

from __future__ import annotations

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-RAMBO-Quadruped-Go2-v0",
    entry_point=f"{__name__}.qp_env:QPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qp_env:QPEnvCfg",
        "crl2_cfg_entry_point": f"{agents.__name__}:crl2_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-RAMBO-Quadruped-Button-Go2-v0",
    entry_point=f"{__name__}.button_env:ButtonQPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.button_env:ButtonQPEnvCfg",
        "crl2_cfg_entry_point": f"{agents.__name__}:crl2_flat_ppo_cfg.yaml",
    },
)

for _task_id, _cfg_name in (
    ("Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0", "LiftBasketQPEnvCfg"),
    ("Isaac-RAMBO-Quadruped-Pull-Object-Into-Basket-Go2-v0", "PullObjectQPEnvCfg"),
    ("Isaac-RAMBO-Quadruped-Shoot-Ball-Into-Goal-Go2-v0", "ShootBallQPEnvCfg"),
):
    gym.register(
        id=_task_id,
        entry_point=f"{__name__}.object_tasks_env:ObjectTaskQPEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.object_tasks_env:{_cfg_name}",
            "crl2_cfg_entry_point": f"{agents.__name__}:crl2_flat_ppo_cfg.yaml",
        },
    )

gym.register(
    id="Isaac-RAMBO-Quadruped-Push-Box-V2-Go2-v0",
    entry_point=f"{__name__}.push_box_v2:PushBoxV2Env",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.push_box_v2:PushBoxV2Cfg",
            "crl2_cfg_entry_point": f"{agents.__name__}:crl2_flat_ppo_cfg.yaml"},
)
