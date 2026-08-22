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
