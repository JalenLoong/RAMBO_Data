"""CRL2 adapter that is independent of the official ``isaaclab_rl`` package."""

from __future__ import annotations

from typing import Any

import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - static-tooling fallback.
    gym = None  # type: ignore[assignment]

try:
    from crl2.env import VecEnv as _VecEnv
except ModuleNotFoundError:  # pragma: no cover - allows static imports before CRL2 installation.
    _VecEnv = object


def _flatdim(space: Any) -> int:
    if gym is not None:
        return int(gym.spaces.flatdim(space))
    shape = getattr(space, "shape", None)
    if shape is None:
        raise RuntimeError("Gymnasium is required to flatten non-Box CRL2 spaces.")
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


class Crl2VecEnvWrapper(_VecEnv):
    """Adapt DirectRL or ManagerBasedRL Isaac Lab environments to CRL2."""

    def __init__(self, env: Any):
        try:
            from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
        except ModuleNotFoundError as error:  # pragma: no cover - static-tooling path.
            raise RuntimeError("Crl2VecEnvWrapper requires an installed Isaac Lab runtime.") from error
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(f"Expected a DirectRLEnv or ManagerBasedRLEnv, received {type(env)}.")
        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = _flatdim(self.unwrapped.single_action_space)
        observation_space = self.unwrapped.single_observation_space
        # Isaac Lab exposes policy observations in a Gymnasium Dict.  CRL2's
        # tensor tuple contract consumes the flattened policy member only.
        if gym is not None and isinstance(observation_space, gym.spaces.Dict):
            observation_space = observation_space["policy"]
        self.num_obs = _flatdim(observation_space)
        self._last_observations: dict[str, torch.Tensor] | None = None
        self.reset()

    def __str__(self) -> str:
        return f"<{type(self).__name__}{self.env}>"

    __repr__ = __str__

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        # DirectRLEnv tasks may build history inside ``_get_observations``.
        # Returning the last buffer is therefore materially different from
        # recomputing it: validation and PPO setup must not advance RAMBO's
        # observation history before the first policy action.
        cached_observations = self._last_observations
        if isinstance(cached_observations, dict) and "policy" in cached_observations:
            observations = cached_observations
        else:
            cached_observations = getattr(self.unwrapped, "obs_buf", None)
            if isinstance(cached_observations, dict) and "policy" in cached_observations:
                observations = cached_observations
            elif hasattr(self.unwrapped, "observation_manager"):
                observations = self.unwrapped.observation_manager.compute()
            else:
                observations = self.unwrapped._get_observations()
        return observations["policy"], {"observations": observations}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        observations, _ = self.env.reset()
        self._last_observations = observations
        return observations["policy"], {"observations": observations}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        self._last_observations = observations
        extras["observations"] = observations
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return observations["policy"], rewards, (terminated | truncated).to(dtype=torch.long), extras

    def close(self) -> None:
        self.env.close()
