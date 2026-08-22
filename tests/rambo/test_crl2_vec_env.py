"""Pure-Python behavior tests for the RAMBO CRL2 vector-environment adapter."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

gym = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from rambo.rl.crl2_vec_env import Crl2VecEnvWrapper  # noqa: E402


def _install_fake_isaaclab_env_types(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Provide only the dynamic type check used by ``Crl2VecEnvWrapper``."""

    class DirectRLEnv:
        pass

    class ManagerBasedRLEnv:
        pass

    isaaclab_module = types.ModuleType("isaaclab")
    envs_module = types.ModuleType("isaaclab.envs")
    envs_module.DirectRLEnv = DirectRLEnv
    envs_module.ManagerBasedRLEnv = ManagerBasedRLEnv
    isaaclab_module.envs = envs_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs_module)
    return DirectRLEnv, ManagerBasedRLEnv


def test_wrapper_flattens_single_policy_space_without_losing_tensor_tuple_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DirectRLEnv, _ = _install_fake_isaaclab_env_types(monkeypatch)

    class FakeBaseEnv(DirectRLEnv):
        def __init__(self) -> None:
            self.num_envs = 2
            self.device = torch.device("cpu")
            self.max_episode_length = 100
            self.cfg = SimpleNamespace(is_finite_horizon=False)
            self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
            self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=float)
            self.single_observation_space = gym.spaces.Dict(
                {
                    "policy": gym.spaces.Box(-1.0, 1.0, shape=(5, 3), dtype=float),
                    "critic": gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=float),
                }
            )

        def _get_observations(self):
            return {"policy": torch.full((self.num_envs, 5, 3), 0.25)}

        def seed(self, seed: int) -> int:
            return seed

    class FakeGymEnv:
        def __init__(self) -> None:
            self.unwrapped = FakeBaseEnv()
            self.render_mode = None
            self.observation_space = self.unwrapped.single_observation_space
            self.action_space = self.unwrapped.single_action_space
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            return self.unwrapped._get_observations(), {}

        def step(self, actions: torch.Tensor):
            observations = self.unwrapped._get_observations()
            rewards = torch.tensor([1.0, 2.0])
            terminated = torch.tensor([False, True])
            truncated = torch.tensor([True, False])
            return observations, rewards, terminated, truncated, {}

        def close(self) -> None:
            return None

    env = FakeGymEnv()
    wrapper = Crl2VecEnvWrapper(env)

    assert env.reset_count == 1
    assert wrapper.num_actions == 6
    assert wrapper.num_obs == 15

    observations, rewards, dones, extras = wrapper.step(torch.zeros((2, 6)))
    assert observations.shape == (2, 5, 3)
    assert rewards.tolist() == [1.0, 2.0]
    assert dones.dtype == torch.long
    assert dones.tolist() == [1, 1]
    assert torch.equal(extras["time_outs"], torch.tensor([True, False]))
    assert torch.equal(extras["observations"]["policy"], observations)


def test_wrapper_reads_cached_direct_env_observations_without_advancing_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    DirectRLEnv, _ = _install_fake_isaaclab_env_types(monkeypatch)

    class FakeBaseEnv(DirectRLEnv):
        def __init__(self) -> None:
            self.num_envs = 1
            self.device = torch.device("cpu")
            self.max_episode_length = 100
            self.cfg = SimpleNamespace(is_finite_horizon=True)
            self.single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
            self.single_observation_space = gym.spaces.Dict(
                {"policy": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=float)}
            )
            self.reset_observations = {"policy": torch.tensor([[1.0, 2.0, 3.0]])}
            self.observation_recomputes = 0

        def _get_observations(self):
            self.observation_recomputes += 1
            return {"policy": torch.tensor([[9.0, 9.0, 9.0]])}

    class FakeGymEnv:
        def __init__(self) -> None:
            self.unwrapped = FakeBaseEnv()
            self.render_mode = None
            self.observation_space = self.unwrapped.single_observation_space
            self.action_space = self.unwrapped.single_action_space

        def reset(self):
            return self.unwrapped.reset_observations, {}

    wrapper = Crl2VecEnvWrapper(FakeGymEnv())
    observations, extras = wrapper.get_observations()

    assert torch.equal(observations, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(extras["observations"]["policy"], observations)
    assert wrapper.unwrapped.observation_recomputes == 0
