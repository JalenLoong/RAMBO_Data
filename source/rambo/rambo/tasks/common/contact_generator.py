"""Contact schedule generator shared by RAMBO quadruped and biped tasks."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class ContactGenerator:
    """Evaluate RAMBO's stance, swing, and phase contact schedule.

    Visualization hooks intentionally remain no-ops in the external extension;
    simulation behavior and all public schedule tensors are preserved.
    """

    _MODE_VALUES = {"stance": 1.0, "phase": 0.0, "swing": -1.0}

    def __init__(self, env: Any):
        self._env = env
        sequence = env.cfg.contact_generator_config["contact_sequence"]
        modes: list[list[float]] = []
        timings: list[list[float]] = []
        extras: list[list[list[float]]] = []
        for foot_name, phases in sequence.items():
            del foot_name
            if not phases:
                raise ValueError("Every RAMBO contact sequence must have at least one phase.")
            modes.append([])
            timings.append([])
            extras.append([])
            for phase in phases:
                try:
                    modes[-1].append(self._MODE_VALUES[phase[0]])
                except KeyError as error:
                    raise ValueError(f"Invalid contact mode: {phase[0]!r}") from error
                timings[-1].append(float(phase[1]))
                extras[-1].append([float(value) for value in phase[2:]])

        self._contact_mode = torch.tensor(modes, dtype=torch.float32, device=env.device)
        self._contact_timing = torch.tensor(timings, dtype=torch.float32, device=env.device)
        self._contact_extra = torch.tensor(extras, dtype=torch.float32, device=env.device)
        if self._contact_mode.shape[0] != env.num_feet:
            raise ValueError(
                f"Contact schedule describes {self._contact_mode.shape[0]} feet, expected {env.num_feet}."
            )
        if self._contact_extra.shape[-1] < 3:
            raise ValueError("Every contact phase must provide three extra values for phase behavior.")
        self._contact_timing_cumsum = torch.cumsum(self._contact_timing, dim=1)
        self._contact_max_index = torch.tensor([len(phases) for phases in sequence.values()], device=env.device)

        shape = (env.num_envs, env.num_feet)
        self._current_contact_mode = torch.ones(shape, dtype=torch.float32, device=env.device)
        self._next_contact_mode = torch.ones(shape, dtype=torch.float32, device=env.device)
        self._current_contact_timing = torch.zeros(shape, dtype=torch.float32, device=env.device)
        self._current_contact_extra = torch.zeros((*shape, 3), dtype=torch.float32, device=env.device)
        self._current_contact_phase = torch.zeros(shape, dtype=torch.float32, device=env.device)
        self._current_contact_state = torch.ones(shape, dtype=torch.bool, device=env.device)
        self._until_next_contact_mode = torch.zeros(shape, dtype=torch.float32, device=env.device)
        self._time_since_reset = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self.debug_vis = bool(env.cfg.contact_generator_config.get("contact_generator_debug_vis", False))
        self.contact_generator_vis_handle = None

    def _indices(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None or env_ids == slice(None):
            return torch.arange(self._env.num_envs, device=self._env.device)
        if isinstance(env_ids, slice):
            return torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)

    def _update(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        time_since_reset = self._env.time_since_reset.index_select(0, env_ids)
        self._time_since_reset[env_ids] = time_since_reset
        current_index = torch.sum(
            time_since_reset[:, None, None] >= self._contact_timing_cumsum[None, :, :] + 1.0e-4,
            dim=-1,
        )
        previous_index = current_index - 1
        next_index = torch.minimum(current_index + 1, self._contact_max_index[None, :] - 1)

        for foot_index in range(self._env.num_feet):
            current = current_index[:, foot_index]
            previous = previous_index[:, foot_index]
            following = next_index[:, foot_index]
            self._current_contact_mode[env_ids, foot_index] = self._contact_mode[foot_index].index_select(0, current)
            self._current_contact_timing[env_ids, foot_index] = self._contact_timing[foot_index].index_select(0, current)
            self._current_contact_extra[env_ids, foot_index] = self._contact_extra[foot_index].index_select(0, current)
            self._next_contact_mode[env_ids, foot_index] = self._contact_mode[foot_index].index_select(0, following)
            previous_end = self._contact_timing_cumsum[foot_index].index_select(0, previous.clamp_min(0))
            self._until_next_contact_mode[env_ids, foot_index] = (
                self._contact_timing_cumsum[foot_index].index_select(0, current) - time_since_reset
            )
            elapsed = torch.where(previous < 0, time_since_reset, time_since_reset - previous_end)
            mode = self._current_contact_mode[env_ids, foot_index]
            extra = self._current_contact_extra[env_ids, foot_index]
            phase = self._current_contact_phase[env_ids, foot_index]
            state = self._current_contact_state[env_ids, foot_index]

            stance = mode >= 0.5
            swing = mode <= -0.5
            phase_mode = ~(stance | swing)
            phase_value = torch.remainder(elapsed + extra[:, 0] * extra[:, 1], extra[:, 1]) / extra[:, 1]
            self._current_contact_phase[env_ids, foot_index] = torch.where(
                stance, torch.ones_like(phase), torch.where(swing, torch.zeros_like(phase), phase_value)
            )
            updated_phase = self._current_contact_phase[env_ids, foot_index]
            phase_state = updated_phase < extra[:, 2]
            self._current_contact_state[env_ids, foot_index] = torch.where(
                stance, torch.ones_like(state), torch.where(swing, torch.zeros_like(state), phase_state)
            )

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        self._update(self._indices(env_ids))

    def update(self) -> None:
        self._update(self._indices(None))

    @property
    def desired_contact_state(self) -> torch.Tensor:
        return self._current_contact_state

    @property
    def desired_contact_phase(self) -> torch.Tensor:
        return self._current_contact_phase

    @property
    def desired_contact_mode(self) -> torch.Tensor:
        return self._current_contact_mode

    @property
    def desired_contact_mode_next(self) -> torch.Tensor:
        return self._next_contact_mode

    @property
    def until_next_contact_mode(self) -> torch.Tensor:
        return self._until_next_contact_mode

    @property
    def normalized_phase(self) -> torch.Tensor:
        contact_phase = self._current_contact_phase.clone()
        phase_mode = (self._current_contact_mode < 0.5) & (self._current_contact_mode > -0.5)
        swing_ratio = self._current_contact_extra[:, :, 2]
        before_swing = phase_mode & (contact_phase < swing_ratio)
        after_swing = phase_mode & ~before_swing
        contact_phase = torch.where(before_swing, contact_phase / swing_ratio, contact_phase)
        return torch.where(after_swing, (contact_phase - swing_ratio) / (1.0 - swing_ratio), contact_phase)

    @property
    def stance_duration(self) -> torch.Tensor:
        duration = self._current_contact_timing.clone()
        duration = torch.where(self._current_contact_mode <= -0.5, torch.zeros_like(duration), duration)
        phase_mode = (self._current_contact_mode < 0.5) & (self._current_contact_mode > -0.5)
        return torch.where(
            phase_mode,
            self._current_contact_extra[:, :, 1] * self._current_contact_extra[:, :, 2],
            duration,
        )

    def _set_contact_generator_vis(self, debug_vis: bool) -> None:
        self.debug_vis = debug_vis

    def _set_contact_generator_vis_impl(self, debug_vis: bool) -> None:
        self.debug_vis = debug_vis
