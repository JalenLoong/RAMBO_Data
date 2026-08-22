"""A RAMBO-owned delayed DC motor for Isaac Lab 2.3.2.

Isaac Lab 2.3.2 retains a delayed PD actuator but no longer exposes a
``DelayedDCMotor``.  This module adds only command delay and deliberately
delegates torque-speed saturation to the current upstream :class:`DCMotor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from .._runtime import kit_runtime_ready


if kit_runtime_ready():
    # Importing ``isaaclab`` is safe only after AppLauncher has created Kit.
    from isaaclab.actuators import DCMotor
    from isaaclab.utils import DelayBuffer
    from isaaclab.utils.types import ArticulationActions

    _ISAACLAB_AVAILABLE = True
else:  # pragma: no cover - exercised only outside a running Isaac Lab process.
    DCMotor = object  # type: ignore[assignment,misc]
    DelayBuffer = None  # type: ignore[assignment,misc]
    ArticulationActions = Any  # type: ignore[misc,assignment]
    _ISAACLAB_AVAILABLE = False

if TYPE_CHECKING:
    from .delayed_dc_motor_cfg import DelayedDCMotorCfg


def _batch_count(env_ids: Sequence[int] | torch.Tensor | slice | None, total: int) -> int:
    """Return the number of selected environments without assuming tensor IDs."""

    if env_ids is None:
        return total
    if isinstance(env_ids, slice):
        return len(range(total)[env_ids])
    return int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)


if _ISAACLAB_AVAILABLE:

    class DelayedDCMotor(DCMotor):
        """DC motor with reset-time per-environment command delay.

        The delay applies to position, velocity, and feed-forward effort together,
        preserving RAMBO's legacy command ordering while using Isaac Lab 2.3.2's
        current motor model and buffer implementation.
        """

        cfg: "DelayedDCMotorCfg"

        def __init__(self, cfg: "DelayedDCMotorCfg", *args, **kwargs):
            if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
                raise ValueError(
                    f"Expected 0 <= min_delay <= max_delay, got {cfg.min_delay=} and {cfg.max_delay=}."
                )
            if cfg.delay is not None and not cfg.min_delay <= cfg.delay <= cfg.max_delay:
                raise ValueError(
                    f"Fixed delay {cfg.delay} must be within [{cfg.min_delay}, {cfg.max_delay}]."
                )
            super().__init__(cfg, *args, **kwargs)
            self._position_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
            self._velocity_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
            self._effort_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

        @property
        def time_lags(self) -> torch.Tensor:
            """Current delay in physics steps for every environment."""

            return self._position_delay_buffer.time_lags

        def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None):
            super().reset(env_ids)
            count = _batch_count(env_ids, self._num_envs)
            batch_ids: Sequence[int] | torch.Tensor | slice | None = slice(None) if env_ids is None else env_ids
            if self.cfg.delay is None:
                delays = torch.randint(
                    low=self.cfg.min_delay,
                    high=self.cfg.max_delay + 1,
                    size=(count,),
                    dtype=torch.int,
                    device=self._device,
                )
            else:
                delays = torch.full((count,), self.cfg.delay, dtype=torch.int, device=self._device)

            for buffer in (self._position_delay_buffer, self._velocity_delay_buffer, self._effort_delay_buffer):
                buffer.set_time_lag(delays, batch_ids)
                buffer.reset(batch_ids)

        def compute(
            self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
        ) -> ArticulationActions:
            control_action.joint_positions = self._position_delay_buffer.compute(control_action.joint_positions)
            control_action.joint_velocities = self._velocity_delay_buffer.compute(control_action.joint_velocities)
            control_action.joint_efforts = self._effort_delay_buffer.compute(control_action.joint_efforts)
            return super().compute(control_action, joint_pos, joint_vel)

else:

    class DelayedDCMotor:  # pragma: no cover - defensive fallback for static imports.
        """Placeholder that explains why simulation execution is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("DelayedDCMotor requires an installed Isaac Lab runtime.")
