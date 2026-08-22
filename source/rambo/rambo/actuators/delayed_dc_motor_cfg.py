"""Configuration for RAMBO's delayed DC motor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._runtime import kit_runtime_ready


if kit_runtime_ready():
    # See ``rambo._runtime``: importing installed Isaac Lab before Kit exists
    # starts the simulator and is intentionally avoided for static tooling.
    from isaaclab.actuators import DCMotorCfg
    from isaaclab.utils import configclass
else:  # pragma: no cover - static-tooling fallback.

    @dataclass
    class DCMotorCfg:  # type: ignore[no-redef]
        class_type: type | None = None
        joint_names_expr: list[str] = field(default_factory=list)
        effort_limit: float | dict[str, float] | None = None
        saturation_effort: float | None = None
        velocity_limit: float | dict[str, float] | None = None
        stiffness: float | dict[str, float] | None = None
        damping: float | dict[str, float] | None = None
        friction: float | dict[str, float] | None = None
        dynamic_friction: float | dict[str, float] | None = None
        viscous_friction: float | dict[str, float] | None = None
        effort_limit_sim: float | dict[str, float] | None = None
        velocity_limit_sim: float | dict[str, float] | None = None

    def configclass(cls: type[Any]) -> type[Any]:  # type: ignore[no-redef]
        return dataclass(cls)

from .delayed_dc_motor import DelayedDCMotor


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """DC motor configuration with a command delay sampled at reset."""

    class_type: type = DelayedDCMotor
    min_delay: int = 0
    """Minimum command delay in physics time-steps."""

    max_delay: int = 0
    """Maximum command delay in physics time-steps."""

    delay: int | None = None
    """Fixed delay, or ``None`` to sample uniformly from the configured range."""
