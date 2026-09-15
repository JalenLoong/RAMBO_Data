"""Shared Go2 delayed-actuator contract for both RAMBO control modes."""

from __future__ import annotations

from .delayed_dc_motor_cfg import DelayedDCMotorCfg


GO2_ACTUATOR_DELAY_STEPS = 10
"""Maximum command delay in physics steps for the original RAMBO policies."""


def make_go2_delayed_dc_motor_cfgs() -> dict[str, DelayedDCMotorCfg]:
    """Return fresh calibrated Go2 motor configurations.

    The quadruped policies use the same Go2 hardware contract.
    Keeping this factory centralized prevents a seemingly harmless task-local
    edit from changing torque-speed saturation or delay behavior for only one
    checkpoint.
    """

    return {
        "calf": DelayedDCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=40.887,
            saturation_effort=40.887,
            velocity_limit=15.70,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
            min_delay=0,
            max_delay=GO2_ACTUATOR_DELAY_STEPS,
        ),
        "hip_thigh": DelayedDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=21.33,
            saturation_effort=21.33,
            velocity_limit=30.1,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
            min_delay=0,
            max_delay=GO2_ACTUATOR_DELAY_STEPS,
        ),
    }


__all__ = ("GO2_ACTUATOR_DELAY_STEPS", "make_go2_delayed_dc_motor_cfgs")
