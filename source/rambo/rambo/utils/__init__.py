"""Shared RAMBO helpers independent of legacy Isaac Lab patches."""

from .articulation import (
    GO2_BODY_ORDER,
    GO2_BODY_NAME_TO_SLOT,
    GO2_FOOT_BODY_NAMES,
    GO2_FOOT_NAME_TO_SLOT,
    GO2_JOINT_ORDER,
    GO2_CALF_BODY_NAMES,
    GO2_THIGH_BODY_NAMES,
    Go2Indices,
    ordered_body_coms,
    ordered_body_inertias,
    ordered_body_masses,
    ordered_body_state,
    ordered_jacobians,
    ordered_joint_pos,
    ordered_joint_vel,
    ordered_sensor_body_ids,
    resolve_go2_indices,
)
from .math import quat_error, rp_rotation_from_gravity_b, wxyz_to_xyzw, xyzw_to_wxyz
from .registry import load_cfg_from_registry, parse_env_cfg
from .physx import assert_physx_environment, assert_physx_runtime, configure_physx
from .tensor import to_torch

__all__ = [
    "GO2_BODY_ORDER",
    "GO2_BODY_NAME_TO_SLOT",
    "GO2_CALF_BODY_NAMES",
    "GO2_FOOT_BODY_NAMES",
    "GO2_FOOT_NAME_TO_SLOT",
    "GO2_JOINT_ORDER",
    "GO2_THIGH_BODY_NAMES",
    "Go2Indices",
    "ordered_body_coms",
    "ordered_body_inertias",
    "ordered_body_masses",
    "ordered_body_state",
    "ordered_jacobians",
    "ordered_joint_pos",
    "ordered_joint_vel",
    "ordered_sensor_body_ids",
    "load_cfg_from_registry",
    "parse_env_cfg",
    "assert_physx_environment",
    "assert_physx_runtime",
    "configure_physx",
    "quat_error",
    "resolve_go2_indices",
    "rp_rotation_from_gravity_b",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
    "to_torch",
]
