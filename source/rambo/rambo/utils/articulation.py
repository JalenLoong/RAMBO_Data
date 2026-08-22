"""Name-based Go2 articulation access for RAMBO's QP tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


GO2_BODY_ORDER = (
    "base",
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
)
"""QP body order; head links are deliberately excluded."""

GO2_FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_THIGH_BODY_NAMES = ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")
GO2_CALF_BODY_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
GO2_BODY_NAME_TO_SLOT = {name: index for index, name in enumerate(GO2_BODY_ORDER)}
GO2_FOOT_NAME_TO_SLOT = {name: index for index, name in enumerate(GO2_FOOT_BODY_NAMES)}

GO2_JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)


@dataclass(frozen=True)
class Go2Indices:
    """Resolved index tensors in the fixed RAMBO logical ordering."""

    body_ids: torch.Tensor
    joint_ids: torch.Tensor
    foot_body_ids: torch.Tensor
    head_body_ids: torch.Tensor

    def body_id(self, body_name: str) -> torch.Tensor:
        """Return a one-element physical body-id tensor for a logical Go2 name."""

        try:
            logical_slot = GO2_BODY_NAME_TO_SLOT[body_name]
        except KeyError as error:
            raise KeyError(f"{body_name!r} is not a RAMBO QP body") from error
        return self.body_ids[logical_slot : logical_slot + 1]

    def foot_body_id(self, foot_name: str) -> torch.Tensor:
        """Return a one-element physical body-id tensor for a named Go2 foot."""

        try:
            logical_slot = GO2_FOOT_NAME_TO_SLOT[foot_name]
        except KeyError as error:
            raise KeyError(f"{foot_name!r} is not a RAMBO Go2 foot") from error
        return self.foot_body_ids[logical_slot : logical_slot + 1]


def _resolve_names(available: list[str], required: tuple[str, ...], label: str, device: torch.device | str) -> torch.Tensor:
    positions = {name: index for index, name in enumerate(available)}
    missing = [name for name in required if name not in positions]
    if missing:
        raise RuntimeError(f"Go2 {label} name mismatch; missing {missing}, available={available}.")
    return torch.tensor([positions[name] for name in required], dtype=torch.long, device=device)


def resolve_go2_indices(robot: Any) -> Go2Indices:
    """Resolve all RAMBO Go2 indices by name and validate the expected root link."""

    data = robot.data
    device = getattr(robot, "device", "cpu")
    body_ids = _resolve_names(list(data.body_names), GO2_BODY_ORDER, "body", device)
    joint_ids = _resolve_names(list(data.joint_names), GO2_JOINT_ORDER, "joint", device)
    foot_body_ids = _resolve_names(list(data.body_names), GO2_FOOT_BODY_NAMES, "foot body", device)
    head_names = tuple(name for name in data.body_names if name.startswith("Head_"))
    head_body_ids = _resolve_names(list(data.body_names), head_names, "head body", device) if head_names else torch.empty(
        0, dtype=torch.long, device=device
    )
    if int(body_ids[0]) != 0:
        raise RuntimeError(
            "RAMBO expects the Go2 base to be PhysX body 0 so it can synthesize the root Jacobian row. "
            f"Resolved base index: {int(body_ids[0])}."
        )
    return Go2Indices(body_ids=body_ids, joint_ids=joint_ids, foot_body_ids=foot_body_ids, head_body_ids=head_body_ids)


def ordered_jacobians(robot: Any, indices: Go2Indices) -> torch.Tensor:
    """Return Jacobians in :data:`GO2_BODY_ORDER` without numerical body literals.

    Isaac Sim 5's PhysX Jacobian view omits the floating-base root row.  RAMBO's
    gravity compensation only reads the root row's joint columns, which are zero;
    this function therefore inserts an explicit zero root row before gathering the
    remaining links by their resolved body IDs.
    """

    jacobians = robot.root_physx_view.get_jacobians().clone().to(robot.device)
    body_count = len(robot.data.body_names)
    if jacobians.ndim != 4:
        raise RuntimeError(f"Expected Jacobian rank 4, received shape {tuple(jacobians.shape)}.")
    if jacobians.shape[1] == body_count:
        return jacobians.index_select(1, indices.body_ids)
    if jacobians.shape[1] != body_count - 1:
        raise RuntimeError(
            "Unexpected PhysX Jacobian body dimension "
            f"{jacobians.shape[1]} for {body_count} Go2 bodies."
        )
    result = jacobians.new_zeros((jacobians.shape[0], len(GO2_BODY_ORDER), *jacobians.shape[2:]))
    result[:, 1:] = jacobians.index_select(1, indices.body_ids[1:] - 1)
    return result


def ordered_body_masses(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.root_physx_view.get_masses().clone().to(robot.device).index_select(1, indices.body_ids)


def ordered_body_inertias(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.root_physx_view.get_inertias().clone().to(robot.device).index_select(1, indices.body_ids)


def ordered_body_coms(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.root_physx_view.get_coms().clone().to(robot.device).index_select(1, indices.body_ids)


def ordered_body_state(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.data.body_state_w.index_select(1, indices.body_ids)


def ordered_joint_pos(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.data.joint_pos.index_select(1, indices.joint_ids)


def ordered_joint_vel(robot: Any, indices: Go2Indices) -> torch.Tensor:
    return robot.data.joint_vel.index_select(1, indices.joint_ids)


def ordered_sensor_body_ids(sensor: Any, body_names: tuple[str, ...], label: str) -> list[int]:
    """Resolve contact-sensor bodies in an explicit RAMBO logical name order."""

    body_ids, resolved_names = sensor.find_bodies(list(body_names), preserve_order=True)
    if tuple(resolved_names) != body_names:
        raise RuntimeError(
            f"Go2 contact sensor {label} name mismatch; expected {list(body_names)}, "
            f"resolved {resolved_names}."
        )
    return body_ids
