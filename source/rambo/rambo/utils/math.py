"""Small math helpers that are not public Isaac Lab 2.3.2 APIs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rp_rotation_from_gravity_b(grav_b: torch.Tensor) -> torch.Tensor:
    """Return the root-to-body roll/pitch rotation from body-frame gravity.

    The implementation intentionally matches the legacy RAMBO convention.  It
    assumes normalized, non-zero gravity vectors, as supplied by the task's
    projected-gravity observation.
    """

    col1 = F.normalize(torch.stack((-grav_b[..., 2], torch.zeros_like(grav_b[..., 2]), grav_b[..., 0]), dim=-1), dim=-1)
    col2 = F.normalize(torch.cross(-grav_b, col1, dim=-1), dim=-1)
    return torch.stack((col1, col2, -grav_b), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert a scalar-first quaternion to an axis-angle vector."""

    quat = F.normalize(quat, dim=-1)
    quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
    half_angle = torch.atan2(torch.linalg.vector_norm(quat[..., 1:], dim=-1), quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(
        angle.abs() > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return quat[..., 1:] / sin_half_angle_over_angle.unsqueeze(-1)


def quat_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Return the RAMBO axis-angle orientation error from ``q2`` to ``q1``."""

    q2_conjugate = torch.cat((q2[..., :1], -q2[..., 1:]), dim=-1)
    return _axis_angle_from_quat(_quat_mul(q1, q2_conjugate))


@torch.jit.script
def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Bridge simulator XYZW quaternions into RAMBO's legacy WXYZ QP math.

    This is intentionally a conversion boundary rather than a rewrite of
    :func:`quat_error`: checkpoints and the QP formulation retain their
    historical scalar-first convention while Isaac Lab 3 public data is XYZW.
    """

    return torch.cat((quat[..., 3:4], quat[..., :3]), dim=-1)


@torch.jit.script
def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Bridge RAMBO's legacy WXYZ quaternions into simulator XYZW data."""

    return torch.cat((quat[..., 1:], quat[..., :1]), dim=-1)
