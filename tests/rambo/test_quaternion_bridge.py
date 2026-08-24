"""Quaternion-convention boundaries between Isaac Lab 3 and RAMBO QP math."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from rambo.utils.math import quat_error, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402


def test_xyzw_wxyz_bridge_round_trips_identity_and_rotation() -> None:
    simulator_xyzw = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]], dtype=torch.float64
    )
    expected_rambo_wxyz = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], dtype=torch.float64
    )
    assert torch.equal(xyzw_to_wxyz(simulator_xyzw), expected_rambo_wxyz)
    assert torch.equal(wxyz_to_xyzw(expected_rambo_wxyz), simulator_xyzw)


def test_rambo_wxyz_qp_error_is_only_evaluated_after_explicit_bridge() -> None:
    identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    yaw_90_xyzw = torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]], dtype=torch.float64)

    error = quat_error(xyzw_to_wxyz(yaw_90_xyzw), xyzw_to_wxyz(identity_xyzw))
    torch.testing.assert_close(error, torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float64))
