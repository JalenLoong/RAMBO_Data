"""Name-based Go2 body/joint and Jacobian ordering contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rambo.utils.articulation import (  # noqa: E402
    GO2_BODY_ORDER,
    GO2_FOOT_BODY_NAMES,
    GO2_JOINT_ORDER,
    ordered_sensor_body_ids,
    ordered_body_coms,
    ordered_body_inertias,
    ordered_body_masses,
    ordered_jacobians,
    ordered_joint_pos,
    ordered_joint_vel,
    resolve_go2_indices,
)


class _Proxy:
    """Minimal Isaac Lab 3 ProxyArray stand-in exposing explicit Torch access."""

    def __init__(self, value: torch.Tensor) -> None:
        self.torch = value


def _fake_robot(*, root_row_included: bool) -> SimpleNamespace:
    # The reverse physical ordering ensures tests cannot accidentally pass with
    # historical raw body/joint integers such as 1, 2, 4, 5, ... .
    body_names = ["base", "Head_link"] + list(reversed(GO2_BODY_ORDER[1:]))
    joint_names = ["unused_joint"] + list(reversed(GO2_JOINT_ORDER))
    body_count = len(body_names)
    columns = 6 + len(joint_names)
    rows = body_count if root_row_included else body_count - 1
    jacobians = torch.empty((1, rows, 6, columns), dtype=torch.float64)
    for physical_row in range(rows):
        physical_body_id = physical_row if root_row_included else physical_row + 1
        jacobians[:, physical_row].fill_(1000.0 + physical_body_id)

    data = SimpleNamespace(
        body_names=body_names,
        joint_names=joint_names,
        body_com_jacobian_w=_Proxy(jacobians),
        body_mass=_Proxy(torch.arange(body_count, dtype=torch.float64).reshape(1, -1)),
        body_inertia=_Proxy(torch.arange(body_count * 9, dtype=torch.float64).reshape(1, body_count, 9)),
        body_com_pose_b=_Proxy(torch.arange(body_count * 7, dtype=torch.float64).reshape(1, body_count, 7)),
        body_link_pose_w=_Proxy(torch.arange(body_count * 7, dtype=torch.float64).reshape(1, body_count, 7)),
        body_link_vel_w=_Proxy((1000 + torch.arange(body_count * 6, dtype=torch.float64)).reshape(1, body_count, 6)),
        joint_pos=_Proxy(torch.arange(len(joint_names), dtype=torch.float64).reshape(1, -1)),
        joint_vel=_Proxy((100 + torch.arange(len(joint_names), dtype=torch.float64)).reshape(1, -1)),
    )
    return SimpleNamespace(data=data, device=torch.device("cpu"))


@pytest.mark.parametrize("root_row_included", (False, True))
def test_go2_mapping_orders_joints_and_jacobians_by_name(root_row_included: bool) -> None:
    robot = _fake_robot(root_row_included=root_row_included)
    indices = resolve_go2_indices(robot)

    expected_body_ids = [robot.data.body_names.index(name) for name in GO2_BODY_ORDER]
    expected_joint_ids = [robot.data.joint_names.index(name) for name in GO2_JOINT_ORDER]
    assert indices.body_ids.tolist() == expected_body_ids
    assert indices.joint_ids.tolist() == expected_joint_ids
    assert indices.head_body_ids.tolist() == [robot.data.body_names.index("Head_link")]

    jacobians = ordered_jacobians(robot, indices)
    assert jacobians.shape[1] == len(GO2_BODY_ORDER)
    if root_row_included:
        expected_rows = torch.tensor([1000.0 + body_id for body_id in expected_body_ids])
    else:
        expected_rows = torch.tensor([0.0] + [1000.0 + body_id for body_id in expected_body_ids[1:]])
    assert torch.equal(jacobians[0, :, 0, 0], expected_rows)

    assert torch.equal(
        ordered_joint_pos(robot, indices), robot.data.joint_pos.torch[:, expected_joint_ids]
    )
    assert torch.equal(
        ordered_joint_vel(robot, indices), robot.data.joint_vel.torch[:, expected_joint_ids]
    )

    assert torch.equal(
        ordered_body_masses(robot, indices), robot.data.body_mass.torch[:, expected_body_ids]
    )
    assert torch.equal(
        ordered_body_inertias(robot, indices), robot.data.body_inertia.torch[:, expected_body_ids]
    )
    assert torch.equal(
        ordered_body_coms(robot, indices), robot.data.body_com_pose_b.torch[:, expected_body_ids]
    )


def test_go2_mapping_rejects_missing_required_names() -> None:
    robot = _fake_robot(root_row_included=False)
    robot.data.joint_names.remove("FL_calf_joint")
    with pytest.raises(RuntimeError, match="FL_calf_joint"):
        resolve_go2_indices(robot)


def test_contact_sensor_mapping_requests_the_declared_logical_name_order() -> None:
    class FakeContactSensor:
        def find_sensors(self, names, preserve_order=False):
            assert names == list(GO2_FOOT_BODY_NAMES)
            assert preserve_order is True
            return [13, 7, 11, 5], list(GO2_FOOT_BODY_NAMES)

    assert ordered_sensor_body_ids(FakeContactSensor(), GO2_FOOT_BODY_NAMES, "foot") == [13, 7, 11, 5]


def test_contact_sensor_mapping_rejects_a_wrong_return_order() -> None:
    class FakeContactSensor:
        def find_sensors(self, _names, preserve_order=False):
            assert preserve_order is True
            return [13, 7, 11, 5], list(reversed(GO2_FOOT_BODY_NAMES))

    with pytest.raises(RuntimeError, match="contact sensor foot name mismatch"):
        ordered_sensor_body_ids(FakeContactSensor(), GO2_FOOT_BODY_NAMES, "foot")
