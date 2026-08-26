"""Pure-Python task-profile and success-detector tests for Dataset V1 object tasks."""

from __future__ import annotations

import numpy as np
import pytest

from rambo.validation import lingbot_object_dataset_v1 as schema


def _base() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(3000, dtype=np.bool_), np.zeros(3000, dtype=np.bool_)


def test_profiles_expose_exact_instructions_and_command_programs() -> None:
    expected = {
        "lift_basket": ("Lift the basket", "rambo-lift-basket-script-v1"),
        "pull_object_into_basket": ("Pull the object on the table into the basket", "rambo-pull-object-into-basket-script-v1"),
        "shoot_ball_into_goal": ("Shoot the ball into the goal", "rambo-shoot-ball-into-goal-script-v1"),
    }
    assert {name: (schema.profile(name)["instruction"], schema.profile(name)["command_program_id"]) for name in expected} == expected


def test_lift_detector_requires_contact_clearance_tilt_and_25_steps() -> None:
    success, contact = _base(); clearance = np.zeros(3000, np.float32); tilt = np.zeros(3000, np.float32)
    clearance[50:75] = 0.061; contact[50:75] = True; success[74:] = True
    events = schema.derive_events("lift_basket", {"fl_contact": contact, "clearance_m": clearance, "tilt_rad": tilt}, success)
    assert events["first_success_step"] == 75
    success[:] = False; success[73:] = True
    with pytest.raises(schema.ObjectDatasetValidationError, match="disagrees"):
        schema.derive_events("lift_basket", {"fl_contact": contact, "clearance_m": clearance, "tilt_rad": tilt}, success)


def test_lift_rejects_insufficient_clearance_or_excess_tilt() -> None:
    success, contact = _base(); clearance = np.full(3000, 0.059, np.float32); tilt = np.zeros(3000, np.float32); contact[:] = True
    assert not schema.derive_events("lift_basket", {"fl_contact": contact, "clearance_m": clearance, "tilt_rad": tilt}, success)["success"]
    clearance[:] = 0.07; tilt[:] = np.deg2rad(36.0)
    assert not schema.derive_events("lift_basket", {"fl_contact": contact, "clearance_m": clearance, "tilt_rad": tilt}, success)["success"]


def test_pull_detector_requires_full_containment_settle_contact_and_departure() -> None:
    success, contact = _base(); inside = np.zeros(3000, bool); speed = np.ones(3000, np.float32); left = np.zeros(3000, bool)
    inside[100:125] = True; speed[100:125] = 0.14; left[100:] = True; contact[20:] = True; success[124:] = True
    events = schema.derive_events("pull_object_into_basket", {"fl_contact": contact, "inside_basket": inside, "speed_m_s": speed, "left_table": left}, success)
    assert events["first_success_step"] == 125
    speed[110] = 0.16; success[:] = False
    assert not schema.derive_events("pull_object_into_basket", {"fl_contact": contact, "inside_basket": inside, "speed_m_s": speed, "left_table": left}, success)["success"]


def test_shoot_detector_rejects_no_contact_or_short_capture() -> None:
    success, contact = _base(); scored = np.zeros(3000, bool); behind = np.zeros(3000, np.float32)
    scored[400:410] = True; behind[400:410] = 0.11; success[409:] = True
    with pytest.raises(schema.ObjectDatasetValidationError, match="disagrees"):
        schema.derive_events("shoot_ball_into_goal", {"fl_contact": contact, "scored": scored, "behind_goal_m": behind}, success)
    contact[300:] = True
    assert schema.derive_events("shoot_ball_into_goal", {"fl_contact": contact, "scored": scored, "behind_goal_m": behind}, success)["first_success_step"] == 410


@pytest.mark.parametrize("profile_name", tuple(schema.PROFILES))
def test_detector_rejects_nan_or_terminal_style_invalid_trace_shapes(profile_name: str) -> None:
    success = np.zeros(2999, dtype=np.bool_)
    with pytest.raises(schema.ObjectDatasetValidationError):
        schema.derive_events(profile_name, {"fl_contact": np.zeros(3000, dtype=np.bool_)}, success)
