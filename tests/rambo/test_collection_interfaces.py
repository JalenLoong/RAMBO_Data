from __future__ import annotations

import numpy as np

from rambo.collection import CameraFrame, RamboStepSnapshot, TaskOutcomeSnapshot


def test_collection_snapshots_keep_native_9d_and_18d_surfaces_separate() -> None:
    frame = CameraFrame("ego", 80_000_000, np.zeros((480, 640, 3), dtype=np.uint8))
    outcome = TaskOutcomeSnapshot({"contact": True}, True, False, None)
    snapshot = RamboStepSnapshot(
        observation=np.zeros(405, dtype=np.float32),
        policy_action_18d=np.zeros(18, dtype=np.float32),
        robot_state={"joint_pos": np.zeros(12, dtype=np.float32)},
        cameras=(frame,),
        outcome=outcome,
        command_9d=np.zeros(9, dtype=np.float32),
    )
    assert snapshot.command_9d.shape == (9,)
    assert snapshot.policy_action_18d.shape == (18,)
    assert snapshot.observation.shape == (405,)
