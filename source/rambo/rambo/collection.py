"""Simulator-native collection interfaces with no dependency on WAM types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    timestamp_ns: int
    rgb: np.ndarray


@dataclass(frozen=True)
class TaskOutcomeSnapshot:
    metrics: Mapping[str, float | bool]
    success: bool
    terminated: bool
    termination_reason: str | None


@dataclass(frozen=True)
class RamboStepSnapshot:
    observation: np.ndarray
    policy_action_18d: np.ndarray
    robot_state: Mapping[str, np.ndarray]
    cameras: tuple[CameraFrame, ...]
    outcome: TaskOutcomeSnapshot
    command_9d: np.ndarray


class RamboCollectionBackend(Protocol):
    """One-environment simulator backend consumed by an external collector."""

    def reset(self, *, seed: int) -> RamboStepSnapshot: ...

    def step(self, command_9d: np.ndarray) -> RamboStepSnapshot: ...

    def close(self) -> None: ...


__all__ = [
    "CameraFrame",
    "RamboCollectionBackend",
    "RamboStepSnapshot",
    "TaskOutcomeSnapshot",
]
