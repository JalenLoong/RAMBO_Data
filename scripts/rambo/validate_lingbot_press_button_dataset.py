#!/usr/bin/env python3
"""Validate the complete 10-episode randomized LingBot-RAMBO Press-button dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from rambo.validation.lingbot_dataset_v1 import (
    DatasetValidationError,
    TASK_ID,
    TASK_INSTRUCTION,
    sha256_file,
    validate_episode,
)


def _fail(message: str) -> None:
    raise DatasetValidationError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/datasets/lingbot_rambo_v1/press_button"))
    parser.add_argument("--config", type=Path, default=Path("configs/lingbot_rambo_v1/press_button_randomization.json"))
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    try:
        expected_ids = [f"episode_{number:06d}" for number in range(1, 11)]
        actual_ids = sorted(path.name for path in root.glob("episode_*") if path.is_dir())
        if actual_ids != expected_ids:
            _fail(f"episode directories differ: expected {expected_ids}, got {actual_ids}")
        golden = root / "episode_000001"
        if sha256_file(golden / "checksums.sha256") != config["golden_checksums_sha256"]:
            _fail("golden episode checksum manifest was modified")
        results = []
        positions = []
        golden_metadata = None
        for number, episode_id in enumerate(expected_ids, 1):
            episode = root / episode_id
            result = validate_episode(episode, verify_checksums=True, verify_images=True)
            metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
            task = json.loads((episode / "task.json").read_text(encoding="utf-8"))
            outcome = json.loads((episode / "outcome.json").read_text(encoding="utf-8"))
            if metadata["task_instruction"] != TASK_INSTRUCTION or task["instruction"] != TASK_INSTRUCTION:
                _fail(f"{episode_id} instruction differs")
            if metadata["task_id"] != TASK_ID or task["task_id"] != TASK_ID:
                _fail(f"{episode_id} task differs")
            if golden_metadata is None:
                golden_metadata = metadata
            else:
                for key in ("sha256",):
                    if metadata["checkpoint"][key] != golden_metadata["checkpoint"][key]:
                        _fail(f"{episode_id} checkpoint provenance differs")
                if metadata["rambo"]["git_commit"] != golden_metadata["rambo"]["git_commit"]:
                    _fail(f"{episode_id} RAMBO commit differs")
                if metadata["runtime"]["isaacsim_version"] != golden_metadata["runtime"]["isaacsim_version"]:
                    _fail(f"{episode_id} Isaac Sim version differs")
                if metadata["cameras"] != golden_metadata["cameras"]:
                    _fail(f"{episode_id} camera contract differs")
            position = np.asarray(task["initial_state"]["button_cap_pose_w_xyzw"][:3], dtype=np.float64)
            positions.append(position)
            if number > 1:
                randomization = metadata.get("randomization")
                compatible_versions = set(config.get("compatible_config_versions", [config["version"]]))
                if not isinstance(randomization, dict) or randomization.get("config_version") not in compatible_versions:
                    _fail(f"{episode_id} randomization metadata is incomplete")
                if metadata["seed"] != number or randomization.get("seed") != number:
                    _fail(f"{episode_id} seed convention differs")
            if outcome["success"] is not True or outcome["terminal"] is not False:
                _fail(f"{episode_id} is not a successful non-terminal demonstration")
            results.append(result)
        positions_array = np.asarray(positions, dtype=np.float64)
        if not np.isfinite(positions_array).all():
            _fail("Button positions contain NaN/Inf")
        new_positions = positions_array[1:, :2]
        distances = [
            float(np.linalg.norm(new_positions[i] - new_positions[j]))
            for i in range(len(new_positions))
            for j in range(i + 1, len(new_positions))
        ]
        if not distances or min(distances) < float(config["minimum_xy_separation_m"]):
            _fail("randomized Button positions violate the diversity threshold")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("episode_count") != 10 or manifest.get("successful_episode_count") != 10:
            _fail("dataset manifest counts differ from 10 successful episodes")
        if len(manifest.get("episodes", [])) != 10:
            _fail("dataset manifest episode list is incomplete")
        for required in ("button_positions.png", "DATASET_SUMMARY.md", "batch_generation_log.json"):
            if not (root / required).is_file():
                _fail(f"dataset artifact missing: {required}")
    except (DatasetValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"LINGBOT_PRESS_BUTTON_DATASET_VALIDATION_FAILED: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "passed": True,
                "dataset_root": str(root),
                "episode_count": 10,
                "successful_episode_count": 10,
                "minimum_new_xy_separation_m": min(distances),
                "success_steps": [result["success_policy_step"] for result in results],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("LINGBOT_PRESS_BUTTON_DATASET_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
