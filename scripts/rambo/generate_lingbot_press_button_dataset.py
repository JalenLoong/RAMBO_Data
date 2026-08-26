#!/usr/bin/env python3
"""Generate randomized Button-position Dataset V1 episodes 000002 through 000010."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from rambo.validation.lingbot_dataset_v1 import (
    DATASET_VERSION,
    TASK_ID,
    TASK_INSTRUCTION,
    DatasetValidationError,
    sha256_file,
    validate_episode,
    write_json,
)


DATASET_ROOT = Path("/workspace/datasets/lingbot_rambo_v1/press_button")
DEFAULT_CONFIG = Path("configs/lingbot_rambo_v1/press_button_randomization.json")


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("version", "reference_episode", "reference_button_pose", "x_range_m", "y_range_m", "z_fixed_m", "minimum_xy_separation_m"):
        if key not in value:
            raise RuntimeError(f"randomization config lacks {key}")
    return value


def _candidate(seed: int, attempt: int, config: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(seed * 10_000 + attempt)
    return np.asarray(
        [
            rng.uniform(*config["x_range_m"]),
            rng.uniform(*config["y_range_m"]),
            config["z_fixed_m"],
        ],
        dtype=np.float64,
    )


def _cheap_validity_reason(position: np.ndarray, accepted: list[np.ndarray], config: dict[str, Any]) -> str | None:
    if position.shape != (3,) or not np.isfinite(position).all():
        return "non_finite_position"
    x_range, y_range = config["x_range_m"], config["y_range_m"]
    if not x_range[0] <= position[0] <= x_range[1] or not y_range[0] <= position[1] <= y_range[1]:
        return "outside_conservative_xy_bounds"
    if not math.isclose(float(position[2]), float(config["z_fixed_m"]), abs_tol=1e-6):
        return "z_not_fixed"
    constraints = config["constraints"]
    robot_xy = np.asarray(constraints["robot_initial_xy_m"], dtype=np.float64)
    if np.linalg.norm(position[:2] - robot_xy) < float(constraints["minimum_robot_to_button_xy_distance_m"]):
        return "robot_initial_collision_or_overlap_risk"
    if abs(float(position[1]) - float(constraints["scripted_fl_target_y_m"])) > float(
        constraints["scripted_fl_target_y_tolerance_m"]
    ):
        return "outside_scripted_fl_lateral_reach"
    minimum_distance = float(config["minimum_xy_separation_m"])
    if any(np.linalg.norm(position[:2] - prior[:2]) < minimum_distance for prior in accepted):
        return "insufficient_xy_diversity"
    return None


def _write_attempt_log(path: Path, record: dict[str, Any], *, stdout: str = "", stderr: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n" + stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")


def _run_episode(
    *,
    repo_root: Path,
    checkpoint: Path,
    output_dir: Path,
    episode_id: str,
    seed: int,
    position: np.ndarray,
    config_path: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(repo_root / "scripts/rambo/run.sh"),
        str(repo_root / "scripts/rambo/record_lingbot_dataset_v1.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--episode-id",
        episode_id,
        "--seed",
        str(seed),
        "--randomization-seed",
        str(seed),
        "--randomization-config",
        str(config_path),
        "--button-position",
        *(f"{value:.9f}" for value in position),
        "--viz",
        "none",
    ]
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "Y"
    return subprocess.run(command, cwd=repo_root, env=environment, text=True, capture_output=True, check=False)


def _create_position_plot(path: Path, config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    width, height, margin = 900, 650, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x0, x1 = map(float, config["x_range_m"])
    y0, y1 = map(float, config["y_range_m"])
    pad_x, pad_y = 0.01, 0.015
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y
    def project(point: list[float]) -> tuple[int, int]:
        x = margin + (point[0] - x0) / (x1 - x0) * (width - 2 * margin)
        y = height - margin - (point[1] - y0) / (y1 - y0) * (height - 2 * margin)
        return round(x), round(y)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black", width=2)
    reference = config["reference_button_pose"]["position"]
    rx, ry = project(reference)
    draw.ellipse((rx - 8, ry - 8, rx + 8, ry + 8), fill="gold", outline="black")
    draw.text((rx + 10, ry - 12), "golden 000001", fill="black")
    robot = config["constraints"]["robot_initial_xy_m"]
    qx, qy = project([robot[0], robot[1]])
    draw.rectangle((qx - 6, qy - 6, qx + 6, qy + 6), fill="black")
    draw.text((qx + 10, qy - 12), "robot initial XY", fill="black")
    for record in records:
        x, y = project(record["button_position"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="royalblue", outline="black")
        draw.text((x + 9, y - 10), record["episode_id"], fill="black")
    draw.text((margin, 28), "LingBot-RAMBO Press-the-button: Button XY randomization", fill="black")
    draw.text((margin, height - 55), f"Bounds x={config['x_range_m']} m, y={config['y_range_m']} m", fill="black")
    image.save(path)


def _write_dataset_artifacts(root: Path, config_path: Path, config: dict[str, Any]) -> None:
    episode_records: list[dict[str, Any]] = []
    for number in range(1, 11):
        episode_id = f"episode_{number:06d}"
        episode = root / episode_id
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        task = json.loads((episode / "task.json").read_text(encoding="utf-8"))
        outcome = json.loads((episode / "outcome.json").read_text(encoding="utf-8"))
        position = task["initial_state"]["button_cap_pose_w_xyzw"][:3]
        record = {
            "episode_id": episode_id,
            "seed": metadata["seed"],
            "button_position": position,
            "position_delta_from_reference": [position[index] - config["reference_button_pose"]["position"][index] for index in range(3)],
            "success": outcome["success"],
            "success_step": outcome["first_success_policy_step"],
            "success_timestamp_ns": outcome["first_success_timestamp_ns"],
            "rebound_step": outcome["first_rebound_policy_step"],
            "terminal": outcome["terminal"],
            "duration_s": outcome["duration_s"],
            "rgb_frames_per_view": metadata["counts"]["rgb_frames_per_view"],
            "raw_action_count": metadata["counts"]["raw_commands"],
            "train_action_count": metadata["counts"]["train_commands"],
            "checksum_path": f"{episode_id}/checksums.sha256",
            "metadata_path": f"{episode_id}/metadata.json",
        }
        episode_records.append(record)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "task_instruction": TASK_INSTRUCTION,
        "task_id": TASK_ID,
        "episode_count": 10,
        "successful_episode_count": 10,
        "randomization_config": str(config_path.resolve()),
        "randomization_config_version": config["version"],
        "golden_checksums_sha256": config["golden_checksums_sha256"],
        "rambo_commit": json.loads((root / "episode_000001/metadata.json").read_text(encoding="utf-8"))["rambo"]["git_commit"],
        "checkpoint_sha256": json.loads((root / "episode_000001/metadata.json").read_text(encoding="utf-8"))["checkpoint"]["sha256"],
        "isaac_version": json.loads((root / "episode_000001/metadata.json").read_text(encoding="utf-8"))["runtime"]["isaacsim_version"],
        "episodes": episode_records,
    }
    write_json(root / "manifest.json", manifest)
    _create_position_plot(root / "button_positions.png", config, episode_records[1:])
    rejected = sorted((root.parent / "_failed_attempts" / "press_button").glob("*.log"))
    success_steps = [record["success_step"] for record in episode_records]
    summary = "\n".join(
        [
            "# LingBot-RAMBO Dataset V1 — Press the button",
            "",
            f"- Task prompt: `{TASK_INSTRUCTION}`",
            "- Episodes: 10/10 successful; only Button XY translation varies for episodes 000002–000010.",
            f"- Randomization config: `{config_path}` ({config['version']})",
            f"- Bounds: x={config['x_range_m']} m; y={config['y_range_m']} m; z={config['z_fixed_m']} m; min XY separation={config['minimum_xy_separation_m']} m.",
            "- Seeds: golden=42; randomized episodes=2–10.",
            f"- Success-step distribution: min={min(success_steps)}, max={max(success_steps)}, values={success_steps}.",
            "- Per episode: 3000×9 raw commands, 1500×9 train commands, and 375 RGB frames for each of ego/left/right.",
            "- Camera, policy, QP, physics, action semantics, and timing are fixed to the V1 golden contract.",
            f"- Failed/rejected attempts retained outside the training directory: {len(rejected)}.",
            "- Limitation: variation is deliberately conservative and only covers a small reachable/viewable XY region.",
            "",
        ]
    )
    (root / "DATASET_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-attempts", type=int, default=24)
    args = parser.parse_args()
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    root = args.dataset_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    golden = root / "episode_000001"
    try:
        validate_episode(golden)
    except DatasetValidationError as error:
        raise SystemExit(f"golden sample validation failed: {error}") from error
    if sha256_file(golden / "checksums.sha256") != config["golden_checksums_sha256"]:
        raise SystemExit("golden checksums.sha256 differs from the immutable configured reference")
    accepted = [np.asarray(config["reference_button_pose"]["position"], dtype=np.float64)]
    failed_root = root.parent / "_failed_attempts" / "press_button"
    attempt_prefix = str(config["version"]).replace("/", "_")
    failures: list[dict[str, Any]] = []
    for seed in range(2, 11):
        episode_id = f"episode_{seed:06d}"
        output = root / episode_id
        if output.exists():
            try:
                validate_episode(output)
                metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                task = json.loads((output / "task.json").read_text(encoding="utf-8"))
                if metadata.get("seed") != seed:
                    raise DatasetValidationError("seed differs from episode ID")
                position = np.asarray(task["initial_state"]["button_cap_pose_w_xyzw"][:3], dtype=np.float64)
                if _cheap_validity_reason(position, accepted, config) is not None:
                    raise DatasetValidationError("existing episode does not satisfy current cheap validity/diversity gate")
            except (DatasetValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise SystemExit(f"refusing to overwrite invalid existing {episode_id}: {error}") from error
            accepted.append(position)
            print(f"RESUMED {episode_id} position={position.tolist()}", flush=True)
            continue
        accepted_this_episode = False
        for attempt in range(1, args.max_attempts + 1):
            attempt_log = failed_root / f"{attempt_prefix}_{episode_id}_attempt_{attempt:02d}.log"
            attempt_staging = failed_root / f"{attempt_prefix}_{episode_id}_attempt_{attempt:02d}.inprogress"
            if attempt_log.exists() or attempt_staging.exists():
                continue
            position = _candidate(seed, attempt, config)
            reason = _cheap_validity_reason(position, accepted, config)
            record = {"episode_id": episode_id, "seed": seed, "attempt": attempt, "position": position.tolist()}
            if reason is not None:
                record["result"] = "rejected_before_rollout"
                record["reason"] = reason
                failures.append(record)
                _write_attempt_log(attempt_log, record)
                continue
            process = _run_episode(
                repo_root=repo_root,
                checkpoint=args.checkpoint.expanduser().resolve(),
                output_dir=output,
                episode_id=episode_id,
                seed=seed,
                position=position,
                config_path=config_path,
            )
            # Kit may terminate with a non-zero process status during its final
            # renderer teardown after the recorder has already atomically
            # published a fully validated artifact.  The artifact validator,
            # not that teardown status, is the acceptance authority.
            if output.is_dir():
                try:
                    validate_episode(output)
                except DatasetValidationError as error:
                    record.update({"result": "failed_validation", "reason": str(error)})
                else:
                    accepted.append(position)
                    accepted_this_episode = True
                    print(f"ACCEPTED {episode_id} attempt={attempt} position={position.tolist()}", flush=True)
                    break
            else:
                record.update({"result": "failed_rollout", "reason": f"recorder_returncode={process.returncode}"})
            failures.append(record)
            _write_attempt_log(
                attempt_log, record, stdout=process.stdout, stderr=process.stderr
            )
            staging = output.with_name(output.name + ".inprogress")
            if staging.exists():
                os.replace(staging, attempt_staging)
        if not accepted_this_episode:
            raise SystemExit(f"could not obtain a successful {episode_id} within {args.max_attempts} attempts")
    _write_dataset_artifacts(root, config_path, config)
    write_json(root / "batch_generation_log.json", {"failed_or_rejected_attempts": failures})
    print("LINGBOT_PRESS_BUTTON_BATCH_GENERATION_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
