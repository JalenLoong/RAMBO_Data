#!/usr/bin/env python3
"""Serially generate the ten successful Dataset V1 episodes for one object task."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from rambo.validation import lingbot_dataset_v1 as common
from rambo.validation.lingbot_object_dataset_v1 import (
    ObjectDatasetValidationError,
    profile,
    validate_episode,
)


ROOT = Path("/workspace/datasets/lingbot_rambo_v1")
CONFIGS = {
    "lift_basket": Path("configs/lingbot_rambo_v1/lift_basket_randomization.json"),
    "pull_object_into_basket": Path("configs/lingbot_rambo_v1/pull_object_into_basket_randomization.json"),
    "shoot_ball_into_goal": Path("configs/lingbot_rambo_v1/shoot_ball_into_goal_randomization.json"),
}


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("version", "reference_primary_position", "x_range_m", "y_range_m", "z_fixed_m", "minimum_xy_separation_m"):
        if key not in value:
            raise RuntimeError(f"randomization config is missing {key}")
    return value


def _candidate(seed: int, attempt: int, config: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(seed * 10_000 + attempt)
    return np.asarray((rng.uniform(*config["x_range_m"]), rng.uniform(*config["y_range_m"]), config["z_fixed_m"]), dtype=np.float64)


def _reject(position: np.ndarray, accepted: list[np.ndarray], config: dict[str, Any]) -> str | None:
    if position.shape != (3,) or not np.isfinite(position).all():
        return "non_finite_pose"
    if not config["x_range_m"][0] <= position[0] <= config["x_range_m"][1] or not config["y_range_m"][0] <= position[1] <= config["y_range_m"][1]:
        return "outside_xy_bounds"
    if not math.isclose(float(position[2]), float(config["z_fixed_m"]), abs_tol=1e-8):
        return "z_or_orientation_not_fixed"
    if np.linalg.norm(position[:2]) < 0.40:
        return "initial_robot_collision_risk"
    if any(np.linalg.norm(position[:2] - prior[:2]) < float(config["minimum_xy_separation_m"]) for prior in accepted):
        return "insufficient_xy_diversity"
    return None


def _write_log(path: Path, value: dict[str, Any], stdout: str = "", stderr: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n" + stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")


def _run(repo: Path, checkpoint: Path, profile_name: str, output: Path, episode_id: str, seed: int, position: np.ndarray, config: Path | None) -> subprocess.CompletedProcess[str]:
    command = [str(repo / "scripts/rambo/run.sh"), str(repo / "scripts/rambo/record_lingbot_object_dataset_v1.py"), "--checkpoint", str(checkpoint), "--task-profile", profile_name, "--output-dir", str(output), "--episode-id", episode_id, "--seed", str(seed), "--primary-position", *(f"{value:.9f}" for value in position), "--viz", "none"]
    if config is not None:
        command.extend(("--randomization-config", str(config), "--randomization-seed", str(seed)))
    env = os.environ.copy(); env["OMNI_KIT_ACCEPT_EULA"] = "Y"
    return subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, check=False)


def _plot(path: Path, task: str, config: dict[str, Any], records: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (900, 650), "white"); draw = ImageDraw.Draw(image)
    x0, x1 = map(float, config["x_range_m"]); y0, y1 = map(float, config["y_range_m"])
    pad = 0.01; x0 -= pad; x1 += pad; y0 -= pad; y1 += pad
    margin = 80
    def point(position: list[float]) -> tuple[int, int]:
        return (round(margin + (position[0] - x0) / (x1 - x0) * (900 - 2 * margin)), round(650 - margin - (position[1] - y0) / (y1 - y0) * (650 - 2 * margin)))
    draw.rectangle((margin, margin, 900 - margin, 650 - margin), outline="black", width=2)
    for record in records:
        x, y = point(record["primary_position_w_m"]); color = "gold" if record["episode_id"] == "episode_000001" else "royalblue"
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="black"); draw.text((x + 9, y - 11), record["episode_id"], fill="black")
    draw.text((margin, 25), f"LingBot-RAMBO V1 — {task}: primary object XY", fill="black")
    draw.text((margin, 615), f"Bounds: x={config['x_range_m']} m; y={config['y_range_m']} m; minimum separation={config['minimum_xy_separation_m']} m", fill="black")
    image.save(path)


def _dataset_artifacts(root: Path, task: str, config_path: Path, config: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    cfg = profile(task); records: list[dict[str, Any]] = []
    for number in range(1, 11):
        episode_id = f"episode_{number:06d}"; episode = root / episode_id
        result = validate_episode(episode, verify_checksums=True, verify_images=True)
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8")); task_doc = json.loads((episode / "task.json").read_text(encoding="utf-8")); outcome = json.loads((episode / "outcome.json").read_text(encoding="utf-8"))
        records.append({"episode_id": episode_id, "seed": metadata["seed"], "runtime_seed": metadata["runtime_seed"], "primary_position_w_m": task_doc["initial_state"]["primary_pose_w_xyzw"][:3], "success": outcome["success"], "success_step": outcome["first_success_policy_step"], "first_contact_step": outcome["first_contact_policy_step"], "terminal": outcome["terminal"], "raw_actions": 3000, "train_actions": 1500, "rgb_frames_per_view": 375, "checksum_path": f"{episode_id}/checksums.sha256", "validation": result})
    manifest = {"dataset": common.DATASET_NAME, "dataset_version": common.DATASET_VERSION, "task_profile": task, "task_id": cfg["task_id"], "instruction": cfg["instruction"], "command_program_id": cfg["command_program_id"], "episode_count": 10, "successful_episode_count": 10, "randomization_config": str(config_path.resolve()), "randomization_config_version": config["version"], "episodes": records}
    common.write_json(root / "manifest.json", manifest); common.write_json(root / "batch_generation_log.json", {"failed_or_rejected_attempts": failures})
    _plot(root / "primary_positions.png", task, config, records)
    steps = [item["success_step"] for item in records]
    (root / "DATASET_SUMMARY.md").write_text("\n".join((f"# LingBot-RAMBO Dataset V1 — {cfg['instruction']}", "", f"- Task ID: `{cfg['task_id']}`", f"- Episodes: 10/10 successful; only {cfg['primary_key']} XY varies after golden 000001.", f"- Sample bounds: x={config['x_range_m']} m; y={config['y_range_m']} m; Z={config['z_fixed_m']} m; min XY separation={config['minimum_xy_separation_m']} m.", "- Runtime/controller seed is fixed at 42; sampling seeds are episode numbers 2–10.", f"- Success policy-step distribution: min={min(steps)}, max={max(steps)}, values={steps}.", "- Per episode: 3000 raw 9D commands, 1500 exact raw[::2] train commands, 3000 state/action rows, 375 RGB PNGs per view, and three H.264 MP4s.", f"- Failed/rejected rollout attempts retained at: `{root.parent / '_failed_attempts' / task}` ({len(failures)} recorded).", "")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-profile", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=24)
    args = parser.parse_args()
    if args.max_attempts <= 0: parser.error("--max-attempts must be positive")
    repo = Path(__file__).resolve().parents[2]; root = (args.dataset_root or ROOT / args.task_profile).resolve(); config_path = (args.config or CONFIGS[args.task_profile]).resolve(); config = _read_config(config_path)
    root.mkdir(parents=True, exist_ok=True); failed_root = root.parent / "_failed_attempts" / args.task_profile; accepted: list[np.ndarray] = []; failures: list[dict[str, Any]] = []
    for number in range(1, 11):
        episode_id = f"episode_{number:06d}"; output = root / episode_id
        if output.is_dir():
            try:
                validate_episode(output, verify_checksums=True, verify_images=True)
                position = np.asarray(json.loads((output / "task.json").read_text())["initial_state"]["primary_pose_w_xyzw"][:3], dtype=np.float64)
                if _reject(position, accepted, config) and number != 1: raise ObjectDatasetValidationError("existing sample fails pose/diversity gates")
            except (ObjectDatasetValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise SystemExit(f"refusing to overwrite invalid existing {episode_id}: {error}") from error
            accepted.append(position); print(f"RESUMED {episode_id} position={position.tolist()}", flush=True); continue
        if number == 1:
            candidates = [(1, np.asarray(config["reference_primary_position"], dtype=np.float64), None)]
        else:
            candidates = [(attempt, _candidate(number, attempt, config), config_path) for attempt in range(1, args.max_attempts + 1)]
        accepted_this = False
        for attempt, position, run_config in candidates:
            record = {"episode_id": episode_id, "sampling_seed": number if number > 1 else 42, "attempt": attempt, "position": position.tolist()}
            reason = _reject(position, accepted, config)
            if number > 1 and reason:
                record.update({"result": "rejected_before_rollout", "reason": reason}); failures.append(record); _write_log(failed_root / f"{episode_id}_attempt_{attempt:02d}.log", record); continue
            process = _run(repo, args.checkpoint.resolve(), args.task_profile, output, episode_id, 42 if number == 1 else number, position, run_config)
            if output.is_dir():
                try: validate_episode(output, verify_checksums=True, verify_images=True)
                except ObjectDatasetValidationError as error: record.update({"result": "failed_validation", "reason": str(error)})
                else:
                    accepted.append(position); accepted_this = True; print(f"ACCEPTED {episode_id} attempt={attempt} position={position.tolist()}", flush=True); break
            else: record.update({"result": "failed_rollout", "reason": f"recorder_returncode={process.returncode}"})
            failures.append(record); _write_log(failed_root / f"{episode_id}_attempt_{attempt:02d}.log", record, process.stdout, process.stderr)
            staging = output.with_name(output.name + ".inprogress")
            if staging.exists(): os.replace(staging, failed_root / f"{episode_id}_attempt_{attempt:02d}.inprogress")
        if not accepted_this: raise SystemExit(f"could not obtain successful {episode_id} within {args.max_attempts} attempts")
    _dataset_artifacts(root, args.task_profile, config_path, config, failures)
    print(f"LINGBOT_OBJECT_BATCH_{args.task_profile.upper()}_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
