"""Offline contracts for RAMBO's opt-in final third-person validation capture.

The normal checkpoint validator deliberately records only the production front
camera.  This module describes the extra, validation-only evidence emitted by
``validate.py --third-person-diagnostic final`` after that rollout has already
completed.  Keeping the timing and checksum checks here makes the evidence
auditable without starting Isaac Sim or selecting a physics backend.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 1
ARTIFACT_KIND = "rambo_final_third_person_diagnostic"
PROVENANCE_SCHEMA_VERSION = 1
FINAL_ACTION_STEPS = 3_000
FINAL_PHYSICS_TICKS = 15_000
FINAL_FRONT_RGB_FRAMES = 375
CHECKSUMS_FILENAME = "checksums.sha256"
MANIFEST_FILENAME = "third_person_diagnostic_manifest.json"
PROCESS_EXIT_FILENAME = "process_exit.json"
TARGET_ISAACSIM_VERSION = "6.0.1.0"
TARGET_ISAACLAB_TAG = "v3.0.0-beta2.patch1"
TARGET_ISAACLAB_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
PHYSX_MANAGER_FACTORY_FQN = "isaaclab_physx.physics.physx_manager:PhysxManager"


class ThirdPersonDiagnosticError(RuntimeError):
    """Raised when final-capture evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThirdPersonDiagnosticError(message)


def _as_finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ThirdPersonDiagnosticError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ThirdPersonDiagnosticError(f"{label} must be finite")
    return result


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ThirdPersonDiagnosticError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ThirdPersonDiagnosticError(f"{label} must be an integer") from exc
    if result != value:
        raise ThirdPersonDiagnosticError(f"{label} must be an exact integer")
    return result


def _require_non_empty_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value


def _require_git_head(value: Any, label: str) -> str:
    head = _require_non_empty_string(value, label)
    _require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, f"{label} must be a 40-character Git SHA")
    return head


def _require_absolute_path(value: Any, label: str) -> str:
    path = _require_non_empty_string(value, label)
    _require(Path(path).is_absolute(), f"{label} must be absolute")
    return path


def _command_option(command: list[str], option: str) -> str | None:
    """Return a conventional CLI option value without interpreting shell input."""

    for index, argument in enumerate(command):
        if argument == option and index + 1 < len(command):
            return command[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def validate_prelaunch_source_provenance(
    provenance: dict[str, Any], *, require_clean_source: bool = True
) -> None:
    """Validate source facts captured before ``AppLauncher`` can start Kit.

    The RAMBO Git HEAD and porcelain status must be sampled before creating an
    Isaac Sim application.  This prevents a final image artifact from claiming
    a clean source revision after a simulator-side action has already begun.
    The function is deliberately pure Python so auditors can test an artifact
    without importing Isaac Sim or selecting any physics backend.
    """

    _require(isinstance(provenance, dict), "third-person provenance is required")
    _require(
        provenance.get("schema_version") == PROVENANCE_SCHEMA_VERSION,
        "third-person provenance schema version is invalid",
    )
    _require(
        provenance.get("collection_phase") == "pre_app_launcher_source_then_runtime",
        "third-person provenance was not collected before AppLauncher",
    )
    command = provenance.get("command")
    _require(isinstance(command, list) and len(command) >= 2, "provenance.command is required")
    _require(all(isinstance(argument, str) and argument for argument in command), "provenance.command is invalid")
    _require_absolute_path(command[0], "provenance.command[0]")
    _require_absolute_path(command[1], "provenance.command[1]")
    _require(
        _command_option(command, "--third-person-diagnostic") == "final",
        "provenance.command must record --third-person-diagnostic final",
    )
    _require(
        _command_option(command, "--steps") == str(FINAL_ACTION_STEPS),
        "provenance.command must record --steps 3000",
    )

    rambo = provenance.get("rambo")
    _require(isinstance(rambo, dict), "provenance.rambo is required")
    _require_absolute_path(rambo.get("module_path"), "provenance.rambo.module_path")
    _require_absolute_path(rambo.get("git_root"), "provenance.rambo.git_root")
    _require_git_head(rambo.get("git_head"), "provenance.rambo.git_head")
    clean = rambo.get("pre_run_worktree_clean")
    _require(isinstance(clean, bool), "provenance.rambo.pre_run_worktree_clean must be bool")
    status = rambo.get("pre_run_status_porcelain_v1")
    _require(
        isinstance(status, list) and all(isinstance(line, str) and line for line in status),
        "provenance.rambo.pre_run_status_porcelain_v1 must be a list of non-empty strings",
    )
    _require(
        clean is (len(status) == 0),
        "provenance.rambo clean flag disagrees with pre-run porcelain status",
    )
    if require_clean_source:
        _require(clean, "Final third-person artifact requires a clean RAMBO Git worktree before AppLauncher")


def validate_runtime_provenance(provenance: dict[str, Any], *, require_clean_source: bool = True) -> None:
    """Fail closed unless final third-person evidence identifies its full runtime.

    This is intentionally stricter than a diagnostic log: the manifest must
    bind the final RGBD evidence to the pinned Isaac Sim and Isaac Lab sources,
    plus the Python/Torch/CUDA runtime that actually rendered it.
    """

    validate_prelaunch_source_provenance(provenance, require_clean_source=require_clean_source)

    rambo = provenance["rambo"]
    _require(
        rambo.get("runtime_module_path") == rambo.get("module_path"),
        "RAMBO module path changed after pre-AppLauncher provenance collection",
    )
    _require(
        rambo.get("runtime_git_root") == rambo.get("git_root"),
        "RAMBO Git root changed after pre-AppLauncher provenance collection",
    )
    _require(
        rambo.get("runtime_git_head") == rambo.get("git_head"),
        "RAMBO Git HEAD changed after pre-AppLauncher provenance collection",
    )
    runtime_clean = rambo.get("runtime_worktree_clean")
    _require(isinstance(runtime_clean, bool), "RAMBO runtime clean flag must be bool")
    runtime_status = rambo.get("runtime_status_porcelain_v1")
    _require(
        isinstance(runtime_status, list) and all(isinstance(line, str) and line for line in runtime_status),
        "RAMBO runtime worktree status must be a porcelain list",
    )
    _require(
        runtime_clean is (len(runtime_status) == 0),
        "RAMBO runtime clean flag disagrees with runtime porcelain status",
    )
    if require_clean_source:
        _require(runtime_clean, "RAMBO worktree was not clean at runtime provenance collection")

    isaaclab = provenance.get("isaaclab")
    _require(isinstance(isaaclab, dict), "provenance.isaaclab is required")
    _require_absolute_path(isaaclab.get("module_path"), "provenance.isaaclab.module_path")
    _require_absolute_path(isaaclab.get("git_root"), "provenance.isaaclab.git_root")
    _require_git_head(isaaclab.get("git_head"), "provenance.isaaclab.git_head")
    _require(
        isaaclab.get("git_head") == TARGET_ISAACLAB_COMMIT,
        "Isaac Lab Git HEAD is not the pinned target",
    )
    _require(
        isaaclab.get("git_exact_tag") == TARGET_ISAACLAB_TAG,
        "Isaac Lab exact tag is not the pinned target",
    )
    _require_non_empty_string(isaaclab.get("package_version"), "provenance.isaaclab.package_version")
    _require_non_empty_string(isaaclab.get("module_version"), "provenance.isaaclab.module_version")

    isaacsim = provenance.get("isaacsim")
    _require(isinstance(isaacsim, dict), "provenance.isaacsim is required")
    _require(
        isaacsim.get("package_version") == TARGET_ISAACSIM_VERSION,
        "Isaac Sim package version is not the pinned target",
    )

    runtime = provenance.get("runtime")
    _require(isinstance(runtime, dict), "provenance.runtime is required")
    for key in ("python_version", "platform", "torch_version", "cuda_runtime", "cuda_device"):
        _require_non_empty_string(runtime.get(key), f"provenance.runtime.{key}")
    _require_absolute_path(runtime.get("python_executable"), "provenance.runtime.python_executable")


def _validate_physx_evidence(evidence: Any, label: str) -> dict[str, Any]:
    """Require the configured and live backend identities in a final artifact."""

    _require(isinstance(evidence, dict), f"{label} PhysX evidence is required")
    _require(evidence.get("requested_cfg") == PHYSX_CFG_FQN, f"{label} physics config is not PhysxCfg")
    _require(evidence.get("actual_manager") == PHYSX_MANAGER_FQN, f"{label} physics manager is not PhysxManager")
    _require(
        evidence.get("configured_manager") in {PHYSX_MANAGER_FQN, PHYSX_MANAGER_FACTORY_FQN},
        f"{label} configured manager is not PhysxManager",
    )
    _require(evidence.get("use_newton_actuators") is False, f"{label} does not disable Newton actuators")
    return evidence


def _artifact_path(root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path is required")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ThirdPersonDiagnosticError(f"{label} path escapes the artifact root") from exc
    return candidate


def final_capture_timing(
    *,
    requested_steps: int,
    rollout: dict[str, Any],
    rgb: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless one complete 30-second front-camera rollout exists.

    The diagnostic is intentionally restricted to the migration's fixed M7/M9
    contract.  It is not a generic screenshot facility: every image must refer
    to the same completed 3,000-action checkpoint policy rollout.
    """

    _require(requested_steps == FINAL_ACTION_STEPS, "final diagnostic requires exactly 3000 action steps")
    _require(isinstance(rollout, dict), "rollout evidence is required")
    _require(isinstance(rgb, dict), "RGB evidence is required")
    _require(_as_int(rollout.get("steps"), "rollout.steps") == FINAL_ACTION_STEPS, "rollout steps disagree")

    policy_timestep_s = _as_finite(rollout.get("policy_timestep_s"), "rollout.policy_timestep_s")
    physics_timestep_s = _as_finite(rollout.get("physics_timestep_s"), "rollout.physics_timestep_s")
    _require(
        math.isclose(policy_timestep_s, 0.01, abs_tol=1.0e-9, rel_tol=0.0),
        "final diagnostic requires the 100 Hz policy timestep",
    )
    _require(
        math.isclose(physics_timestep_s, 0.002, abs_tol=1.0e-9, rel_tol=0.0),
        "final diagnostic requires the 500 Hz physics timestep",
    )
    ticks_per_action = round(policy_timestep_s / physics_timestep_s)
    _require(
        ticks_per_action == 5
        and math.isclose(ticks_per_action * physics_timestep_s, policy_timestep_s, abs_tol=1.0e-9, rel_tol=0.0),
        "policy and physics timesteps do not form the required five-tick action cadence",
    )
    physics_ticks = requested_steps * ticks_per_action
    _require(physics_ticks == FINAL_PHYSICS_TICKS, "final diagnostic physics tick count is invalid")

    expected_rollout_frames = _as_int(
        rollout.get("expected_camera_frame_count"), "rollout.expected_camera_frame_count"
    )
    actual_rgb_frames = _as_int(rgb.get("frame_count"), "rgb.frame_count")
    expected_rgb_frames = _as_int(rgb.get("expected_frame_count"), "rgb.expected_frame_count")
    _require(rgb.get("passed") is True, "front RGB evidence did not pass")
    _require(
        expected_rollout_frames == expected_rgb_frames == actual_rgb_frames == FINAL_FRONT_RGB_FRAMES,
        "final diagnostic requires exactly 375 fresh front RGB frames",
    )

    return {
        "requested_action_steps": requested_steps,
        "policy_timestep_s": policy_timestep_s,
        "physics_timestep_s": physics_timestep_s,
        "physics_ticks_per_action": ticks_per_action,
        "physics_ticks": physics_ticks,
        "front_rgb_frames": actual_rgb_frames,
        "front_rgb_cadence_s": 0.08,
        "final_timestamp_s": requested_steps * policy_timestep_s,
    }


def validate_final_capture_record(record: dict[str, Any], timing: dict[str, Any]) -> None:
    """Validate public RGBD/targeting facts for the final third-person image."""

    _require(isinstance(record, dict), "third-person capture record is required")
    _require(
        _as_int(record.get("action_step"), "third_person.action_step")
        == _as_int(timing.get("requested_action_steps"), "timing.requested_action_steps"),
        "third-person image was not captured after the final policy action",
    )
    _require(
        _as_int(record.get("physics_ticks"), "third_person.physics_ticks")
        == _as_int(timing.get("physics_ticks"), "timing.physics_ticks"),
        "third-person image does not reference the final physics tick",
    )
    _require(
        math.isclose(
            _as_finite(record.get("timestamp_s"), "third_person.timestamp_s"),
            _as_finite(timing.get("final_timestamp_s"), "timing.final_timestamp_s"),
            abs_tol=1.0e-9,
            rel_tol=0.0,
        ),
        "third-person timestamp is not aligned to the final action",
    )

    rgb = record.get("rgb")
    depth = record.get("distance_to_image_plane")
    _require(isinstance(rgb, dict), "third-person RGB metadata is required")
    _require(isinstance(depth, dict), "third-person depth metadata is required")
    _require(rgb.get("shape") == [1, 480, 640, 3], "third-person RGB shape must be 1x480x640x3")
    _require(depth.get("shape") == [1, 480, 640, 1], "third-person depth shape must be 1x480x640x1")
    _require(_as_finite(rgb.get("mean"), "third_person.rgb.mean") > 2.0, "third-person RGB is black")
    _require(_as_finite(rgb.get("std"), "third_person.rgb.std") > 1.0, "third-person RGB lacks variation")
    _require(
        _as_finite(depth.get("positive_fraction"), "third_person.depth.positive_fraction") > 0.01,
        "third-person depth lacks visible scene geometry",
    )
    _require(record.get("camera_target_is_final_robot_root") is True, "camera was not targeted at final robot root")
    _require(
        record.get("final_state_preserved_during_sensor_initialization") is True,
        "sensor initialization changed final robot state",
    )
    expected_tick_counter = [_as_int(timing.get("physics_ticks"), "timing.physics_ticks")]
    _require(
        record.get("front_camera_ticks_before_sensor_initialization") == expected_tick_counter,
        "production front-camera counter was incomplete before sensor initialization",
    )
    _require(
        record.get("front_camera_ticks_after_sensor_initialization") == expected_tick_counter,
        "production front-camera counter changed during sensor initialization",
    )
    _require(
        record.get("front_camera_counter_preserved_during_sensor_initialization") is True,
        "sensor initialization reset the production front-camera counter",
    )
    _require(record.get("robot_and_scene_visual_review_required") is True, "manual visual review marker is required")


def build_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, self-contained manifest for an opt-in diagnostic artifact."""

    provenance = summary.get("provenance")
    # A 3,000-step third-person image is final migration evidence only when it
    # was captured from a clean RAMBO source tree.  Keep this strict even if a
    # future caller adds looser diagnostic-only collection elsewhere.
    validate_runtime_provenance(provenance, require_clean_source=True)
    diagnostic = summary.get("third_person_diagnostic")
    _require(isinstance(diagnostic, dict), "summary lacks third-person diagnostic evidence")
    _require(
        diagnostic.get("production_front_camera_contract_unchanged") is True,
        "third-person diagnostic must not modify the production front-camera contract",
    )
    timing = diagnostic.get("timing")
    _require(isinstance(timing, dict), "third-person diagnostic timing is required")
    validate_final_capture_record(diagnostic.get("capture"), timing)
    _require(summary.get("passed") is True, "only a passed validation can produce final diagnostic evidence")
    backend_before = _validate_physx_evidence(summary.get("backend_before"), "backend_before")
    backend_after_rollout = _validate_physx_evidence(
        summary.get("backend_after_rollout"), "backend_after_rollout"
    )
    backend_before_sensor_initialization = _validate_physx_evidence(
        diagnostic.get("backend_before_sensor_initialization"), "backend_before_sensor_initialization"
    )
    backend_after_capture = _validate_physx_evidence(
        diagnostic.get("backend_after_capture"), "backend_after_capture"
    )
    backend_after = _validate_physx_evidence(summary.get("backend_after"), "backend_after")

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "task": summary.get("task"),
        "checkpoint": summary.get("checkpoint"),
        "provenance": provenance,
        "backend_before": backend_before,
        "backend_after_rollout": backend_after_rollout,
        "backend_before_sensor_initialization": backend_before_sensor_initialization,
        "backend_after_capture": backend_after_capture,
        "backend_after": backend_after,
        "validation_only_overrides": summary.get("validation_only_overrides"),
        "production_front_camera_contract_unchanged": diagnostic["production_front_camera_contract_unchanged"],
        "timing": timing,
        "capture": diagnostic["capture"],
        "files": {
            "summary": "summary.json",
            "front_rgb_directory": "rgb",
            "front_rgb_timestamps": "rgb_timestamps.npy",
            "front_rgb_frame_ids": "rgb_frame_ids.npy",
            "third_person_rgb": "third_person_final_robot_scene_rgb.png",
            "third_person_depth": "third_person_final_robot_scene_distance_to_image_plane.npy",
            "third_person_manifest": MANIFEST_FILENAME,
            "checksum_manifest": CHECKSUMS_FILENAME,
            "process_exit": PROCESS_EXIT_FILENAME,
        },
    }


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    """Write stable UTF-8 JSON for an artifact sidecar."""

    target = Path(path)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ThirdPersonDiagnosticError(f"{label} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ThirdPersonDiagnosticError(f"{label} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def sha256_file(path: str | Path) -> str:
    """Hash one artifact file without loading its whole content into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(output_dir: str | Path) -> Path:
    """Checksum every evidence file exactly once, excluding the manifest itself."""

    root = Path(output_dir)
    checksum_path = root / CHECKSUMS_FILENAME
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files), encoding="utf-8"
    )
    return checksum_path


def verify_checksum_manifest(output_dir: str | Path) -> int:
    """Verify that a checksum manifest covers exactly all artifact evidence files."""

    root = Path(output_dir)
    checksum_path = root / CHECKSUMS_FILENAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ThirdPersonDiagnosticError("checksum manifest is missing") from exc
    _require(lines, "checksum manifest is empty")

    listed: set[Path] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ThirdPersonDiagnosticError(f"malformed checksum line {line_number}") from exc
        _require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), "invalid checksum")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ThirdPersonDiagnosticError("checksum path escapes artifact root") from exc
        _require(path.is_file(), f"checksum target is missing: {relative}")
        _require(path not in listed, f"duplicate checksum path: {relative}")
        _require(sha256_file(path) == digest, f"checksum mismatch: {relative}")
        listed.add(path)

    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != checksum_path.resolve()
    }
    _require(listed == actual, "checksum manifest does not cover exactly all evidence files")
    return len(listed)


def validate_final_artifact(output_dir: str | Path) -> dict[str, Any]:
    """Validate the complete, post-close M7/M9 final artifact offline.

    Unlike a generic checksum check, this requires every final-contract file,
    all 375 production RGB frames, exact PhysX evidence, and the parent-shell
    exit-status sidecar written after the Kit child exits.
    """

    root = Path(output_dir).resolve()
    _require(root.is_dir(), "final artifact directory is missing")
    manifest_path = root / MANIFEST_FILENAME
    summary_path = root / "summary.json"
    manifest = _read_json(manifest_path, label="final third-person manifest")
    summary = _read_json(summary_path, label="final third-person summary")
    _require(summary.get("passed") is True, "final third-person summary did not pass before process exit")
    expected_manifest = build_manifest(summary)
    for key in (
        "schema_version",
        "artifact_kind",
        "task",
        "checkpoint",
        "provenance",
        "backend_before",
        "backend_after_rollout",
        "backend_before_sensor_initialization",
        "backend_after_capture",
        "backend_after",
        "timing",
        "capture",
    ):
        _require(manifest.get(key) == expected_manifest.get(key), f"final third-person manifest field disagrees: {key}")

    files = manifest.get("files")
    _require(isinstance(files, dict), "final third-person manifest files are required")
    for name in (
        "summary",
        "front_rgb_directory",
        "front_rgb_timestamps",
        "front_rgb_frame_ids",
        "third_person_rgb",
        "third_person_depth",
        "third_person_manifest",
        "checksum_manifest",
        "process_exit",
    ):
        _require(files.get(name) == expected_manifest["files"][name], f"final third-person file mapping is invalid: {name}")

    expected_file_paths = {
        "summary": _artifact_path(root, files["summary"], "summary"),
        "front_rgb_timestamps": _artifact_path(root, files["front_rgb_timestamps"], "front RGB timestamps"),
        "front_rgb_frame_ids": _artifact_path(root, files["front_rgb_frame_ids"], "front RGB frame ids"),
        "third_person_rgb": _artifact_path(root, files["third_person_rgb"], "third-person RGB"),
        "third_person_depth": _artifact_path(root, files["third_person_depth"], "third-person depth"),
        "third_person_manifest": _artifact_path(root, files["third_person_manifest"], "third-person manifest"),
        "checksum_manifest": _artifact_path(root, files["checksum_manifest"], "checksum manifest"),
        "process_exit": _artifact_path(root, files["process_exit"], "process exit"),
    }
    for label, path in expected_file_paths.items():
        _require(path.is_file(), f"final third-person evidence file is missing: {label}")
    rgb_dir = _artifact_path(root, files["front_rgb_directory"], "front RGB directory")
    _require(rgb_dir.is_dir(), "final third-person front RGB directory is missing")
    rgb_frames = sorted(path for path in rgb_dir.glob("*.png") if path.is_file())
    _require(
        len(rgb_frames) == FINAL_FRONT_RGB_FRAMES,
        f"final third-person artifact requires exactly {FINAL_FRONT_RGB_FRAMES} front RGB PNG files",
    )

    process_exit = _read_json(expected_file_paths["process_exit"], label="final process exit")
    _require(process_exit.get("shell_exit_status") == 0, "final runtime process exit status is not zero")
    _require(process_exit.get("exit_status_zero") is True, "final runtime exit sidecar does not confirm zero")
    _require(process_exit.get("acceptance_passed") is True, "final runtime exit sidecar does not accept the artifact")
    _require(manifest.get("process_exit") == process_exit, "manifest process-exit evidence disagrees with sidecar")
    checksummed_files = verify_checksum_manifest(root)
    return {
        "artifact_dir": str(root),
        "front_rgb_frame_count": len(rgb_frames),
        "checksummed_files": checksummed_files,
        "process_exit_status": process_exit["shell_exit_status"],
    }


__all__ = (
    "ARTIFACT_KIND",
    "CHECKSUMS_FILENAME",
    "FINAL_ACTION_STEPS",
    "FINAL_FRONT_RGB_FRAMES",
    "FINAL_PHYSICS_TICKS",
    "MANIFEST_FILENAME",
    "PROCESS_EXIT_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TARGET_ISAACLAB_COMMIT",
    "TARGET_ISAACLAB_TAG",
    "TARGET_ISAACSIM_VERSION",
    "ThirdPersonDiagnosticError",
    "build_manifest",
    "final_capture_timing",
    "sha256_file",
    "validate_final_capture_record",
    "validate_final_artifact",
    "validate_prelaunch_source_provenance",
    "validate_runtime_provenance",
    "verify_checksum_manifest",
    "write_checksum_manifest",
    "write_json",
)
