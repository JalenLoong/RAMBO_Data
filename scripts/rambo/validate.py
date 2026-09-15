#!/usr/bin/env python3
"""Validate a RAMBO checkpoint with a finite RGB-enabled PhysX rollout."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any


class RuntimeProvenanceCollectionError(RuntimeError):
    """Raised when final third-person evidence cannot identify its execution."""


def _git_output(cwd: Path, *arguments: str) -> str:
    """Read one Git fact with a bounded, side-effect-free subprocess."""

    try:
        process = subprocess.run(
            ["git", "-C", str(cwd), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeProvenanceCollectionError(f"Cannot inspect Git provenance in {cwd}") from exc
    if process.returncode != 0:
        message = process.stderr.strip() or process.stdout.strip() or "unknown Git error"
        raise RuntimeProvenanceCollectionError(f"Cannot inspect Git provenance in {cwd}: {message}")
    return process.stdout.rstrip("\n")


def _provenance_command() -> list[str]:
    """Record the exact Python process command without attempting shell reconstruction."""

    return [
        str(Path(sys.executable).resolve()),
        str(Path(sys.argv[0]).resolve()),
        *sys.argv[1:],
    ]


def _collect_pre_app_launcher_provenance() -> dict[str, Any]:
    """Capture the actual imported RAMBO source state before ``AppLauncher`` starts Kit."""

    launcher_hint = Path(__file__).resolve().parents[2]
    launcher_root = Path(_git_output(launcher_hint, "rev-parse", "--show-toplevel")).resolve()
    try:
        import rambo
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime installation guard.
        raise RuntimeProvenanceCollectionError("Final third-person provenance cannot import RAMBO") from exc
    rambo_module_path = Path(rambo.__file__).resolve()
    rambo_root = Path(_git_output(rambo_module_path.parent, "rev-parse", "--show-toplevel")).resolve()
    if rambo_root != launcher_root:
        raise RuntimeProvenanceCollectionError(
            "Imported RAMBO module is not from this validate.py checkout: "
            f"module_root={rambo_root}, launcher_root={launcher_root}"
        )
    status_lines = _git_output(
        rambo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "schema_version": 1,
        "collection_phase": "pre_app_launcher_source_then_runtime",
        "command": _provenance_command(),
        "rambo": {
            "module_path": str(rambo_module_path),
            "git_root": str(rambo_root.resolve()),
            "git_head": _git_output(rambo_root, "rev-parse", "HEAD"),
            "pre_run_worktree_clean": not status_lines,
            "pre_run_status_porcelain_v1": status_lines,
        },
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeProvenanceCollectionError(
            f"Required package metadata is unavailable for {distribution!r}"
        ) from exc


def _complete_runtime_provenance(prelaunch: dict[str, Any]) -> dict[str, Any]:
    """Add post-Kit runtime identities to a source state captured pre-launch."""

    try:
        import isaaclab
        import rambo
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - requires target Kit runtime.
        raise RuntimeProvenanceCollectionError("Pinned Isaac runtime is unavailable after AppLauncher") from exc

    if not torch.cuda.is_available():
        raise RuntimeProvenanceCollectionError("Final third-person provenance requires an active CUDA runtime")
    isaaclab_module_path = Path(isaaclab.__file__).resolve()
    isaaclab_root = Path(_git_output(isaaclab_module_path.parent, "rev-parse", "--show-toplevel"))
    rambo_module_path = Path(rambo.__file__).resolve()
    rambo_root = Path(_git_output(rambo_module_path.parent, "rev-parse", "--show-toplevel")).resolve()
    prelaunch_rambo = prelaunch.get("rambo")
    if not isinstance(prelaunch_rambo, dict):
        raise RuntimeProvenanceCollectionError("Final third-person prelaunch RAMBO provenance is missing")
    if (
        str(rambo_module_path) != prelaunch_rambo.get("module_path")
        or str(rambo_root) != prelaunch_rambo.get("git_root")
        or _git_output(rambo_root, "rev-parse", "HEAD") != prelaunch_rambo.get("git_head")
    ):
        raise RuntimeProvenanceCollectionError(
            "RAMBO module provenance changed between pre-AppLauncher and runtime collection"
        )
    rambo_runtime_status = _git_output(
        rambo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    if rambo_runtime_status:
        raise RuntimeProvenanceCollectionError(
            "Final third-person provenance requires the RAMBO worktree to remain clean through runtime collection"
        )
    isaaclab_module_version = getattr(isaaclab, "__version__", None)
    if not isinstance(isaaclab_module_version, str) or not isaaclab_module_version:
        raise RuntimeProvenanceCollectionError("Isaac Lab module does not expose an exact version")
    cuda_runtime = torch.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise RuntimeProvenanceCollectionError("Torch does not expose an exact CUDA runtime version")

    provenance = dict(prelaunch)
    provenance.update(
        {
            "rambo": {
                **prelaunch_rambo,
                "runtime_module_path": str(rambo_module_path),
                "runtime_git_root": str(rambo_root),
                "runtime_git_head": _git_output(rambo_root, "rev-parse", "HEAD"),
                "runtime_worktree_clean": True,
                "runtime_status_porcelain_v1": rambo_runtime_status,
            },
            "isaaclab": {
                "module_path": str(isaaclab_module_path),
                "package_version": _package_version("isaaclab"),
                "module_version": isaaclab_module_version,
                "git_root": str(isaaclab_root.resolve()),
                "git_head": _git_output(isaaclab_root, "rev-parse", "HEAD"),
                "git_exact_tag": _git_output(isaaclab_root, "describe", "--tags", "--exact-match"),
            },
            "isaacsim": {
                "package_version": _package_version("isaacsim"),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "platform": platform.platform(),
                "torch_version": str(torch.__version__),
                "cuda_runtime": cuda_runtime,
                "cuda_device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
            },
        }
    )
    return provenance


def _contact_schedule_durations(env_cfg: Any) -> dict[str, float]:
    """Return copied-config contact-program coverage for validation provenance."""

    config = getattr(env_cfg, "contact_generator_config", None)
    sequences = config.get("contact_sequence") if isinstance(config, dict) else None
    if not isinstance(sequences, dict):
        return {}
    durations: dict[str, float] = {}
    for foot, sequence in sequences.items():
        if not isinstance(sequence, list):
            continue
        try:
            durations[str(foot)] = sum(float(segment[1]) for segment in sequence)
        except (IndexError, TypeError, ValueError):
            continue
    return durations


def _validation_only_overrides(env_cfg: Any) -> dict[str, Any]:
    """Snapshot the validation clone's horizon and contact schedule overrides."""

    return {
        "scope": "validation-only copied environment configuration; production registry is unchanged",
        "episode_length_s": float(getattr(env_cfg, "episode_length_s")),
        "contact_schedule_duration_s": _contact_schedule_durations(env_cfg),
    }


def _build_parser() -> tuple[argparse.ArgumentParser, type]:
    """Build the CLI parser while keeping source imports simulator-independent."""

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover - target-runtime guard.
        raise RuntimeError(
            "scripts/rambo/validate.py must be run inside the RAMBO Isaac Lab environment"
        ) from exc

    parser = argparse.ArgumentParser(description="Validate a RAMBO CRL2 policy checkpoint.")
    parser.add_argument("--task", choices=(
        "Isaac-RAMBO-Quadruped-Go2-v0",
    ), required=True, help="Registered RAMBO Gym task.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local model_*.pt path.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed validation seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="100 Hz policy steps; 3000 is the 30-second acceptance run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New empty directory for summary.json and RGB artifacts.",
    )
    parser.add_argument(
        "--third-person-diagnostic",
        choices=("final",),
        default=None,
        help=(
            "After a passed 3000-step/375-frame checkpoint rollout, render one independent final "
            "third-person RGBD evidence capture. Disabled by default."
        ),
    )
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
        help="Use USD I/O instead of Fabric.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser, AppLauncher


def _runtime_imports() -> dict[str, Any]:
    """Import simulator dependencies after AppLauncher has initialized Isaac Sim."""

    from rambo.torch_runtime import ensure_cuda_linalg_loaded

    ensure_cuda_linalg_loaded()
    import gymnasium as gym
    import rambo
    from crl2.algorithms import PPO
    from rambo.rl import Crl2VecEnvWrapper
    from rambo.utils.physx import assert_physx_environment, configure_physx
    from rambo.utils.registry import load_cfg_from_registry, parse_env_cfg
    from rambo.validation.checkpoints import (
        contract_for_task,
        load_verified_checkpoint,
        restore_runner,
    )
    from rambo.validation.rollout import (
        RgbFrameRecorder,
        RolloutValidationError,
        configure_validation_cfg,
        prepare_output_dir,
        run_policy_rollout,
        runtime_metadata,
        seed_everything,
        validate_environment_contract,
        write_summary,
    )
    from rambo.validation.third_person_diagnostic import (
        MANIFEST_FILENAME,
        build_manifest,
        final_capture_timing,
        validate_final_capture_record,
        validate_runtime_provenance,
        write_checksum_manifest,
        write_json,
    )

    rambo.register_tasks()
    return {
        "gym": gym,
        "PPO": PPO,
        "Crl2VecEnvWrapper": Crl2VecEnvWrapper,
        "RgbFrameRecorder": RgbFrameRecorder,
        "RolloutValidationError": RolloutValidationError,
        "assert_physx_environment": assert_physx_environment,
        "configure_physx": configure_physx,
        "configure_validation_cfg": configure_validation_cfg,
        "contract_for_task": contract_for_task,
        "load_cfg_from_registry": load_cfg_from_registry,
        "load_verified_checkpoint": load_verified_checkpoint,
        "parse_env_cfg": parse_env_cfg,
        "prepare_output_dir": prepare_output_dir,
        "restore_runner": restore_runner,
        "run_policy_rollout": run_policy_rollout,
        "runtime_metadata": runtime_metadata,
        "seed_everything": seed_everything,
        "validate_environment_contract": validate_environment_contract,
        "MANIFEST_FILENAME": MANIFEST_FILENAME,
        "build_manifest": build_manifest,
        "final_capture_timing": final_capture_timing,
        "validate_final_capture_record": validate_final_capture_record,
        "validate_runtime_provenance": validate_runtime_provenance,
        "write_checksum_manifest": write_checksum_manifest,
        "write_json": write_json,
        "write_summary": write_summary,
    }


def _print_failure(prefix: str, error: BaseException) -> None:
    """Print a failure while preserving the underlying traceback."""

    print(prefix, file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    sys.stderr.flush()


def _require_proxy_torch(value: Any, label: str) -> Any:
    """Require Isaac Lab 3's public ProxyArray-to-Torch bridge."""

    tensor = getattr(value, "torch", None)
    if tensor is None:
        raise RuntimeError(f"{label} is not an Isaac Lab ProxyArray with public .torch access")
    return tensor


def _capture_final_third_person_diagnostic(
    env: Any,
    *,
    timing: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Capture final RGBD evidence without changing the completed policy rollout.

    This function is called only after the front-camera recorder has accepted
    all 375 frames.  The independent camera is therefore absent from the
    policy loop and cannot alter the production front-camera cadence, actions,
    observations, checkpoint restoration, or the completed 15,000 physics
    ticks.  ``sim.reset()`` below is the public sensor-initialization lifecycle
    hook, not an environment reset; final robot root state is snapshotted
    before and after it and must remain unchanged.
    """

    import imageio.v3 as iio
    import numpy as np
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    base_env = env.unwrapped
    front_camera = getattr(base_env, "front_camera", None)
    if front_camera is None:
        raise RuntimeError("RAMBO task did not expose public front_camera for final diagnostic")
    front_ticks = _require_proxy_torch(
        getattr(front_camera, "rambo_physics_ticks", None), "front_camera.rambo_physics_ticks"
    ).detach().cpu().tolist()
    expected_ticks = int(timing["physics_ticks"])
    if front_ticks != [expected_ticks]:
        raise RuntimeError(
            "Final third-person capture requires the completed front-camera tick counter: "
            f"got {front_ticks}, expected {[expected_ticks]}"
        )

    def robot_root_position() -> Any:
        return _require_proxy_torch(
            base_env._robot.data.root_link_pos_w, "robot.data.root_link_pos_w"
        )[0:1].detach().clone()

    root_before_sensor_initialization = robot_root_position()
    diagnostic_camera = None
    try:
        # This camera is deliberately independent of RAMBO's base-mounted front
        # camera.  Its world-space view targets the final public robot root so
        # the persisted image can be reviewed for both robot and scene.
        diagnostic_camera = Camera(
            CameraCfg(
                prim_path="/World/RAMBOValidation/FinalThirdPersonCamera",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb", "distance_to_image_plane"],
                update_latest_camera_pose=True,
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 30.0),
                ),
            )
        )
        # Directly constructed Isaac Lab sensors receive their public
        # PHYSICS_READY lifecycle callback on SimulationContext.reset().  No
        # environment reset or physics action occurs after the rollout.
        base_env.sim.reset()
        if not diagnostic_camera.is_initialized:
            raise RuntimeError("Final third-person Camera did not initialize through the public lifecycle")

        root_after_sensor_initialization = robot_root_position()
        front_ticks_after_sensor_initialization = _require_proxy_torch(
            getattr(front_camera, "rambo_physics_ticks", None), "front_camera.rambo_physics_ticks after sensor init"
        ).detach().cpu().tolist()
        if front_ticks_after_sensor_initialization != [expected_ticks]:
            raise RuntimeError(
                "Diagnostic sensor initialization reset the production front-camera tick counter: "
                f"got {front_ticks_after_sensor_initialization}, expected {[expected_ticks]}"
            )
        state_preserved = bool(
            torch.allclose(
                root_before_sensor_initialization,
                root_after_sensor_initialization,
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        if not state_preserved:
            raise RuntimeError("Final robot root changed while initializing the diagnostic sensor")

        target = root_after_sensor_initialization
        eye_offset = torch.tensor([[2.2, 2.2, 1.5]], dtype=target.dtype, device=target.device)
        eye = target + eye_offset
        diagnostic_camera.set_world_poses_from_view(eyes=eye, targets=target)
        base_env.sim.render()
        diagnostic_camera.update(dt=float(base_env.physics_dt), force_recompute=True)
        data = diagnostic_camera.data
        output = data.output
        if not isinstance(output, dict):
            raise RuntimeError("Final third-person camera output is not a dictionary")
        if "rgb" not in output or "distance_to_image_plane" not in output:
            raise RuntimeError(f"Final third-person RGBD outputs are incomplete: {sorted(output)}")
        rgb = _require_proxy_torch(output["rgb"], "third_person_camera.data.output.rgb")
        depth = _require_proxy_torch(
            output["distance_to_image_plane"], "third_person_camera.data.output.distance_to_image_plane"
        )
        if tuple(rgb.shape) != (1, 480, 640, 3):
            raise RuntimeError(f"Unexpected final third-person RGB shape: {tuple(rgb.shape)}")
        if tuple(depth.shape) != (1, 480, 640, 1):
            raise RuntimeError(f"Unexpected final third-person depth shape: {tuple(depth.shape)}")
        if not bool(torch.isfinite(rgb).all()):
            raise RuntimeError("Final third-person RGB contains NaN or Inf")

        rgb_float = rgb.float()
        rgb_mean = float(rgb_float.mean().item())
        rgb_std = float(rgb_float.std().item())
        if rgb_mean <= 2.0 or rgb_std <= 1.0:
            raise RuntimeError(f"Final third-person RGB is black or lacks variation: mean={rgb_mean}, std={rgb_std}")
        finite_depth = torch.isfinite(depth)
        positive_depth = finite_depth & (depth > 0.0)
        finite_values = depth[finite_depth]
        if finite_values.numel() == 0:
            raise RuntimeError("Final third-person depth has no finite values")
        finite_fraction = float(finite_depth.float().mean().item())
        positive_fraction = float(positive_depth.float().mean().item())
        if positive_fraction <= 0.01:
            raise RuntimeError(
                "Final third-person depth has no meaningful scene geometry: "
                f"finite_fraction={finite_fraction}, positive_fraction={positive_fraction}"
            )

        rgb_path = output_dir / "third_person_final_robot_scene_rgb.png"
        depth_path = output_dir / "third_person_final_robot_scene_distance_to_image_plane.npy"
        iio.imwrite(rgb_path, rgb[0].detach().cpu().numpy())
        np.save(depth_path, depth[0].detach().cpu().numpy())
        frame_id = int(
            _require_proxy_torch(diagnostic_camera.frame, "third_person_camera.frame")
            .detach()
            .reshape(-1)[0]
            .item()
        )

        return {
            "capture_phase": "after_passed_front_rollout_and_rgb_finalization",
            "action_step": int(timing["requested_action_steps"]),
            "physics_ticks": expected_ticks,
            "timestamp_s": float(timing["final_timestamp_s"]),
            "front_camera_ticks_before_sensor_initialization": front_ticks,
            "front_camera_ticks_after_sensor_initialization": front_ticks_after_sensor_initialization,
            "front_camera_counter_preserved_during_sensor_initialization": True,
            "camera_frame_id": frame_id,
            "camera_class": f"{type(diagnostic_camera).__module__}.{type(diagnostic_camera).__qualname__}",
            "renderer_class": (
                f"{type(diagnostic_camera.cfg.renderer_cfg).__module__}."
                f"{type(diagnostic_camera.cfg.renderer_cfg).__qualname__}"
            ),
            "camera_target_is_final_robot_root": True,
            "final_state_preserved_during_sensor_initialization": state_preserved,
            "robot_and_scene_visual_review_required": True,
            "robot_and_scene_visibility_evidence": {
                "robot": "World-space target equals final public robot root position.",
                "scene": "Non-black RGB and positive depth geometry are required; inspect persisted PNG.",
            },
            "camera_target_robot_root_w": target.detach().cpu().tolist(),
            "camera_eye_w": eye.detach().cpu().tolist(),
            "robot_root_before_sensor_initialization_w": root_before_sensor_initialization.detach().cpu().tolist(),
            "robot_root_after_sensor_initialization_w": root_after_sensor_initialization.detach().cpu().tolist(),
            "camera_pose_and_intrinsics": {
                "position_w": _require_proxy_torch(data.pos_w, "third_person_camera.data.pos_w")
                .detach()
                .cpu()
                .tolist(),
                "orientation_w_xyzw": _require_proxy_torch(
                    data.quat_w_world, "third_person_camera.data.quat_w_world"
                )
                .detach()
                .cpu()
                .tolist(),
                "intrinsic_matrices": _require_proxy_torch(
                    data.intrinsic_matrices, "third_person_camera.data.intrinsic_matrices"
                )
                .detach()
                .cpu()
                .tolist(),
            },
            "rgb": {
                "shape": list(rgb.shape),
                "dtype": str(rgb.dtype),
                "channel_order": "RGB",
                "mean": rgb_mean,
                "std": rgb_std,
                "minimum": float(rgb_float.min().item()),
                "maximum": float(rgb_float.max().item()),
                "artifact": rgb_path.name,
            },
            "distance_to_image_plane": {
                "shape": list(depth.shape),
                "dtype": str(depth.dtype),
                "finite_fraction": finite_fraction,
                "positive_fraction": positive_fraction,
                "finite_minimum": float(finite_values.min().item()),
                "finite_maximum": float(finite_values.max().item()),
                "artifact": depth_path.name,
            },
        }
    finally:
        if diagnostic_camera is not None:
            del diagnostic_camera


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args_cli = parser.parse_args()
    # The helper is simulator-safe and rejects every unaudited visualizer
    # before AppLauncher can create a Kit experience.
    from rambo.utils.physx import validate_rambo_visualizer_args

    visualizer_selection = validate_rambo_visualizer_args(parser, args_cli, sys.argv[1:])
    if args_cli.steps <= 0:
        parser.error("--steps must be positive")
    # The RAMBO camera contract is 0.08 s, i.e. one fresh RGB frame per eight
    # 100 Hz policy steps.  This makes 3000 steps exactly 375 frames.
    if args_cli.steps % 8:
        parser.error("--steps must be divisible by 8 for the 12.5 Hz RGB contract")
    if args_cli.third_person_diagnostic == "final" and args_cli.steps != 3000:
        parser.error("--third-person-diagnostic final requires exactly --steps 3000")
    args_cli.enable_cameras = True

    prelaunch_provenance: dict[str, Any] | None = None
    if args_cli.third_person_diagnostic == "final":
        # This import is intentionally pure Python.  The source cleanliness
        # gate must fail before AppLauncher can create a Kit process or start
        # any simulation work for an artifact that calls itself final.
        from rambo.validation.third_person_diagnostic import validate_prelaunch_source_provenance

        try:
            prelaunch_provenance = _collect_pre_app_launcher_provenance()
            validate_prelaunch_source_provenance(prelaunch_provenance, require_clean_source=True)
        except (RuntimeProvenanceCollectionError, RuntimeError) as exc:
            parser.error(str(exc))

    app_launcher = app_launcher_type(args_cli)
    simulation_app = app_launcher.app
    runtime: dict[str, Any] | None = None
    env = None
    output_dir = None
    rgb_recorder = None
    rgb_metrics: dict[str, Any] | None = None
    rollout_metrics: dict[str, Any] | None = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "task": args_cli.task,
        "seed": args_cli.seed,
        "requested_steps": args_cli.steps,
        "viz": visualizer_selection,
        "eula_acceptance": "explicit_user_consent",
    }
    if args_cli.third_person_diagnostic is not None:
        summary["third_person_diagnostic_requested"] = args_cli.third_person_diagnostic
    captured_error: BaseException | None = None
    secondary_errors: list[tuple[str, BaseException]] = []

    try:
        runtime = _runtime_imports()
        output_dir = runtime["prepare_output_dir"](args_cli.output_dir)
        summary["runtime"] = runtime["runtime_metadata"]()
        if args_cli.third_person_diagnostic == "final":
            if prelaunch_provenance is None:  # Defensive: final mode must have passed the pre-launch gate.
                raise RuntimeProvenanceCollectionError("Final third-person provenance preflight is missing")
            summary["provenance"] = _complete_runtime_provenance(prelaunch_provenance)
            runtime["validate_runtime_provenance"](summary["provenance"], require_clean_source=True)

        contract = runtime["contract_for_task"](args_cli.task)
        checkpoint = runtime["load_verified_checkpoint"](args_cli.checkpoint, contract)
        summary["checkpoint"] = {
            "path": str(args_cli.checkpoint.expanduser().resolve()),
            "sha256": contract.sha256,
            "iteration": contract.iteration,
            "observation_dim": contract.observation_dim,
            "action_dim": contract.action_dim,
            "normalizer_count": contract.normalizer_count,
        }
        summary["mode"] = contract.mode

        parse_kwargs = {"num_envs": 1, "use_fabric": not args_cli.disable_fabric}
        device = getattr(args_cli, "device", None)
        if device is not None:
            parse_kwargs["device"] = device
        env_cfg = runtime["parse_env_cfg"](args_cli.task, **parse_kwargs)
        # Re-assign at this validation call site as well as in the registry:
        # a RAMBO validation must never rely on a future simulator default.
        runtime["configure_physx"](env_cfg)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args_cli.seed
        validation_config_before = _validation_only_overrides(env_cfg)
        runtime["configure_validation_cfg"](
            env_cfg,
            duration_s=31.0,
            enable_rgb_camera=True,
        )
        validation_config_after = _validation_only_overrides(env_cfg)
        summary["validation_only_overrides"] = {
            "scope": validation_config_after["scope"],
            "episode_length_s": {
                "before": validation_config_before["episode_length_s"],
                "after": validation_config_after["episode_length_s"],
            },
            "contact_schedule_duration_s": {
                "before": validation_config_before["contact_schedule_duration_s"],
                "after": validation_config_after["contact_schedule_duration_s"],
            },
        }
        summary["validation_config"] = {
            "episode_length_s": float(env_cfg.episode_length_s),
            "randomize_initial_state": bool(getattr(env_cfg, "randomize_initial_state", False)),
            "randomize_episode_progress": bool(getattr(env_cfg, "randomize_episode_progress", False)),
            "observation_noise": bool(getattr(env_cfg, "obs_noise", False)),
        }

        agent_cfg = runtime["load_cfg_from_registry"](args_cli.task, "crl2_cfg_entry_point")
        if not isinstance(agent_cfg, dict):
            raise runtime["RolloutValidationError"](
                "RAMBO CRL2 agent config must load as a YAML dictionary"
            )
        agent_cfg["seed"] = args_cli.seed
        agent_cfg["general"]["num_envs"] = 1

        env = runtime["Crl2VecEnvWrapper"](runtime["gym"].make(args_cli.task, cfg=env_cfg))
        summary["backend_before"] = runtime["assert_physx_environment"](env)
        runtime["seed_everything"](args_cli.seed, env)
        env.reset()
        runtime["validate_environment_contract"](env, contract)

        runner = runtime["PPO"](
            task=args_cli.task,
            env=env,
            agent_cfg=agent_cfg,
            train=False,
            device=env.device,
        )
        runtime["restore_runner"](runner, checkpoint, load_values=True, verify=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # The rollout helper independently verifies the actual camera period.
        rgb_recorder = runtime["RgbFrameRecorder"](
            output_dir=output_dir,
            expected_count=args_cli.steps // 8,
            width=640,
            height=480,
        )
        rollout_metrics = runtime["run_policy_rollout"](
            env,
            policy,
            contract,
            steps=args_cli.steps,
            rgb_recorder=rgb_recorder,
        )
        rgb_metrics = rgb_recorder.finalize()
        if not rgb_metrics["passed"]:
            raise runtime["RolloutValidationError"](
                "RGB validation failed: " + "; ".join(rgb_metrics["failures"])
            )
        summary["backend_after_rollout"] = runtime["assert_physx_environment"](env)
        if args_cli.third_person_diagnostic == "final":
            timing = runtime["final_capture_timing"](
                requested_steps=args_cli.steps,
                rollout=rollout_metrics,
                rgb=rgb_metrics,
            )
            diagnostic = {
                "mode": "final",
                "production_front_camera_contract_unchanged": True,
                "timing": timing,
                "backend_before_sensor_initialization": runtime["assert_physx_environment"](env),
            }
            capture = _capture_final_third_person_diagnostic(env, timing=timing, output_dir=output_dir)
            runtime["validate_final_capture_record"](capture, timing)
            diagnostic["capture"] = capture
            diagnostic["backend_after_capture"] = runtime["assert_physx_environment"](env)
            summary["third_person_diagnostic"] = diagnostic
        summary["backend_after"] = runtime["assert_physx_environment"](env)
    except BaseException as exc:
        captured_error = exc
        memory_monitor = getattr(exc, "memory_monitor", None)
        if isinstance(memory_monitor, dict):
            # Keep full samples even on a fail-closed threshold breach: they
            # are the evidence needed to distinguish a renderer/host leak
            # from a policy or physics failure on the next recovery attempt.
            summary["memory_monitor"] = memory_monitor
    finally:
        if rgb_recorder is not None and rgb_metrics is None:
            try:
                rgb_metrics = rgb_recorder.finalize()
            except BaseException as rgb_exc:  # Preserve the primary rollout failure when present.
                if captured_error is None:
                    captured_error = rgb_exc
                rgb_metrics = {
                    "passed": False,
                    "failures": [f"RGB finalization exception: {type(rgb_exc).__name__}: {rgb_exc}"],
                }

        if env is not None:
            try:
                env.close()
            except BaseException as close_exc:
                if captured_error is None:
                    captured_error = close_exc
                else:
                    secondary_errors.append(("environment close", close_exc))

        if captured_error is not None:
            summary["error"] = f"{type(captured_error).__name__}: {captured_error}"
        if rollout_metrics is not None:
            summary["rollout"] = rollout_metrics
        if rgb_metrics is not None:
            summary["rgb"] = rgb_metrics
        summary["passed"] = bool(
            captured_error is None
            and rollout_metrics is not None
            and rgb_metrics is not None
            and rgb_metrics.get("passed", False)
            and (
                args_cli.third_person_diagnostic is None
                or (
                    isinstance(summary.get("third_person_diagnostic"), dict)
                    and isinstance(summary.get("provenance"), dict)
                )
            )
        )

        if secondary_errors:
            summary["secondary_errors"] = [
                f"{stage}: {type(error).__name__}: {error}"
                for stage, error in secondary_errors
            ]

        if output_dir is not None and runtime is not None:
            try:
                diagnostic_manifest = None
                if args_cli.third_person_diagnostic == "final" and summary["passed"]:
                    summary["third_person_diagnostic_artifacts"] = {
                        "manifest": runtime["MANIFEST_FILENAME"],
                        "checksums": "checksums.sha256",
                    }
                    diagnostic_manifest = runtime["build_manifest"](summary)
                summary_path = runtime["write_summary"](output_dir, summary)
                if diagnostic_manifest is not None:
                    runtime["write_json"](output_dir / runtime["MANIFEST_FILENAME"], diagnostic_manifest)
                    runtime["write_checksum_manifest"](output_dir)
                print(f"SUMMARY_PATH={summary_path}", flush=True)
            except BaseException as summary_exc:
                if captured_error is None:
                    captured_error = summary_exc
                else:
                    secondary_errors.append(("summary write", summary_exc))
                # A missing or stale manifest must never leave a successful
                # summary behind.  Best-effort rewrite preserves the primary
                # diagnostic error even when the original writer failed.
                summary["passed"] = False
                summary["error"] = f"{type(captured_error).__name__}: {captured_error}"
                if secondary_errors:
                    summary["secondary_errors"] = [
                        f"{stage}: {type(error).__name__}: {error}" for stage, error in secondary_errors
                    ]
                try:
                    summary_path = runtime["write_summary"](output_dir, summary)
                    print(f"SUMMARY_PATH={summary_path}", flush=True)
                except BaseException as rewrite_exc:
                    secondary_errors.append(("failure summary rewrite", rewrite_exc))

    if captured_error is not None:
        _print_failure("RAMBO validation failed:", captured_error)
        for stage, error in secondary_errors:
            _print_failure(f"Additional {stage} failure:", error)
        exit_code = 1
    else:
        print("VALIDATION_SUCCESS", flush=True)
        exit_code = 0

    # All artifacts and the environment have been finalized.  Use the normal
    # Isaac Sim lifecycle on both success and failure; never bypass cleanup.
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
