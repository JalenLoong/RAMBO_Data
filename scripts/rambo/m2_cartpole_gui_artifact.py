"""Offline evidence helpers for the manual M2.3 Cartpole Kit GUI gate.

The module deliberately uses only the Python standard library.  It never
launches Kit, selects a simulator backend, or creates GUI input.  The runtime
recorder in :mod:`official_physx_smoke` supplies the Kit-live GPU samples; a
separate interactive command writes the named operator's post-close
attestation sidecar.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable


SCHEMA_VERSION = 1
ARTIFACT_KIND = "official_cartpole_direct_gui_observation"
RUNTIME_TASK = "Isaac-Cartpole-Direct-v0"
RUNTIME_SCENARIO = "cartpole-direct"
VISUALIZER = "kit"
PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"

MIN_OBSERVATION_SECONDS = 15.0
MAX_OBSERVATION_SECONDS = 300.0
MIN_GPU_SAMPLES = 2
TARGET_GPU_NAME_FRAGMENT = "RTX 4090"

# The pinned Isaac Sim 6.0.1 base experience enables this RTX Hydra render
# delegate.  ``omni.kit.renderer.core`` is plumbing shared by non-RTX render
# paths, so it is deliberately not evidence for this GUI gate.
RTX_RENDERER_EXTENSION_IDS = ("omni.hydra.rtx",)

ATTESTATION_FILENAME = "attestation.json"
CHECKSUMS_FILENAME = "checksums.sha256"

GUI_OBSERVATION_MODE = "bounded_manual_operator_observation"
TERMINAL_CONFIRMATION_PHRASE = "I OBSERVED THE M2.3 CARTPOLE KIT GUI"
OPERATOR_ATTESTATION_STATEMENT = (
    "I personally observed the visible Isaac Sim Kit GUI for the official "
    "Isaac-Cartpole-Direct-v0 gate. I confirmed the Kit window and viewport "
    "appeared normally, the timeline Play and Stop controls worked, no Vulkan "
    "swapchain error appeared, nvidia-smi showed Kit GPU memory, and the active "
    "renderer used the NVIDIA GeForce RTX 4090 rather than llvmpipe. I did not "
    "substitute logs or automated or synthetic UI evidence for my visual observation."
)
OBSERVATION_LIMITATION = (
    "This sidecar binds a sealed Kit runtime artifact to a named operator's "
    "interactive terminal self-attestation. It cannot independently prove that "
    "the visual observation occurred or independently determine every UI detail."
)


class M2CartpoleGuiArtifactError(RuntimeError):
    """Raised when an M2.3 GUI artifact is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2CartpoleGuiArtifactError(message)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise M2CartpoleGuiArtifactError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise M2CartpoleGuiArtifactError(f"{label} must be an integer") from error
    if result != value:
        raise M2CartpoleGuiArtifactError(f"{label} must be an exact integer")
    return result


def _as_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise M2CartpoleGuiArtifactError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise M2CartpoleGuiArtifactError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise M2CartpoleGuiArtifactError(f"{label} must be finite")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M2CartpoleGuiArtifactError(f"Cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root: Path) -> None:
    checksum_path = root / CHECKSUMS_FILENAME
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )


def _validate_checksums(root: Path) -> int:
    checksum_path = root / CHECKSUMS_FILENAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise M2CartpoleGuiArtifactError(f"Missing checksum manifest: {checksum_path}") from error
    _require(lines, "Checksum manifest is empty")
    listed: set[Path] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise M2CartpoleGuiArtifactError(f"Malformed checksum line {line_number}") from error
        _require(
            len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
            f"Checksum line {line_number} has an invalid SHA-256 digest",
        )
        target_relative = Path(relative)
        _require(
            not target_relative.is_absolute() and ".." not in target_relative.parts and str(target_relative) != ".",
            f"Checksum path escapes artifact: {relative}",
        )
        target = root / target_relative
        _require(target.is_file(), f"Checksummed file is missing: {relative}")
        _require(target not in listed, f"Checksum path is duplicated: {relative}")
        _require(sha256_file(target) == digest, f"Checksum mismatch: {relative}")
        listed.add(target)
    actual = {path for path in root.rglob("*") if path.is_file() and path != checksum_path}
    _require(listed == actual, "Checksum manifest does not cover exactly all sidecar evidence files")
    return len(listed)


def _load_runtime_finalizer() -> Any:
    path = Path(__file__).with_name("finalize_runtime_artifact.py")
    specification = importlib.util.spec_from_file_location("rambo_runtime_artifact_finalizer", path)
    if specification is None or specification.loader is None:  # pragma: no cover - filesystem guard.
        raise M2CartpoleGuiArtifactError(f"Cannot load runtime finalizer: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _validate_physx_evidence(value: Any, label: str) -> None:
    _require(isinstance(value, dict), f"{label} is required")
    _require(value.get("requested_cfg") == PHYSX_CFG_FQN, f"{label}.requested_cfg is not PhysxCfg")
    _require(value.get("actual_manager") == PHYSX_MANAGER_FQN, f"{label}.actual_manager is not PhysxManager")
    _require(value.get("use_newton_actuators") is False, f"{label}.use_newton_actuators must be false")


def is_explicit_rtx_renderer_extension(value: Any) -> bool:
    """Return whether an extension ID is accepted as direct RTX evidence.

    The runtime writes only IDs queried from Kit's extension manager.  The
    offline validator repeats this narrow check so a generic renderer-core
    entry cannot be substituted for an RTX render delegate in a sealed
    artifact.
    """

    return isinstance(value, str) and value.strip().lower() in RTX_RENDERER_EXTENSION_IDS


def _validate_gpu_sample(sample: Any, *, expected_gpu_index: int, label: str) -> None:
    _require(isinstance(sample, dict), f"{label} must be an object")
    _require(_as_int(sample.get("active_gpu_index"), f"{label}.active_gpu_index") == expected_gpu_index, f"{label} active GPU changed")
    _require(_as_int(sample.get("process_id"), f"{label}.process_id") > 0, f"{label} process ID is invalid")
    _require(_as_int(sample.get("sampled_monotonic_ns"), f"{label}.sampled_monotonic_ns") > 0, f"{label} timestamp is invalid")
    capture = sample.get("capture")
    _require(isinstance(capture, dict), f"{label}.capture is required")
    _require(capture.get("source") == "nvidia-smi", f"{label} is not a live nvidia-smi sample")
    _require(
        capture.get("gpu_query") == "nvidia-smi --query-gpu=index,name,uuid,memory.total",
        f"{label} GPU query is invalid",
    )
    _require(
        capture.get("compute_process_query") == "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory",
        f"{label} process query is invalid",
    )

    gpu_rows = sample.get("gpus")
    _require(isinstance(gpu_rows, list) and gpu_rows, f"{label} has no nvidia-smi GPU rows")
    active_rows = [row for row in gpu_rows if isinstance(row, dict) and row.get("index") == expected_gpu_index]
    _require(len(active_rows) == 1, f"{label} does not identify exactly one active GPU")
    active_gpu = active_rows[0]
    name = active_gpu.get("name")
    _require(isinstance(name, str) and TARGET_GPU_NAME_FRAGMENT in name.upper(), f"{label} is not an RTX 4090")
    _require(
        _as_int(active_gpu.get("memory_total_mib"), f"{label}.active_gpu.memory_total_mib") > 0,
        f"{label} active GPU has no memory",
    )

    process_rows = sample.get("compute_processes")
    _require(isinstance(process_rows, list) and process_rows, f"{label} has no nvidia-smi process rows")
    own_rows = [row for row in process_rows if isinstance(row, dict) and row.get("pid") == sample["process_id"]]
    _require(own_rows, f"{label} does not show the running Kit process in nvidia-smi")
    _require(
        any(_as_int(row.get("used_gpu_memory_mib"), f"{label}.process_memory") > 0 for row in own_rows),
        f"{label} Kit process has no recorded GPU memory",
    )

    renderer = sample.get("renderer")
    _require(isinstance(renderer, dict), f"{label}.renderer is required")
    _require(
        _as_int(renderer.get("active_gpu_setting"), f"{label}.renderer.active_gpu_setting") == expected_gpu_index,
        f"{label} renderer active GPU differs from nvidia-smi",
    )
    active_renderer = renderer.get("active_renderer_setting")
    _require(
        isinstance(active_renderer, str) and active_renderer.strip(),
        f"{label} does not record the active viewport Hydra engine",
    )
    _require(active_renderer.strip().lower() == "rtx", f"{label} viewport Hydra engine is not RTX")
    _require(
        renderer.get("active_renderer_source")
        == "omni.kit.viewport.utility.get_active_viewport().hydra_engine",
        f"{label} active renderer source is not the live Kit viewport",
    )
    legacy_active_renderer = renderer.get("legacy_active_renderer_setting")
    _require(
        legacy_active_renderer is None or isinstance(legacy_active_renderer, str),
        f"{label} legacy active renderer setting has an invalid type",
    )
    _require(
        not isinstance(legacy_active_renderer, str) or "llvmpipe" not in legacy_active_renderer.lower(),
        f"{label} legacy active renderer setting records llvmpipe",
    )
    rtx_extensions = renderer.get("enabled_rtx_extensions")
    _require(
        isinstance(rtx_extensions, list) and all(isinstance(item, str) and item for item in rtx_extensions) and rtx_extensions,
        f"{label} does not record an enabled RTX renderer extension",
    )
    _require(
        any(is_explicit_rtx_renderer_extension(item) for item in rtx_extensions),
        f"{label} does not record an enabled explicitly RTX-labelled renderer extension",
    )


def validate_runtime_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate the M2.3-specific fields written before Kit closes."""

    _require(summary.get("passed") is True, "Runtime summary did not pass")
    _require(summary.get("scenario") == RUNTIME_SCENARIO, "Runtime scenario is not Direct Cartpole")
    _require(summary.get("task") == RUNTIME_TASK, "Runtime task is not official Direct Cartpole")
    _require(summary.get("viz") == [VISUALIZER], "Runtime did not explicitly use --viz kit")
    _require(_as_int(summary.get("num_envs"), "summary.num_envs") == 1, "GUI gate must use one environment")
    _validate_physx_evidence(summary.get("backend_before"), "summary.backend_before")
    _validate_physx_evidence(summary.get("backend_after"), "summary.backend_after")

    observation = summary.get("gui_observation")
    _require(isinstance(observation, dict), "GUI observation record is required")
    _require(observation.get("mode") == GUI_OBSERVATION_MODE, "GUI observation mode is invalid")
    requested_seconds = _as_finite(observation.get("requested_wall_time_s"), "gui_observation.requested_wall_time_s")
    _require(
        MIN_OBSERVATION_SECONDS <= requested_seconds <= MAX_OBSERVATION_SECONDS,
        "GUI observation duration is outside the finite manual range",
    )
    actual_seconds = _as_finite(observation.get("actual_wall_time_s"), "gui_observation.actual_wall_time_s")
    _require(actual_seconds >= requested_seconds, "GUI observation did not reach its requested dwell time")
    _require(
        observation.get("operator_attestation_required_after_close") is True,
        "GUI observation must require a post-close operator attestation",
    )
    _require(
        observation.get("automatic_visual_observation_claimed") is False,
        "Runtime must not claim automatic visual observation",
    )
    requested_steps = _as_int(summary.get("steps"), "summary.steps")
    executed_steps = _as_int(summary.get("executed_steps"), "summary.executed_steps")
    _require(requested_steps > 0 and executed_steps >= requested_steps, "GUI gate did not complete its finite minimum steps")
    _require(
        _as_int(observation.get("maximum_executed_steps"), "gui_observation.maximum_executed_steps") >= executed_steps,
        "GUI gate exceeded its bounded execution limit",
    )
    expected_gpu_index = _as_int(observation.get("expected_active_gpu_index"), "gui_observation.expected_active_gpu_index")
    samples = observation.get("gpu_samples")
    _require(isinstance(samples, list) and len(samples) >= MIN_GPU_SAMPLES, "GUI gate needs Kit-live GPU samples")
    for index, sample in enumerate(samples, start=1):
        _validate_gpu_sample(sample, expected_gpu_index=expected_gpu_index, label=f"gpu_samples[{index}]")

    return {
        "requested_observation_seconds": requested_seconds,
        "actual_observation_seconds": actual_seconds,
        "executed_steps": executed_steps,
        "expected_active_gpu_index": expected_gpu_index,
        "gpu_sample_count": len(samples),
    }


def _parse_nvidia_smi_rows(stdout: str, fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            raise M2CartpoleGuiArtifactError(f"{label} returned malformed CSV at line {line_number}")
        rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _memory_mib(value: str, label: str) -> int:
    normalized = value.strip().replace("MiB", "").strip()
    try:
        result = int(float(normalized))
    except ValueError as error:
        raise M2CartpoleGuiArtifactError(f"{label} is not a GPU-memory value: {value!r}") from error
    if result < 0:
        raise M2CartpoleGuiArtifactError(f"{label} is negative")
    return result


def collect_gui_gpu_sample(
    *,
    active_gpu_index: int,
    renderer: dict[str, Any],
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    process_id: int | None = None,
) -> dict[str, Any]:
    """Capture read-only GPU/renderer evidence while Kit is still running.

    ``nvidia-smi`` is intentionally sampled from the live Kit process twice by
    the caller.  The function rejects missing process memory rather than using
    a post-close or host-only sample as a substitute.
    """

    _require(active_gpu_index >= 0, "Active GPU index must be non-negative")
    pid = os.getpid() if process_id is None else process_id
    _require(pid > 0, "Process ID must be positive")
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = command_runner(gpu_command, check=False, capture_output=True, text=True, timeout=10)
        process_result = command_runner(process_command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise M2CartpoleGuiArtifactError("Cannot execute nvidia-smi during the live Kit gate") from error
    if gpu_result.returncode != 0:
        raise M2CartpoleGuiArtifactError(f"Live nvidia-smi GPU query failed: {gpu_result.stderr.strip()}")
    if process_result.returncode != 0:
        raise M2CartpoleGuiArtifactError(f"Live nvidia-smi process query failed: {process_result.stderr.strip()}")

    raw_gpus = _parse_nvidia_smi_rows(gpu_result.stdout, ("index", "name", "uuid", "memory_total_mib"), "GPU query")
    raw_processes = _parse_nvidia_smi_rows(
        process_result.stdout,
        ("pid", "process_name", "used_gpu_memory_mib"),
        "compute-process query",
    )
    gpus: list[dict[str, Any]] = []
    for row in raw_gpus:
        try:
            index = int(row["index"])
        except ValueError as error:
            raise M2CartpoleGuiArtifactError(f"nvidia-smi GPU index is invalid: {row['index']!r}") from error
        gpus.append(
            {
                "index": index,
                "name": row["name"],
                "uuid": row["uuid"],
                "memory_total_mib": _memory_mib(row["memory_total_mib"], "GPU total memory"),
            }
        )
    processes: list[dict[str, Any]] = []
    for row in raw_processes:
        try:
            process_row_pid = int(row["pid"])
        except ValueError as error:
            raise M2CartpoleGuiArtifactError(f"nvidia-smi process PID is invalid: {row['pid']!r}") from error
        processes.append(
            {
                "pid": process_row_pid,
                "process_name": row["process_name"],
                "used_gpu_memory_mib": _memory_mib(row["used_gpu_memory_mib"], "process GPU memory"),
            }
        )
    sample = {
        "sampled_monotonic_ns": time.monotonic_ns(),
        "process_id": pid,
        "active_gpu_index": active_gpu_index,
        "capture": {
            "source": "nvidia-smi",
            "gpu_query": "nvidia-smi --query-gpu=index,name,uuid,memory.total",
            "compute_process_query": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory",
        },
        "gpus": gpus,
        "compute_processes": processes,
        "renderer": renderer,
    }
    _validate_gpu_sample(sample, expected_gpu_index=active_gpu_index, label="live GPU sample")
    return sample


def _runtime_binding(runtime_root: Path) -> dict[str, str]:
    return {
        "runtime_artifact_dir": str(runtime_root),
        "runtime_checksums_sha256": sha256_file(runtime_root / CHECKSUMS_FILENAME),
        "runtime_summary_sha256": sha256_file(runtime_root / "summary.json"),
        "runtime_process_exit_sha256": sha256_file(runtime_root / "process_exit.json"),
    }


def _validate_runtime_artifact(runtime_artifact_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(runtime_artifact_dir).expanduser().resolve()
    _require(root.is_dir(), f"Runtime artifact directory does not exist: {root}")
    finalizer = _load_runtime_finalizer()
    try:
        runtime_result = finalizer.verify_runtime_artifact(root)
    except finalizer.RuntimeArtifactFinalizationError as error:
        raise M2CartpoleGuiArtifactError(f"Runtime artifact is not sealed and accepted: {error}") from error
    summary = _read_json(root / "summary.json", "runtime summary")
    return root, summary, runtime_result


def write_operator_attestation(
    *, runtime_artifact_dir: str | Path, attestation_dir: str | Path, operator_name: str
) -> dict[str, Any]:
    """Write a fresh post-close self-attestation sidecar for a sealed runtime.

    The caller must obtain interactive confirmation before calling this function.
    It intentionally has no option for a pre-supplied acknowledgement flag.
    """

    name = operator_name.strip()
    _require(name, "Operator name must not be empty")
    runtime_root, summary, _ = _validate_runtime_artifact(runtime_artifact_dir)
    runtime_evidence = validate_runtime_summary(summary)
    root = Path(attestation_dir).expanduser().resolve()
    _require(not root.exists(), f"Refusing to overwrite existing attestation directory: {root}")
    root.mkdir(parents=True)
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "kind": "operator_gui_observation_attestation",
        "operator_name": name,
        "operator_acknowledged": True,
        "confirmation_method": "interactive_tty_exact_phrase_after_kit_close",
        "confirmation_phrase": TERMINAL_CONFIRMATION_PHRASE,
        "statement": OPERATOR_ATTESTATION_STATEMENT,
        "limitation": OBSERVATION_LIMITATION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_binding": _runtime_binding(runtime_root),
        "runtime_evidence": runtime_evidence,
    }
    _write_json(root / ATTESTATION_FILENAME, attestation)
    _write_checksums(root)
    return {
        "attestation_dir": str(root),
        "runtime_artifact_dir": str(runtime_root),
        "checksummed_files": _validate_checksums(root),
    }


def _validate_attestation_sidecar(root: Path, runtime_root: Path, runtime_evidence: dict[str, Any]) -> dict[str, Any]:
    _require(root.is_dir(), f"Attestation directory does not exist: {root}")
    expected_files = {ATTESTATION_FILENAME, CHECKSUMS_FILENAME}
    entries = list(root.iterdir())
    _require(all(path.is_file() for path in entries), "Attestation sidecar must not contain directories")
    actual_files = {path.name for path in entries}
    _require(actual_files == expected_files, "Attestation sidecar must contain only its JSON and checksum manifest")
    checksum_count = _validate_checksums(root)
    attestation = _read_json(root / ATTESTATION_FILENAME, "operator attestation")
    _require(attestation.get("schema_version") == SCHEMA_VERSION, "Attestation schema version is invalid")
    _require(attestation.get("artifact_kind") == ARTIFACT_KIND, "Attestation artifact kind is invalid")
    _require(attestation.get("kind") == "operator_gui_observation_attestation", "Attestation kind is invalid")
    _require(
        isinstance(attestation.get("operator_name"), str) and attestation["operator_name"].strip(),
        "Attestation operator name is required",
    )
    _require(attestation.get("operator_acknowledged") is True, "Operator acknowledgement is required")
    _require(
        attestation.get("confirmation_method") == "interactive_tty_exact_phrase_after_kit_close",
        "Attestation must record an interactive post-close terminal confirmation",
    )
    _require(attestation.get("confirmation_phrase") == TERMINAL_CONFIRMATION_PHRASE, "Attestation phrase is invalid")
    _require(attestation.get("statement") == OPERATOR_ATTESTATION_STATEMENT, "Attestation statement is invalid")
    _require(attestation.get("limitation") == OBSERVATION_LIMITATION, "Attestation limitation is invalid")
    _require(isinstance(attestation.get("recorded_at_utc"), str) and attestation["recorded_at_utc"], "Attestation timestamp is required")
    _require(attestation.get("runtime_evidence") == runtime_evidence, "Attestation runtime evidence disagrees with sealed runtime")
    binding = attestation.get("runtime_binding")
    _require(isinstance(binding, dict), "Attestation runtime binding is required")
    _require(binding == _runtime_binding(runtime_root), "Attestation does not bind this exact sealed runtime artifact")
    return {
        "operator_name": attestation["operator_name"],
        "checksummed_files": checksum_count,
        "recorded_at_utc": attestation["recorded_at_utc"],
    }


def validate_m2_cartpole_gui_gate(
    *, runtime_artifact_dir: str | Path, attestation_dir: str | Path
) -> dict[str, Any]:
    """Validate the sealed runtime plus its post-close human-attestation sidecar."""

    runtime_root, summary, runtime_result = _validate_runtime_artifact(runtime_artifact_dir)
    runtime_evidence = validate_runtime_summary(summary)
    attestation_root = Path(attestation_dir).expanduser().resolve()
    attestation_evidence = _validate_attestation_sidecar(attestation_root, runtime_root, runtime_evidence)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "runtime_artifact_dir": str(runtime_root),
        "attestation_dir": str(attestation_root),
        "runtime_exit_status": runtime_result["process_exit_status"],
        "runtime_evidence": runtime_evidence,
        "attestation": attestation_evidence,
        "limitation": OBSERVATION_LIMITATION,
    }
