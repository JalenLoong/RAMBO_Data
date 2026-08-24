#!/usr/bin/env python3
"""Create and verify a static, source-clean M11 native-freeze artifact.

The program deliberately uses only the Python standard library.  It never
starts Isaac Sim/Kit, Docker, or NVIDIA Container Toolkit, and it never
selects a physics backend.  A caller supplies already-produced installation
verification evidence; this program binds that evidence, the accepted runtime
artifact list, and the source inputs that define the migration hand-off.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Iterable


CHECKSUMS_FILENAME = "checksums.sha256"
FREEZE_INPUTS_FILENAME = "freeze-inputs.sha256"
INSTALL_VERIFICATION_FILENAME = "install-verification.txt"
ACCEPTED_ARTIFACTS_FILENAME = "accepted-runtime-artifacts.txt"
METADATA_FILENAME = "freeze-metadata.json"
GIT_HEAD_FILENAME = "rambo_git_head.txt"
GIT_STATUS_FILENAME = "rambo_git_status_porcelain_v1.txt"
INSTALL_VERIFICATION_SENTINEL = "ISAACSIM60_INSTALL_VERIFIED"
SCHEMA_VERSION = 1

# Keep this explicit rather than globbing: a freeze must explain exactly why a
# source file participates in the hand-off.  The launch chain includes run.sh,
# which is reached transitively by run60.sh.
FREEZE_INPUT_GROUPS: dict[str, tuple[str, ...]] = {
    "pinned_installation": (
        "requirements/isaacsim60.in",
        "requirements/isaacsim60.lock",
        "scripts/setup_isaacsim60.sh",
        "scripts/verify_isaacsim60_install.py",
    ),
    "runtime_launch_and_sealing_wrappers": (
        "scripts/rambo/run.sh",
        "scripts/rambo/run60.sh",
        "scripts/rambo/run60_contract.sh",
        "scripts/rambo/run_runtime_artifact.sh",
        "scripts/rambo/finalize_runtime_artifact.py",
        "scripts/rambo/validate_runtime_artifact.py",
    ),
    "runtime_acceptance_validators": (
        "scripts/rambo/validate_final_third_person_artifact.py",
        "scripts/rambo/validate_button_physx_episode.py",
        "scripts/rambo/validate_gui_keyboard_artifact.py",
    ),
    "manifests": (
        "checkpoint-manifest.json",
        "host-manifest.json",
    ),
    "migration_records": (
        "CODEX_EXECUTION_PLAN.md",
        "migration-report.md",
        "MIGRATION_STATUS.md",
        "MIGRATION_FINAL_MATRIX.md",
    ),
    "freeze_generator": (
        "scripts/rambo/freeze_native_migration.py",
    ),
}
ALL_FREEZE_INPUT_PATHS = tuple(path for paths in FREEZE_INPUT_GROUPS.values() for path in paths)
if len(ALL_FREEZE_INPUT_PATHS) != len(set(ALL_FREEZE_INPUT_PATHS)):
    raise RuntimeError("native-freeze input paths must be unique")


class NativeFreezeError(RuntimeError):
    """Raised when a native-freeze artifact cannot be created or verified."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeFreezeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NativeFreezeError(f"{label} is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise NativeFreezeError(f"{label} is not valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _object(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{label} must be a non-empty string")
    return value


def _relative_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    _require(bool(path.parts), f"{label} must not be empty")
    _require(not path.is_absolute(), f"{label} must be relative")
    _require(".." not in path.parts and "." not in path.parts, f"{label} must not traverse directories")
    return path


def _repo_path(repo_root: Path, relative: str, *, label: str) -> Path:
    normalized = _relative_path(relative, label=label)
    path = (repo_root / Path(*normalized.parts)).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise NativeFreezeError(f"{label} escapes repository root: {relative}") from exc
    return path


def freeze_input_paths() -> tuple[str, ...]:
    """Return the explicit source paths that a current M11 freeze must bind."""

    return ALL_FREEZE_INPUT_PATHS


def _validate_host_manifest(host_manifest: dict[str, Any]) -> None:
    runtime = _object(host_manifest.get("runtime"), label="host manifest runtime")
    physics = _object(runtime.get("physics"), label="host manifest runtime.physics")
    _require(physics.get("required_backend") == "PhysX", "host manifest must require PhysX")
    _require(
        physics.get("required_manager") == "isaaclab_physx.physics.physx_manager.PhysxManager",
        "host manifest must require the exact PhysxManager FQN",
    )
    _require(physics.get("use_newton_actuators") is False, "host manifest must disable Newton actuators")
    _require(
        physics.get("optional_isaaclab_newton_extras_requested") is False,
        "host manifest must not request optional Isaac Lab Newton extras",
    )

    host = _object(host_manifest.get("host"), label="host manifest host")
    memory = _object(host.get("memory"), label="host manifest host.memory")
    for key in ("physical_total_bytes", "swap_total_bytes"):
        value = memory.get(key)
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"host memory {key} must be a non-negative integer")
    _nonempty_string(memory.get("probe"), label="host memory probe")

    toolchain = _object(host.get("toolchain"), label="host manifest host.toolchain")
    for tool in ("uv", "git"):
        details = _object(toolchain.get(tool), label=f"host toolchain {tool}")
        _nonempty_string(details.get("version"), label=f"host toolchain {tool}.version")
        _nonempty_string(details.get("path"), label=f"host toolchain {tool}.path")

    container = _object(host_manifest.get("container_status"), label="host manifest container_status")
    _require(
        isinstance(container.get("docker_engine_or_nvidia_container_toolkit_available_on_validation_host"), bool),
        "container availability must be recorded as a boolean",
    )
    _require(isinstance(container.get("docker_build_or_runtime_validated"), bool), "container runtime validation state must be a boolean")
    _nonempty_string(container.get("probe_scope"), label="container probe_scope")
    _object(container.get("docker_cli"), label="container docker_cli")
    toolkit = _object(container.get("nvidia_container_toolkit"), label="container nvidia_container_toolkit")
    _require(isinstance(toolkit.get("nvidia_container_cli_present"), bool), "toolkit CLI presence must be a boolean")
    _require(isinstance(toolkit.get("nvidia_container_runtime_present"), bool), "toolkit runtime presence must be a boolean")


def _artifact_root_from_host_manifest(host_manifest: dict[str, Any]) -> Path:
    evidence = _object(host_manifest.get("evidence"), label="host manifest evidence")
    return Path(_nonempty_string(evidence.get("artifact_root"), label="host manifest evidence.artifact_root")).expanduser().resolve()


def _accepted_runtime_artifacts(host_manifest: dict[str, Any]) -> list[Path]:
    evidence = _object(host_manifest.get("evidence"), label="host manifest evidence")
    values: list[str] = []
    m3_artifact = evidence.get("m3_clean_inventory_artifact")
    if m3_artifact is not None:
        values.append(_nonempty_string(m3_artifact, label="host manifest M3 artifact"))
    accepted = evidence.get("accepted_runtime_artifacts")
    _require(isinstance(accepted, list) and accepted, "host manifest accepted runtime artifacts must be a non-empty list")
    for index, entry in enumerate(accepted, start=1):
        entry_object = _object(entry, label=f"accepted runtime artifact {index}")
        values.append(_nonempty_string(entry_object.get("path"), label=f"accepted runtime artifact {index}.path"))

    unique: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).expanduser().resolve()
        _require(path.is_dir(), f"accepted runtime artifact is missing: {path}")
        if path not in seen:
            unique.append(path)
            seen.add(path)
    _require(unique, "native freeze needs at least one accepted runtime artifact")
    return unique


def _read_install_verification(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise NativeFreezeError(f"installation verification is missing: {path}") from exc
    lines = content.splitlines()
    _require(lines and lines[-1] == INSTALL_VERIFICATION_SENTINEL, "installation verification sentinel is missing")
    try:
        value = json.loads("\n".join(lines[:-1]))
    except json.JSONDecodeError as exc:
        raise NativeFreezeError("installation verification JSON is invalid") from exc
    value = _object(value, label="installation verification")
    _require(value.get("verified") is True, "installation verification is not marked verified")
    _require(value.get("verification_mode") == "gpu-required", "installation verification must be gpu-required")
    physics = _object(value.get("physics_policy"), label="installation verification physics_policy")
    _require(physics.get("required_runtime_backend") == "PhysX", "installation verification must require PhysX")
    return content, value


def _parse_digest_lines(content: str, *, label: str) -> dict[str, str]:
    records: dict[str, str] = {}
    lines = content.splitlines()
    _require(lines, f"{label} must not be empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise NativeFreezeError(f"{label} line {line_number} must be '<sha256>  <path>'") from exc
        _require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), f"{label} line {line_number} has an invalid digest")
        _relative_path(relative, label=f"{label} line {line_number} path")
        _require(relative not in records, f"{label} has duplicate path: {relative}")
        records[relative] = digest
    return records


def _write_checksums(artifact_dir: Path) -> int:
    checksum_path = artifact_dir / CHECKSUMS_FILENAME
    files = sorted(path for path in artifact_dir.rglob("*") if path.is_file() and path != checksum_path)
    _write_text(
        checksum_path,
        "".join(f"{_sha256(path)}  {path.relative_to(artifact_dir).as_posix()}\n" for path in files),
    )
    return len(files)


def _verify_checksums(artifact_dir: Path) -> int:
    checksum_path = artifact_dir / CHECKSUMS_FILENAME
    try:
        records = _parse_digest_lines(checksum_path.read_text(encoding="utf-8"), label="native-freeze checksums")
    except FileNotFoundError as exc:
        raise NativeFreezeError("native-freeze checksum manifest is missing") from exc
    listed: set[Path] = set()
    for relative, digest in records.items():
        path = _repo_path(artifact_dir, relative, label="native-freeze checksum path")
        _require(path.is_file(), f"native-freeze checksum target is missing: {relative}")
        _require(_sha256(path) == digest, f"native-freeze checksum mismatch: {relative}")
        listed.add(path)
    actual = {path.resolve() for path in artifact_dir.rglob("*") if path.is_file() and path.name != CHECKSUMS_FILENAME}
    _require(listed == actual, "native-freeze checksums do not cover exactly all artifact files")
    return len(listed)


def _write_freeze_inputs(repo_root: Path, artifact_dir: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative in freeze_input_paths():
        source = _repo_path(repo_root, relative, label="native-freeze source input")
        _require(source.is_file(), f"native-freeze source input is missing: {relative}")
        records[relative] = _sha256(source)
    _write_text(
        artifact_dir / FREEZE_INPUTS_FILENAME,
        "".join(f"{digest}  {relative}\n" for relative, digest in records.items()),
    )
    return records


def create_native_freeze_artifact(
    repo_root: Path,
    artifact_dir: Path,
    install_verification_file: Path,
    *,
    git_head: str,
    git_status: str,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a new immutable M11 artifact without launching a runtime.

    ``git_head`` and ``git_status`` are injected so pure-Python tests can
    exercise the core logic without invoking Git.  The command-line entry
    point obtains them using read-only Git commands.
    """

    root = repo_root.expanduser().resolve()
    destination = artifact_dir.expanduser().resolve()
    _require(root.is_dir(), f"repository root does not exist: {root}")
    _require(destination.is_absolute(), "native-freeze artifact directory must be absolute")
    _require(not destination.exists(), f"native-freeze artifact directory already exists: {destination}")
    _require(git_status == "", "native freeze requires a source-clean Git worktree")
    _require(len(git_head) == 40 and all(character in "0123456789abcdef" for character in git_head), "Git head must be a full SHA-1")

    host_manifest = _read_json(root / "host-manifest.json", label="host manifest")
    _validate_host_manifest(host_manifest)
    artifact_root = _artifact_root_from_host_manifest(host_manifest)
    try:
        destination.relative_to(artifact_root)
    except ValueError as exc:
        raise NativeFreezeError(f"native-freeze output must be below configured artifact root: {artifact_root}") from exc

    install_source = install_verification_file.expanduser().resolve()
    install_content, _ = _read_install_verification(install_source)
    accepted_artifacts = _accepted_runtime_artifacts(host_manifest)
    for relative in freeze_input_paths():
        _require(_repo_path(root, relative, label="native-freeze source input").is_file(), f"native-freeze source input is missing: {relative}")

    destination.mkdir(parents=True)
    _write_text(destination / GIT_HEAD_FILENAME, f"{git_head}\n")
    _write_text(destination / GIT_STATUS_FILENAME, git_status)
    _write_text(destination / ACCEPTED_ARTIFACTS_FILENAME, "".join(f"{path}\n" for path in accepted_artifacts))
    _write_text(destination / INSTALL_VERIFICATION_FILENAME, install_content)
    freeze_records = _write_freeze_inputs(root, destination)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "path": "scripts/rambo/freeze_native_migration.py",
            "sha256": freeze_records["scripts/rambo/freeze_native_migration.py"],
            "standard_library_only": True,
        },
        "repository": {
            "worktree": str(root),
            "git_head": git_head,
            "source_clean": True,
        },
        "execution_boundary": {
            "isaac_sim_or_kit_started_by_generator": False,
            "docker_or_nvidia_container_toolkit_started_by_generator": False,
            "installation_verification_source_sha256": _sha256(install_source),
        },
        "freeze_input_groups": {group: list(paths) for group, paths in FREEZE_INPUT_GROUPS.items()},
        "freeze_input_count": len(freeze_records),
        "accepted_runtime_artifact_count": len(accepted_artifacts),
    }
    _write_json(destination / METADATA_FILENAME, metadata)
    checksummed_files = _write_checksums(destination)
    return {
        "artifact_dir": str(destination),
        "checksummed_files": checksummed_files,
        "freeze_input_count": len(freeze_records),
        "accepted_runtime_artifact_count": len(accepted_artifacts),
    }


def verify_native_freeze_artifact(repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    """Verify checksums and that every frozen source input still matches."""

    root = repo_root.expanduser().resolve()
    destination = artifact_dir.expanduser().resolve()
    _require(root.is_dir(), f"repository root does not exist: {root}")
    _require(destination.is_dir(), f"native-freeze artifact directory does not exist: {destination}")
    for filename in (
        CHECKSUMS_FILENAME,
        FREEZE_INPUTS_FILENAME,
        INSTALL_VERIFICATION_FILENAME,
        ACCEPTED_ARTIFACTS_FILENAME,
        METADATA_FILENAME,
        GIT_HEAD_FILENAME,
        GIT_STATUS_FILENAME,
    ):
        _require((destination / filename).is_file(), f"native-freeze artifact is missing {filename}")

    host_manifest = _read_json(root / "host-manifest.json", label="host manifest")
    _validate_host_manifest(host_manifest)
    _read_install_verification(destination / INSTALL_VERIFICATION_FILENAME)
    records = _parse_digest_lines((destination / FREEZE_INPUTS_FILENAME).read_text(encoding="utf-8"), label="native-freeze inputs")
    _require(set(records) == set(freeze_input_paths()), "native-freeze inputs do not match the canonical input list")
    for relative, digest in records.items():
        source = _repo_path(root, relative, label="native-freeze source input")
        _require(source.is_file(), f"native-freeze source input is missing: {relative}")
        _require(_sha256(source) == digest, f"native-freeze source input changed: {relative}")

    metadata = _read_json(destination / METADATA_FILENAME, label="native-freeze metadata")
    _require(metadata.get("schema_version") == SCHEMA_VERSION, "native-freeze metadata schema version is unsupported")
    _require(metadata.get("freeze_input_groups") == {group: list(paths) for group, paths in FREEZE_INPUT_GROUPS.items()}, "native-freeze metadata input groups differ from canonical groups")
    _require((destination / GIT_STATUS_FILENAME).read_text(encoding="utf-8") == "", "native-freeze artifact records a dirty source worktree")
    checksummed_files = _verify_checksums(destination)
    return {
        "artifact_dir": str(destination),
        "checksummed_files": checksummed_files,
        "freeze_input_count": len(records),
    }


def _git_text(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise NativeFreezeError(f"Git command failed: {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--install-verification-file", type=Path)
    parser.add_argument("--verify", action="store_true", help="verify an existing artifact instead of creating one")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            _require(args.install_verification_file is None, "--install-verification-file is only valid when creating an artifact")
            result = verify_native_freeze_artifact(args.repo_root, args.artifact_dir)
        else:
            _require(args.install_verification_file is not None, "--install-verification-file is required when creating an artifact")
            git_head = _git_text(args.repo_root, "rev-parse", "HEAD").strip()
            git_status = _git_text(args.repo_root, "status", "--porcelain=v1")
            result = create_native_freeze_artifact(
                args.repo_root,
                args.artifact_dir,
                args.install_verification_file,
                git_head=git_head,
                git_status=git_status,
            )
    except NativeFreezeError as error:
        print(f"NATIVE_FREEZE_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
