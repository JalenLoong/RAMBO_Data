#!/usr/bin/env python3
"""Seal a finished RAMBO runtime artifact with its parent-observed exit status.

This program is intentionally standard-library-only.  It runs only after the
wrapped Isaac Sim child has exited, never imports a simulator package, and
therefore cannot select a physics backend.  A runtime artifact is accepted
only when its workload summary passed *and* this sidecar records exit status
zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


CHECKSUMS_FILENAME = "checksums.sha256"
SUMMARY_FILENAME = "summary.json"
PROCESS_EXIT_FILENAME = "process_exit.json"
THIRD_PERSON_MANIFEST_FILENAME = "third_person_diagnostic_manifest.json"
SCHEMA_VERSION = 1


class RuntimeArtifactFinalizationError(RuntimeError):
    """Raised when a post-runtime artifact cannot be sealed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeArtifactFinalizationError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeArtifactFinalizationError(f"{label} is missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeArtifactFinalizationError(f"{label} is not valid JSON: {path.name}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> int:
    """Cover every evidence file below ``output_dir`` except this manifest."""

    checksum_path = output_dir / CHECKSUMS_FILENAME
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != checksum_path.resolve()
    )
    checksum_path.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def verify_checksums(output_dir: Path) -> int:
    """Verify that the final checksum manifest covers every evidence file once."""

    checksum_path = output_dir / CHECKSUMS_FILENAME
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeArtifactFinalizationError("runtime checksum manifest is missing") from exc
    _require(lines, "runtime checksum manifest is empty")
    listed: set[Path] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise RuntimeArtifactFinalizationError(f"malformed checksum line {line_number}") from exc
        _require(len(digest) == 64 and all(char in "0123456789abcdef" for char in digest), "invalid checksum")
        path = (output_dir / relative).resolve()
        try:
            path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise RuntimeArtifactFinalizationError("checksum path escapes artifact root") from exc
        _require(path.is_file(), f"checksum target is missing: {relative}")
        _require(path not in listed, f"duplicate checksum target: {relative}")
        _require(_sha256(path) == digest, f"checksum mismatch: {relative}")
        listed.add(path)
    actual = {
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file() and path.resolve() != checksum_path.resolve()
    }
    _require(listed == actual, "runtime checksum manifest does not cover exactly all evidence files")
    return len(listed)


def _exit_record(*, exit_status: int, summary_passed_before_exit: bool) -> dict[str, Any]:
    _require(0 <= exit_status <= 255, f"shell exit status must be in [0, 255], got {exit_status}")
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_by": "scripts/rambo/run_runtime_artifact.sh",
        "shell_exit_status": exit_status,
        "exit_status_zero": exit_status == 0,
        "signal_hint": exit_status - 128 if exit_status >= 128 else None,
        "workload_summary_passed_before_exit": summary_passed_before_exit,
        "acceptance_passed": summary_passed_before_exit and exit_status == 0,
    }


def finalize_runtime_artifact(output_dir: Path, *, exit_status: int) -> dict[str, Any]:
    """Write post-close exit evidence and refresh the artifact checksum manifest."""

    root = output_dir.expanduser().resolve()
    _require(root.is_absolute(), "artifact directory must be absolute")
    _require(root.is_dir(), f"artifact directory does not exist: {root}")
    summary = _read_json(root / SUMMARY_FILENAME, label="runtime summary")
    summary_passed = summary.get("passed") is True
    record = _exit_record(exit_status=exit_status, summary_passed_before_exit=summary_passed)
    _write_json(root / PROCESS_EXIT_FILENAME, record)

    # Keep a final third-person manifest self-contained while preserving the
    # workload summary that was necessarily written before Kit closed.
    diagnostic_manifest = root / THIRD_PERSON_MANIFEST_FILENAME
    if diagnostic_manifest.is_file():
        manifest = _read_json(diagnostic_manifest, label="third-person manifest")
        files = manifest.setdefault("files", {})
        _require(isinstance(files, dict), "third-person manifest files must be an object")
        files["process_exit"] = PROCESS_EXIT_FILENAME
        manifest["process_exit"] = record
        _write_json(diagnostic_manifest, manifest)

    checksummed_files = write_checksums(root)
    return {
        "artifact_dir": str(root),
        "process_exit": record,
        "checksummed_files": checksummed_files,
    }


def verify_runtime_artifact(output_dir: Path) -> dict[str, Any]:
    """Require a passed workload, a zero post-close exit, and exact checksums."""

    root = output_dir.expanduser().resolve()
    _require(root.is_dir(), f"artifact directory does not exist: {root}")
    summary = _read_json(root / SUMMARY_FILENAME, label="runtime summary")
    _require(summary.get("passed") is True, "runtime workload summary did not pass")
    process_exit = _read_json(root / PROCESS_EXIT_FILENAME, label="runtime process exit")
    _require(process_exit.get("shell_exit_status") == 0, "runtime process exit status is not zero")
    _require(process_exit.get("exit_status_zero") is True, "runtime process exit is not marked zero")
    _require(process_exit.get("workload_summary_passed_before_exit") is True, "runtime summary did not pass before exit")
    _require(process_exit.get("acceptance_passed") is True, "runtime artifact is not accepted after exit")
    return {
        "artifact_dir": str(root),
        "checksummed_files": verify_checksums(root),
        "process_exit_status": process_exit["shell_exit_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-exit-status", type=int, required=True)
    args = parser.parse_args()
    try:
        result = finalize_runtime_artifact(args.output_dir, exit_status=args.process_exit_status)
    except RuntimeArtifactFinalizationError as error:
        print(f"RUNTIME_ARTIFACT_FINALIZATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
