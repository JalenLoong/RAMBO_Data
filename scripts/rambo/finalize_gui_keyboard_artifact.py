#!/usr/bin/env python3
"""Seal M8 GUI-keyboard evidence after the Isaac Sim child has exited.

This finalizer is deliberately standard-library-only.  It is invoked by the
dedicated parent runner only after ``run60.sh`` returns, so it cannot start Kit
or select a physics backend.  The M8 offline validator accepts an artifact
only when this sidecar records a completed workload and a real zero child exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ARTIFACT_KIND = "rambo_gui_keyboard_teleop"
SUMMARY_FILENAME = "summary.json"
PROCESS_EXIT_FILENAME = "process_exit.json"
CHECKSUMS_FILENAME = "checksums.sha256"
POST_CLOSE_RUNNER = "scripts/rambo/run_gui_keyboard_artifact.sh"
PROCESS_EXIT_SCHEMA_VERSION = 1


class GuiKeyboardArtifactFinalizationError(RuntimeError):
    """Raised when post-close M8 evidence cannot be sealed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuiKeyboardArtifactFinalizationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GuiKeyboardArtifactFinalizationError(f"{label} is missing: {path.name}") from error
    except json.JSONDecodeError as error:
        raise GuiKeyboardArtifactFinalizationError(f"{label} is not valid JSON: {path.name}") from error
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


def _write_checksums(output_dir: Path) -> int:
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


def _exit_record(*, exit_status: int, summary_outcome: str | None) -> dict[str, Any]:
    _require(0 <= exit_status <= 255, f"shell exit status must be in [0, 255], got {exit_status}")
    summary_completed = summary_outcome == "completed"
    return {
        "schema_version": PROCESS_EXIT_SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "captured_by": POST_CLOSE_RUNNER,
        "captured_after_child_exit": True,
        "captured_at_utc": _utc_now(),
        "shell_exit_status": exit_status,
        "exit_status_zero": exit_status == 0,
        "signal_hint": exit_status - 128 if exit_status >= 128 else None,
        "summary_outcome_before_exit": summary_outcome,
        "workload_summary_completed_before_exit": summary_completed,
        "acceptance_passed": summary_completed and exit_status == 0,
    }


def finalize_gui_keyboard_artifact(output_dir: Path, *, exit_status: int) -> dict[str, Any]:
    """Write parent-observed post-close exit evidence and refresh checksums."""

    root = output_dir.expanduser().resolve()
    _require(root.is_dir(), f"GUI artifact directory does not exist: {root}")
    summary = _read_json(root / SUMMARY_FILENAME, label="GUI workload summary")
    _require(summary.get("artifact_kind") == ARTIFACT_KIND, "GUI workload summary artifact kind is invalid")
    _require(summary.get("post_close_exit_required") is True, "GUI workload summary does not require post-close sealing")
    _require(
        summary.get("post_close_exit_file") == PROCESS_EXIT_FILENAME,
        "GUI workload summary post-close exit file is invalid",
    )
    _require(
        summary.get("post_close_exit_runner") == POST_CLOSE_RUNNER,
        "GUI workload summary post-close runner is invalid",
    )
    process_exit_path = root / PROCESS_EXIT_FILENAME
    _require(not process_exit_path.exists(), f"Post-close exit record already exists: {process_exit_path}")
    outcome = summary.get("outcome")
    record = _exit_record(exit_status=exit_status, summary_outcome=outcome if isinstance(outcome, str) else None)
    _write_json(process_exit_path, record)
    checksummed_files = _write_checksums(root)
    return {
        "artifact_dir": str(root),
        "process_exit": record,
        "checksummed_files": checksummed_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-artifact-dir", type=Path, required=True)
    parser.add_argument("--process-exit-status", type=int, required=True)
    args = parser.parse_args()
    try:
        result = finalize_gui_keyboard_artifact(args.gui_artifact_dir, exit_status=args.process_exit_status)
    except GuiKeyboardArtifactFinalizationError as error:
        print(f"GUI_KEYBOARD_ARTIFACT_FINALIZATION_FAILURE: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
