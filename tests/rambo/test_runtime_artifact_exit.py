"""Pure-Python contracts for post-Kit runtime exit sealing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rambo" / "finalize_runtime_artifact.py"
    spec = importlib.util.spec_from_file_location("rambo_runtime_artifact_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(root: Path, *, passed: bool = True) -> None:
    (root / "summary.json").write_text(json.dumps({"passed": passed}) + "\n", encoding="utf-8")


def test_finalizer_records_real_zero_exit_and_covers_all_files(tmp_path: Path) -> None:
    module = _load_module()
    _write_summary(tmp_path)
    (tmp_path / "payload.bin").write_bytes(b"evidence")

    result = module.finalize_runtime_artifact(tmp_path, exit_status=0)

    assert result["process_exit"]["acceptance_passed"] is True
    record = json.loads((tmp_path / "process_exit.json").read_text(encoding="utf-8"))
    assert record["shell_exit_status"] == 0
    assert record["workload_summary_passed_before_exit"] is True
    manifest = (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    listed = {line.split("  ", maxsplit=1)[1]: line.split("  ", maxsplit=1)[0] for line in manifest}
    assert set(listed) == {"summary.json", "payload.bin", "process_exit.json"}
    for relative, digest in listed.items():
        assert digest == hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
    assert module.verify_runtime_artifact(tmp_path)["process_exit_status"] == 0


def test_finalizer_records_nonzero_exit_as_nonacceptance(tmp_path: Path) -> None:
    module = _load_module()
    _write_summary(tmp_path)

    result = module.finalize_runtime_artifact(tmp_path, exit_status=139)

    assert result["process_exit"] == {
        "schema_version": 1,
        "captured_by": "scripts/rambo/run_runtime_artifact.sh",
        "shell_exit_status": 139,
        "exit_status_zero": False,
        "signal_hint": 11,
        "workload_summary_passed_before_exit": True,
        "acceptance_passed": False,
    }
    with pytest.raises(module.RuntimeArtifactFinalizationError, match="exit status is not zero"):
        module.verify_runtime_artifact(tmp_path)


def test_finalizer_rejects_missing_or_malformed_summary(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.RuntimeArtifactFinalizationError, match="runtime summary is missing"):
        module.finalize_runtime_artifact(tmp_path, exit_status=0)
    (tmp_path / "summary.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(module.RuntimeArtifactFinalizationError, match="must be a JSON object"):
        module.finalize_runtime_artifact(tmp_path, exit_status=0)


def test_finalizer_updates_third_person_manifest_with_exit_sidecar(tmp_path: Path) -> None:
    module = _load_module()
    _write_summary(tmp_path)
    (tmp_path / "third_person_diagnostic_manifest.json").write_text(
        json.dumps({"files": {"summary": "summary.json"}}) + "\n", encoding="utf-8"
    )

    module.finalize_runtime_artifact(tmp_path, exit_status=0)

    manifest = json.loads((tmp_path / "third_person_diagnostic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["process_exit"] == "process_exit.json"
    assert manifest["process_exit"]["acceptance_passed"] is True


def test_runtime_artifact_wrapper_requires_one_output_directory() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "rambo" / "run_runtime_artifact.sh").read_text(encoding="utf-8")
    assert '"${SCRIPT_DIR}/run60.sh" "$@"' in source
    assert "--process-exit-status" in source
    assert "unset OMNI_KIT_ACCEPT_EULA" in source
    assert "Exactly one --output-dir is required" in source
    validator = (root / "scripts" / "rambo" / "validate_runtime_artifact.py").read_text(encoding="utf-8")
    assert "verify_runtime_artifact" in validator


def test_runtime_artifact_wrapper_seals_a_real_child_exit_without_starting_kit(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    child = tmp_path / "write_summary.py"
    child.write_text(
        "from pathlib import Path\n"
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-dir', type=Path, required=True)\n"
        "args = parser.parse_args()\n"
        "args.output_dir.mkdir(parents=True, exist_ok=True)\n"
        "(args.output_dir / 'summary.json').write_text(json.dumps({'passed': True}) + '\\n')\n",
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact"
    environment = dict(os.environ)
    environment["RAMBO_VENV"] = "/workspace/venvs/rambo60"
    environment["OMNI_KIT_ACCEPT_EULA"] = "Y"
    result = subprocess.run(
        [
            str(root / "scripts" / "rambo" / "run_runtime_artifact.sh"),
            str(child),
            "--output-dir",
            str(artifact),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((artifact / "process_exit.json").read_text(encoding="utf-8"))
    assert record["shell_exit_status"] == 0
    assert record["acceptance_passed"] is True
