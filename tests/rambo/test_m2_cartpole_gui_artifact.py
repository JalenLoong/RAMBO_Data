"""Pure-Python contracts for the deferred M2.3 manual Kit GUI evidence path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_schema():
    path = _root() / "scripts" / "rambo" / "m2_cartpole_gui_artifact.py"
    specification = importlib.util.spec_from_file_location("m2_cartpole_gui_artifact_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _physx() -> dict[str, object]:
    return {
        "requested_cfg": "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg",
        "actual_manager": "isaaclab_physx.physics.physx_manager.PhysxManager",
        "configured_manager": "isaaclab_physx.physics.physx_manager:PhysxManager",
        "use_newton_actuators": False,
    }


def _gpu_sample(*, timestamp: int) -> dict[str, object]:
    return {
        "sampled_monotonic_ns": timestamp,
        "process_id": 4242,
        "active_gpu_index": 0,
        "capture": {
            "source": "nvidia-smi",
            "gpu_query": "nvidia-smi --query-gpu=index,name,uuid,memory.total",
            "compute_process_query": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory",
        },
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "uuid": "GPU-test",
                "memory_total_mib": 24564,
            }
        ],
        "compute_processes": [
            {"pid": 4242, "process_name": "python", "used_gpu_memory_mib": 1024}
        ],
        "renderer": {
            "active_gpu_setting": 0,
            "active_renderer_setting": "rtx",
            "enabled_rtx_extensions": ["omni.hydra.rtx"],
        },
    }


def _runtime_summary(*, automatic_claim: bool = False) -> dict[str, object]:
    return {
        "passed": True,
        "scenario": "cartpole-direct",
        "task": "Isaac-Cartpole-Direct-v0",
        "viz": ["kit"],
        "num_envs": 1,
        "steps": 16,
        "executed_steps": 450,
        "backend_before": _physx(),
        "backend_after": _physx(),
        "gui_observation": {
            "mode": "bounded_manual_operator_observation",
            "requested_wall_time_s": 15.0,
            "actual_wall_time_s": 15.01,
            "frame_pacing_s": 1.0 / 30.0,
            "maximum_executed_steps": 10_000,
            "expected_active_gpu_index": 0,
            "gpu_samples": [_gpu_sample(timestamp=100), _gpu_sample(timestamp=200)],
            "operator_attestation_required_after_close": True,
            "automatic_visual_observation_claimed": automatic_claim,
        },
    }


def _seal_runtime(root: Path, summary: dict[str, object]) -> None:
    path = _root() / "scripts" / "rambo" / "finalize_runtime_artifact.py"
    specification = importlib.util.spec_from_file_location("runtime_finalizer_for_m2_gui", path)
    assert specification is not None and specification.loader is not None
    finalizer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(finalizer)
    (root / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    finalizer.finalize_runtime_artifact(root, exit_status=0)


def test_m2_gui_sidecar_binds_a_sealed_physx_runtime(tmp_path: Path) -> None:
    schema = _load_schema()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _seal_runtime(runtime, _runtime_summary())

    sidecar = tmp_path / "attestation"
    written = schema.write_operator_attestation(
        runtime_artifact_dir=runtime,
        attestation_dir=sidecar,
        operator_name="operator",
    )
    result = schema.validate_m2_cartpole_gui_gate(
        runtime_artifact_dir=runtime,
        attestation_dir=sidecar,
    )

    assert written["checksummed_files"] == 1
    assert result["runtime_exit_status"] == 0
    assert result["runtime_evidence"]["gpu_sample_count"] == 2
    assert result["attestation"]["operator_name"] == "operator"


def test_m2_gui_sidecar_rejects_a_runtime_mutated_after_close(tmp_path: Path) -> None:
    schema = _load_schema()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _seal_runtime(runtime, _runtime_summary())
    sidecar = tmp_path / "attestation"
    schema.write_operator_attestation(
        runtime_artifact_dir=runtime,
        attestation_dir=sidecar,
        operator_name="operator",
    )

    summary = json.loads((runtime / "summary.json").read_text(encoding="utf-8"))
    summary["task"] = "forged-task"
    (runtime / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(schema.M2CartpoleGuiArtifactError, match="not sealed and accepted"):
        schema.validate_m2_cartpole_gui_gate(
            runtime_artifact_dir=runtime,
            attestation_dir=sidecar,
        )


def test_m2_gui_attestation_refuses_automatic_visual_claims(tmp_path: Path) -> None:
    schema = _load_schema()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _seal_runtime(runtime, _runtime_summary(automatic_claim=True))

    with pytest.raises(schema.M2CartpoleGuiArtifactError, match="automatic visual observation"):
        schema.write_operator_attestation(
            runtime_artifact_dir=runtime,
            attestation_dir=tmp_path / "attestation",
            operator_name="operator",
        )


def test_m2_gui_runtime_validator_rejects_generic_renderer_core_only() -> None:
    schema = _load_schema()
    summary = _runtime_summary()
    observation = summary["gui_observation"]
    assert isinstance(observation, dict)
    samples = observation["gpu_samples"]
    assert isinstance(samples, list)
    for sample in samples:
        assert isinstance(sample, dict)
        renderer = sample["renderer"]
        assert isinstance(renderer, dict)
        renderer["enabled_rtx_extensions"] = ["omni.kit.renderer.core"]

    with pytest.raises(schema.M2CartpoleGuiArtifactError, match="explicitly RTX-labelled"):
        schema.validate_runtime_summary(summary)


def test_live_gpu_sampler_requires_the_running_process_and_rtx_gpu() -> None:
    schema = _load_schema()

    def fake_nvidia_smi(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[1].startswith("--query-gpu="):
            return subprocess.CompletedProcess(
                command,
                0,
                "0, NVIDIA GeForce RTX 4090, GPU-test, 24564\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "4242, python, 1024\n", "")

    sample = schema.collect_gui_gpu_sample(
        active_gpu_index=0,
        renderer={
            "active_gpu_setting": 0,
            "active_renderer_setting": "rtx",
            "enabled_rtx_extensions": ["omni.hydra.rtx"],
        },
        command_runner=fake_nvidia_smi,
        process_id=4242,
    )

    assert sample["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert sample["compute_processes"][0]["used_gpu_memory_mib"] == 1024


def test_m2_gui_runner_is_dedicated_post_close_and_interactive() -> None:
    root = _root()
    runner = (root / "scripts" / "rambo" / "run_m2_cartpole_gui_gate.sh").read_text(encoding="utf-8")
    recorder = (root / "scripts" / "rambo" / "record_m2_cartpole_gui_attestation.py").read_text(encoding="utf-8")
    validator = (root / "scripts" / "rambo" / "validate_m2_cartpole_gui_gate.py").read_text(encoding="utf-8")
    smoke = (root / "scripts" / "rambo" / "official_physx_smoke.py").read_text(encoding="utf-8")

    assert "run_runtime_artifact.sh" in runner
    assert "--scenario cartpole-direct --num-envs 1 --steps 16" in runner
    assert "--viz kit" in runner
    assert "unset OMNI_KIT_ACCEPT_EULA" in runner
    assert "unknown or unsafe argument" in runner
    assert "sys.stdin.isatty()" in recorder
    assert 'add_argument("--yes"' not in recorder
    assert "interactive_tty_exact_phrase_after_kit_close" in (root / "scripts" / "rambo" / "m2_cartpole_gui_artifact.py").read_text(encoding="utf-8")
    assert "validate_m2_cartpole_gui_gate" in validator
    assert "--viz kit requires --gui-observation-seconds" in smoke
    assert "collect_gui_gpu_sample" in smoke
    assert "automatic_visual_observation_claimed" in smoke
