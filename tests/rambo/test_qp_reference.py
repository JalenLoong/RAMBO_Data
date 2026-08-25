"""Simulator-free contracts for canonical Isaac 6 QP reference bundles."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "rambo" / "qp_reference.py"
    spec = importlib.util.spec_from_file_location("rambo_qp_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    primal = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 10.0]], dtype=np.float64)
    inputs = {
        "qp_quadratic_matrix": np.eye(6, dtype=np.float64)[None, :, :],
        "qp_linear_vector": np.zeros((1, 6), dtype=np.float64),
        "inequality_matrix": np.zeros((1, 2, 6), dtype=np.float64),
        "inequality_rhs": np.ones((1, 2), dtype=np.float64),
        "equality_matrix": np.empty((0,), dtype=np.float32),
        "equality_rhs": np.empty((0,), dtype=np.float32),
        "contact_mask": np.array([[True, True]], dtype=np.bool_),
    }
    outputs = {
        "qp_primal_ground_reaction_force": primal,
        "solved_spatial_acceleration_body": np.zeros((1, 6), dtype=np.float32),
        "qp_cost": np.array([0.25], dtype=np.float32),
        "desired_joint_torque": np.zeros((1, 6), dtype=np.float32),
        "physx_foot_contact_force_world": np.zeros((1, 2, 3), dtype=np.float32),
        "post_step_done": np.array([False]),
    }
    return inputs, outputs


def _metadata() -> dict[str, object]:
    return {
        "task": "Isaac-RAMBO-Quadruped-Go2-v0",
        "seed": 42,
        "checkpoint_sha256": "a" * 64,
        "observation_dim": 405,
        "action_dim": 18,
        "simulation_dt_s": 0.002,
        "decimation": 5,
        "joint_order": ["joint"],
        "foot_order": ["FL", "FR"],
    }


def test_write_and_validate_canonical_qp_reference(tmp_path: Path) -> None:
    module = _load_module()
    inputs, outputs = _fixture_arrays()
    destination = tmp_path / "reference"

    summary = module.write_qp_reference_bundle(
        destination, metadata=_metadata(), inputs=inputs, outputs=outputs
    )
    result = module.validate_qp_reference_bundle(destination)

    assert result["passed"] is True
    assert result["candidate_compared"] is False
    assert summary["residual_metrics"]["inequality_max_positive_violation"] == 0.0
    assert {path.name for path in destination.iterdir()} == {
        "metadata.json",
        "input_snapshot.npz",
        "output_snapshot.npz",
        "summary.json",
        "checksums.sha256",
    }
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["physics_backend"] == "PhysX"
    assert metadata["use_newton_actuators"] is False
    assert metadata["capture_semantics"]["additional_qp_solves"] == 0


def test_candidate_comparison_uses_explicit_output_tolerance_and_exact_mask(tmp_path: Path) -> None:
    module = _load_module()
    inputs, outputs = _fixture_arrays()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    module.write_qp_reference_bundle(reference, metadata=_metadata(), inputs=inputs, outputs=outputs)

    candidate_outputs = {name: value.copy() for name, value in outputs.items()}
    candidate_outputs["desired_joint_torque"][0, 0] = 5.0e-6
    module.write_qp_reference_bundle(
        candidate, metadata=_metadata(), inputs=inputs, outputs=candidate_outputs
    )
    assert module.validate_qp_reference_bundle(reference, candidate)["candidate_compared"] is True

    bad = tmp_path / "bad"
    bad_inputs = {name: value.copy() for name, value in inputs.items()}
    bad_inputs["contact_mask"][0, 0] = False
    module.write_qp_reference_bundle(bad, metadata=_metadata(), inputs=bad_inputs, outputs=outputs)
    with pytest.raises(module.QpReferenceError, match="contact_mask exceeds tolerance"):
        module.validate_qp_reference_bundle(reference, bad)


def test_qp_reference_rejects_overwrite_and_checksum_tampering(tmp_path: Path) -> None:
    module = _load_module()
    inputs, outputs = _fixture_arrays()
    destination = tmp_path / "reference"
    module.write_qp_reference_bundle(destination, metadata=_metadata(), inputs=inputs, outputs=outputs)
    with pytest.raises(module.QpReferenceError, match="overwrite"):
        module.write_qp_reference_bundle(destination, metadata=_metadata(), inputs=inputs, outputs=outputs)

    (destination / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.QpReferenceError, match="checksum mismatch"):
        module.validate_qp_reference_bundle(destination)
