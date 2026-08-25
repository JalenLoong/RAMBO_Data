#!/usr/bin/env python3
"""Write and validate immutable RAMBO Isaac 6 QP regression bundles.

The module is simulator-independent.  Runtime capture is performed by the
existing finite PhysX policy smoke; this module only serializes copied NumPy
arrays and validates them without starting Kit or selecting a physics backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_FILES = ("metadata.json", "input_snapshot.npz", "output_snapshot.npz", "summary.json")
EXACT_ARRAYS = frozenset({"contact_mask", "post_step_done"})
OUTPUT_ARRAYS = frozenset(
    {
        "qp_primal_ground_reaction_force",
        "solved_spatial_acceleration_body",
        "qp_cost",
        "desired_joint_torque",
        "physx_foot_contact_force_world",
        "post_step_policy_observation",
        "post_step_reward",
    }
)
TOLERANCES = {
    "exact": {"rtol": 0.0, "atol": 0.0},
    "input": {"rtol": 1.0e-5, "atol": 1.0e-6},
    "output": {"rtol": 1.0e-4, "atol": 1.0e-5},
    "residual": {"rtol": 1.0e-4, "atol": 1.0e-7},
}


class QpReferenceError(RuntimeError):
    """Raised when a QP reference bundle is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QpReferenceError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_metrics(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    record: dict[str, Any] = {"dtype": str(array.dtype), "shape": list(array.shape)}
    if np.issubdtype(array.dtype, np.number) and array.size:
        finite = bool(np.isfinite(array).all())
        record.update(
            {
                "finite": finite,
                "min": float(array.min()) if finite else None,
                "max": float(array.max()) if finite else None,
                "abs_max": float(np.abs(array).max()) if finite else None,
            }
        )
    return record


def _normalized_arrays(values: Mapping[str, Any], label: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in sorted(values.items()):
        _require(isinstance(name, str) and name, f"{label} array name must be non-empty")
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            _require(bool(np.isfinite(array).all()), f"{label}.{name} contains NaN or Inf")
        arrays[name] = array
    _require(bool(arrays), f"{label} snapshot must contain arrays")
    return arrays


def _residual_metrics(inputs: Mapping[str, np.ndarray], outputs: Mapping[str, np.ndarray]) -> dict[str, float]:
    primal = np.asarray(outputs["qp_primal_ground_reaction_force"], dtype=np.float64)
    matrix = np.asarray(inputs["inequality_matrix"], dtype=np.float64)
    rhs = np.asarray(inputs["inequality_rhs"], dtype=np.float64)
    inequality = np.einsum("bij,bj->bi", matrix, primal) - rhs
    quadratic = np.asarray(inputs["qp_quadratic_matrix"], dtype=np.float64)
    linear = np.asarray(inputs["qp_linear_vector"], dtype=np.float64)
    objective = 0.5 * np.einsum("bi,bij,bj->b", primal, quadratic, primal) + np.einsum(
        "bi,bi->b", linear, primal
    )
    equality_matrix = np.asarray(inputs["equality_matrix"])
    equality_rhs = np.asarray(inputs["equality_rhs"])
    equality_max = 0.0
    if equality_matrix.size:
        equality = np.einsum("bij,bj->bi", equality_matrix, primal) - equality_rhs
        equality_max = float(np.abs(equality).max())
    return {
        "inequality_max_positive_violation": float(np.maximum(inequality, 0.0).max()),
        "inequality_residual_abs_max": float(np.abs(inequality).max()),
        "equality_residual_abs_max": equality_max,
        "objective_min": float(objective.min()),
        "objective_max": float(objective.max()),
    }


def write_qp_reference_bundle(
    destination: Path,
    *,
    metadata: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one fresh, checksummed canonical QP reference directory."""

    root = destination.expanduser().resolve()
    _require(root.is_absolute(), "QP reference destination must be absolute")
    _require(not root.exists(), f"refusing to overwrite QP reference directory: {root}")
    input_arrays = _normalized_arrays(inputs, "input")
    output_arrays = _normalized_arrays(outputs, "output")
    for name in (
        "qp_quadratic_matrix",
        "qp_linear_vector",
        "inequality_matrix",
        "inequality_rhs",
        "equality_matrix",
        "equality_rhs",
        "contact_mask",
    ):
        _require(name in input_arrays, f"input snapshot is missing {name}")
    for name in (
        "qp_primal_ground_reaction_force",
        "solved_spatial_acceleration_body",
        "qp_cost",
        "desired_joint_torque",
        "physx_foot_contact_force_world",
    ):
        _require(name in output_arrays, f"output snapshot is missing {name}")

    root.mkdir(parents=True)
    metadata_record = dict(metadata)
    metadata_record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "physics_backend": "PhysX",
            "use_newton_actuators": False,
            "optional_isaaclab_newton_extras_requested": False,
            "capture_semantics": {
                "additional_policy_calls": 0,
                "additional_env_steps": 0,
                "additional_qp_solves": 0,
                "controller_or_rng_mutation": False,
                "desired_wrench": (
                    "RAMBO QP has no standalone desired-wrench tensor; the snapshot records the exact "
                    "desired spatial acceleration and FL desired/reaction force inputs instead."
                ),
                "physx_contact_force_timing": "read after the same first env.step that executed the captured QP",
            },
            "comparison_tolerances": TOLERANCES,
        }
    )
    _write_json(root / "metadata.json", metadata_record)
    np.savez(root / "input_snapshot.npz", **input_arrays)
    np.savez(root / "output_snapshot.npz", **output_arrays)
    residuals = _residual_metrics(input_arrays, output_arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "physics_backend": "PhysX",
        "input_arrays": {name: _array_metrics(value) for name, value in input_arrays.items()},
        "output_arrays": {name: _array_metrics(value) for name, value in output_arrays.items()},
        "residual_metrics": residuals,
        "stationarity_residual": {
            "available": False,
            "reason": "qpth returns the primal solution but RAMBO does not retain solver dual variables",
        },
        "comparison_tolerances": TOLERANCES,
    }
    _write_json(root / "summary.json", summary)
    (root / "checksums.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in REQUIRED_FILES), encoding="utf-8"
    )
    validate_qp_reference_bundle(root)
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise QpReferenceError(f"cannot read valid JSON from {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (FileNotFoundError, OSError, ValueError) as error:
        raise QpReferenceError(f"cannot read QP snapshot {path}: {error}") from error


def _validate_checksums(root: Path) -> None:
    manifest = root / "checksums.sha256"
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise QpReferenceError(f"checksum manifest is missing: {manifest}") from error
    records: dict[str, str] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, f"malformed checksum line: {line!r}")
        digest, name = parts
        _require(name in REQUIRED_FILES and name not in records, f"unexpected checksum target: {name}")
        _require(len(digest) == 64, f"malformed SHA-256 for {name}")
        records[name] = digest
    _require(set(records) == set(REQUIRED_FILES), "checksum manifest does not cover the four canonical files")
    for name, digest in records.items():
        _require(_sha256(root / name) == digest, f"checksum mismatch: {name}")


def _array_tolerance(name: str) -> Mapping[str, float]:
    if name in EXACT_ARRAYS:
        return TOLERANCES["exact"]
    if name in OUTPUT_ARRAYS:
        return TOLERANCES["output"]
    return TOLERANCES["input"]


def _compare_arrays(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray], label: str) -> None:
    _require(set(reference) == set(candidate), f"{label} array names differ from reference")
    for name, expected in reference.items():
        actual = candidate[name]
        _require(expected.shape == actual.shape, f"{label}.{name} shape differs from reference")
        tolerance = _array_tolerance(name)
        _require(
            bool(np.allclose(actual, expected, rtol=tolerance["rtol"], atol=tolerance["atol"], equal_nan=False)),
            f"{label}.{name} exceeds tolerance rtol={tolerance['rtol']} atol={tolerance['atol']}",
        )


def validate_qp_reference_bundle(reference_dir: Path, candidate_dir: Path | None = None) -> dict[str, Any]:
    """Validate one bundle, and optionally compare another capture against it."""

    reference = reference_dir.expanduser().resolve()
    _require(reference.is_dir(), f"QP reference directory does not exist: {reference}")
    _validate_checksums(reference)
    metadata = _load_json(reference / "metadata.json")
    summary = _load_json(reference / "summary.json")
    _require(metadata.get("schema_version") == SCHEMA_VERSION, "unsupported metadata schema")
    _require(metadata.get("physics_backend") == "PhysX", "QP reference is not explicitly PhysX")
    _require(metadata.get("use_newton_actuators") is False, "QP reference does not disable Newton actuators")
    _require(summary.get("passed") is True, "QP reference summary is not passing")
    inputs = _load_npz(reference / "input_snapshot.npz")
    outputs = _load_npz(reference / "output_snapshot.npz")
    recomputed = _residual_metrics(inputs, outputs)
    for name, expected in summary.get("residual_metrics", {}).items():
        _require(
            bool(np.isclose(recomputed[name], expected, **TOLERANCES["residual"])),
            f"stored residual metric differs: {name}",
        )

    result: dict[str, Any] = {
        "passed": True,
        "reference_dir": str(reference),
        "input_array_count": len(inputs),
        "output_array_count": len(outputs),
        "residual_metrics": recomputed,
        "candidate_compared": False,
    }
    if candidate_dir is not None:
        candidate = candidate_dir.expanduser().resolve()
        _require(candidate.is_dir(), f"QP candidate directory does not exist: {candidate}")
        _validate_checksums(candidate)
        candidate_metadata = _load_json(candidate / "metadata.json")
        signature_fields = (
            "task",
            "seed",
            "checkpoint_sha256",
            "observation_dim",
            "action_dim",
            "simulation_dt_s",
            "decimation",
            "joint_order",
            "foot_order",
        )
        for field in signature_fields:
            _require(candidate_metadata.get(field) == metadata.get(field), f"candidate metadata differs: {field}")
        _require(candidate_metadata.get("physics_backend") == "PhysX", "candidate is not explicitly PhysX")
        _compare_arrays(inputs, _load_npz(candidate / "input_snapshot.npz"), "input")
        _compare_arrays(outputs, _load_npz(candidate / "output_snapshot.npz"), "output")
        result.update({"candidate_compared": True, "candidate_dir": str(candidate)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path)
    args = parser.parse_args()
    try:
        result = validate_qp_reference_bundle(args.reference_dir, args.candidate_dir)
    except QpReferenceError as error:
        print(f"QP_REFERENCE_VALIDATION_FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    print("QP_REFERENCE_VALIDATION_SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
