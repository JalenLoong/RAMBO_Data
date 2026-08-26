"""Trusted-checkpoint contracts and restoration helpers for RAMBO policies.

The pretrained RAMBO policies were written by CRL2 before PyTorch changed the
default ``torch.load`` safety mode.  The files are intentionally accepted only
after they match a task-specific SHA-256 allow-list; this makes the explicit
``weights_only=False`` load both deliberate and auditable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


class CheckpointContractError(RuntimeError):
    """Raised when a checkpoint is not the expected policy for a RAMBO task."""


@dataclass(frozen=True)
class CheckpointContract:
    """Immutable compatibility contract for one released RAMBO policy."""

    task: str
    mode: str
    sha256: str
    observation_dim: int
    action_dim: int
    iteration: int
    normalizer_count: int
    min_base_height: float
    gravity_target: tuple[float, float, float]
    max_orientation_error: float


CHECKPOINT_CONTRACTS: Final[dict[str, CheckpointContract]] = {
    "Isaac-RAMBO-Quadruped-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Quadruped-Go2-v0",
        mode="quadruped",
        sha256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706",
        observation_dim=405,
        action_dim=18,
        iteration=2000,
        normalizer_count=100003840,
        min_base_height=0.1,
        gravity_target=(0.0, 0.0, -1.0),
        max_orientation_error=0.75,
    ),
    "Isaac-RAMBO-Biped-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Biped-Go2-v0",
        mode="biped",
        sha256="c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6",
        observation_dim=435,
        action_dim=18,
        iteration=4000,
        normalizer_count=100003840,
        min_base_height=0.3,
        gravity_target=(-1.0, 0.0, 0.0),
        max_orientation_error=0.8,
    ),
    "Isaac-RAMBO-Quadruped-Button-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Quadruped-Button-Go2-v0",
        mode="quadruped_loco_manip",
        sha256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706",
        observation_dim=405,
        action_dim=18,
        iteration=2000,
        normalizer_count=100003840,
        min_base_height=0.1,
        gravity_target=(0.0, 0.0, -1.0),
        max_orientation_error=0.75,
    ),
    # These three procedural Dataset V1 tasks preserve the released
    # quadruped's 405D observation and 18D residual-action interface.  Their
    # contracts intentionally pin the exact same immutable model_2000.pt as
    # the Button task; only task-side 9D scripted commands and rigid scene
    # assets differ.
    "Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Quadruped-Lift-Basket-Go2-v0",
        mode="quadruped_loco_manip",
        sha256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706",
        observation_dim=405,
        action_dim=18,
        iteration=2000,
        normalizer_count=100003840,
        min_base_height=0.1,
        gravity_target=(0.0, 0.0, -1.0),
        max_orientation_error=0.75,
    ),
    "Isaac-RAMBO-Quadruped-Pull-Object-Into-Basket-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Quadruped-Pull-Object-Into-Basket-Go2-v0",
        mode="quadruped_loco_manip",
        sha256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706",
        observation_dim=405,
        action_dim=18,
        iteration=2000,
        normalizer_count=100003840,
        min_base_height=0.1,
        gravity_target=(0.0, 0.0, -1.0),
        max_orientation_error=0.75,
    ),
    "Isaac-RAMBO-Quadruped-Shoot-Ball-Into-Goal-Go2-v0": CheckpointContract(
        task="Isaac-RAMBO-Quadruped-Shoot-Ball-Into-Goal-Go2-v0",
        mode="quadruped_loco_manip",
        sha256="1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706",
        observation_dim=405,
        action_dim=18,
        iteration=2000,
        normalizer_count=100003840,
        min_base_height=0.1,
        gravity_target=(0.0, 0.0, -1.0),
        max_orientation_error=0.75,
    ),
}


def contract_for_task(task: str) -> CheckpointContract:
    """Return the released-policy contract for ``task`` or raise a clear error."""

    try:
        return CHECKPOINT_CONTRACTS[task]
    except KeyError as exc:
        supported = ", ".join(sorted(CHECKPOINT_CONTRACTS))
        raise CheckpointContractError(
            f"Unsupported RAMBO task {task!r}; expected one of: {supported}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    """Hash a file without retaining its potentially large contents in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointContractError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise CheckpointContractError(f"{name} must be a tensor-like value with a shape")
    try:
        return tuple(int(dimension) for dimension in shape)
    except TypeError as exc:
        raise CheckpointContractError(f"{name} has an invalid shape: {shape!r}") from exc


def _require_tensor_shape(
    state: Mapping[str, Any], key: str, expected: tuple[int, ...], state_name: str
) -> None:
    if key not in state:
        raise CheckpointContractError(f"{state_name} is missing required key {key!r}")
    actual = _shape(state[key], f"{state_name}[{key!r}]")
    if actual != expected:
        raise CheckpointContractError(
            f"{state_name}[{key!r}] shape mismatch: expected {expected}, got {actual}"
        )


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CheckpointContractError(f"{name} must be an integer, got bool")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError(f"{name} must be an integer, got {value!r}") from exc
    if integer != value:
        raise CheckpointContractError(f"{name} must be an exact integer, got {value!r}")
    return integer


def validate_checkpoint_schema(
    checkpoint: Mapping[str, Any], contract: CheckpointContract
) -> None:
    """Validate the architecture and all invariants needed for deterministic replay."""

    checkpoint = _as_mapping(checkpoint, "checkpoint")
    required_keys = {
        "policy_dict",
        "value_dict_0",
        "obs_normalizer",
        "obs_normalizer_count",
        "optimizer_state_dict",
        "iteration",
        "infos",
    }
    missing_keys = sorted(required_keys - set(checkpoint))
    if missing_keys:
        raise CheckpointContractError(f"checkpoint is missing required keys: {missing_keys}")

    policy = _as_mapping(checkpoint["policy_dict"], "policy_dict")
    value = _as_mapping(checkpoint["value_dict_0"], "value_dict_0")
    normalizer = _as_mapping(checkpoint["obs_normalizer"], "obs_normalizer")
    _as_mapping(checkpoint["optimizer_state_dict"], "optimizer_state_dict")

    observation_dim = contract.observation_dim
    action_dim = contract.action_dim
    for key, expected in {
        "log_std": (action_dim,),
        "policy_latent_net.0.weight": (512, observation_dim),
        "policy_latent_net.0.bias": (512,),
        "policy_latent_net.2.weight": (256, 512),
        "policy_latent_net.2.bias": (256,),
        "policy_latent_net.4.weight": (128, 256),
        "policy_latent_net.4.bias": (128,),
        "action_mean_net.weight": (action_dim, 128),
        "action_mean_net.bias": (action_dim,),
    }.items():
        _require_tensor_shape(policy, key, expected, "policy_dict")

    for key, expected in {
        "value.0.weight": (512, observation_dim),
        "value.0.bias": (512,),
        "value.2.weight": (256, 512),
        "value.2.bias": (256,),
        "value.4.weight": (128, 256),
        "value.4.bias": (128,),
        "value.6.weight": (1, 128),
        "value.6.bias": (1,),
    }.items():
        _require_tensor_shape(value, key, expected, "value_dict_0")

    for key in ("_mean", "_var", "_std"):
        _require_tensor_shape(normalizer, key, (1, observation_dim), "obs_normalizer")

    iteration = _require_integer(checkpoint["iteration"], "checkpoint['iteration']")
    if iteration != contract.iteration:
        raise CheckpointContractError(
            f"checkpoint iteration mismatch: expected {contract.iteration}, got {iteration}"
        )
    normalizer_count = _require_integer(
        checkpoint["obs_normalizer_count"], "checkpoint['obs_normalizer_count']"
    )
    if normalizer_count != contract.normalizer_count:
        raise CheckpointContractError(
            "checkpoint normalizer count mismatch: "
            f"expected {contract.normalizer_count}, got {normalizer_count}"
        )


def load_verified_checkpoint(path: str | Path, contract: CheckpointContract) -> dict[str, Any]:
    """Verify the allow-listed file and deserialize it once on CPU.

    ``weights_only=False`` is intentional: CRL2 checkpoints contain historical
    metadata in addition to tensors.  It is executed only after an exact local
    SHA-256 match with the immutable contract above.
    """

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise CheckpointContractError(f"Checkpoint does not exist or is not a file: {checkpoint_path}")

    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != contract.sha256:
        raise CheckpointContractError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {contract.sha256}, got {actual_sha256} for {checkpoint_path}"
        )

    # Delay importing torch so static inspection of the RAMBO extension does
    # not require an Isaac Sim runtime to be installed.
    import torch

    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch versions before the weights_only argument used the same
        # historical (pickle-enabled) behavior.  The SHA guard remains in
        # force for local legacy inspection.
        loaded = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = dict(_as_mapping(loaded, "checkpoint"))
    validate_checkpoint_schema(checkpoint, contract)
    return checkpoint


def assert_state_dict_equal(name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Require byte-exact tensor restoration, including state-dict key layout."""

    actual = _as_mapping(actual, f"{name} actual state")
    expected = _as_mapping(expected, f"{name} expected state")
    if set(actual) != set(expected):
        raise CheckpointContractError(
            f"{name} state keys mismatch: actual={sorted(actual)}, expected={sorted(expected)}"
        )

    import torch

    mismatches: list[str] = []
    for key in actual:
        actual_value = actual[key]
        expected_value = expected[key]
        if not hasattr(actual_value, "detach") or not hasattr(expected_value, "detach"):
            if actual_value != expected_value:
                mismatches.append(key)
            continue
        if not torch.equal(actual_value.detach().cpu(), expected_value.detach().cpu()):
            mismatches.append(key)
    if mismatches:
        raise CheckpointContractError(f"{name} tensor mismatch: {sorted(mismatches)}")


def restore_runner(
    runner: Any,
    checkpoint: Mapping[str, Any],
    *,
    load_values: bool,
    verify: bool = False,
) -> Any:
    """Restore a CRL2 runner from a pre-verified in-memory checkpoint.

    This deliberately does not call ``PPO.load(path)``: the legacy method
    deserializes a second time, lacks an explicit map location, and historically
    omitted the empirical-normalizer count.
    """

    checkpoint = _as_mapping(checkpoint, "checkpoint")
    try:
        runner.policy.load_state_dict(checkpoint["policy_dict"])
    except AttributeError as exc:
        raise CheckpointContractError("runner does not expose a CRL2 policy") from exc

    if load_values:
        values = getattr(runner, "values", None)
        if not values:
            raise CheckpointContractError("runner does not expose CRL2 value networks")
        values[0].load_state_dict(checkpoint["value_dict_0"])

    normalizer = getattr(runner, "obs_normalizer", None)
    if normalizer is None:
        raise CheckpointContractError(
            "RAMBO policy checkpoint requires empirical normalization, but runner has none"
        )
    normalizer.load_state_dict(checkpoint["obs_normalizer"])
    normalizer.count = _require_integer(
        checkpoint["obs_normalizer_count"], "checkpoint['obs_normalizer_count']"
    )
    runner.current_iteration = _require_integer(checkpoint["iteration"], "checkpoint['iteration']")

    if verify:
        assert_state_dict_equal("policy", runner.policy.state_dict(), checkpoint["policy_dict"])
        if load_values:
            assert_state_dict_equal("value_0", runner.values[0].state_dict(), checkpoint["value_dict_0"])
        assert_state_dict_equal("obs_normalizer", normalizer.state_dict(), checkpoint["obs_normalizer"])
        if normalizer.count != checkpoint["obs_normalizer_count"]:
            raise CheckpointContractError(
                "obs_normalizer count did not restore: "
                f"expected {checkpoint['obs_normalizer_count']}, got {normalizer.count}"
            )

    return checkpoint["infos"]


__all__ = (
    "CHECKPOINT_CONTRACTS",
    "CheckpointContract",
    "CheckpointContractError",
    "assert_state_dict_equal",
    "contract_for_task",
    "load_verified_checkpoint",
    "restore_runner",
    "sha256_file",
    "validate_checkpoint_schema",
)
