"""Checkpoint validation and deterministic rollout helpers for RAMBO."""

from .checkpoints import (
    CHECKPOINT_CONTRACTS,
    CheckpointContract,
    CheckpointContractError,
    assert_state_dict_equal,
    contract_for_task,
    load_verified_checkpoint,
    restore_runner,
    sha256_file,
    validate_checkpoint_schema,
)

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
