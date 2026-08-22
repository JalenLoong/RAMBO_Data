"""Released-policy compatibility contracts that do not need Isaac Sim."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from rambo.validation.checkpoints import (  # noqa: E402
    CHECKPOINT_CONTRACTS,
    CheckpointContractError,
    contract_for_task,
    load_verified_checkpoint,
    sha256_file,
)


_POLICY_ROOT = Path(os.environ.get("RAMBO_CHECKPOINT_ROOT", "/workspace/rambo-go2-policies"))


@pytest.mark.parametrize(
    ("task", "relative_path"),
    (
        ("Isaac-RAMBO-Quadruped-Go2-v0", Path("quadruped/model_2000.pt")),
        ("Isaac-RAMBO-Biped-Go2-v0", Path("biped/model_4000.pt")),
    ),
)
def test_released_checkpoint_matches_its_dual_mode_contract(task: str, relative_path: Path) -> None:
    """Hash and deserialize the real policy only when the local input exists."""

    checkpoint_path = _POLICY_ROOT / relative_path
    if not checkpoint_path.is_file():
        pytest.skip(f"released checkpoint is not available locally: {checkpoint_path}")

    contract = contract_for_task(task)
    assert sha256_file(checkpoint_path) == contract.sha256
    checkpoint = load_verified_checkpoint(checkpoint_path, contract)

    assert checkpoint["iteration"] == contract.iteration
    assert checkpoint["obs_normalizer_count"] == contract.normalizer_count
    assert checkpoint["policy_dict"]["log_std"].shape == (contract.action_dim,)
    assert checkpoint["policy_dict"]["policy_latent_net.0.weight"].shape == (
        512,
        contract.observation_dim,
    )


def test_contracts_keep_both_modes_as_explicit_first_class_entries() -> None:
    quadruped = contract_for_task("Isaac-RAMBO-Quadruped-Go2-v0")
    loco_manip = contract_for_task("Isaac-RAMBO-Quadruped-Button-Go2-v0")
    biped = contract_for_task("Isaac-RAMBO-Biped-Go2-v0")

    assert set(CHECKPOINT_CONTRACTS) == {quadruped.task, loco_manip.task, biped.task}
    assert (quadruped.mode, quadruped.observation_dim, quadruped.action_dim) == ("quadruped", 405, 18)
    assert (loco_manip.mode, loco_manip.observation_dim, loco_manip.action_dim) == (
        "quadruped_loco_manip",
        405,
        18,
    )
    assert loco_manip.sha256 == quadruped.sha256
    assert (biped.mode, biped.observation_dim, biped.action_dim) == ("biped", 435, 18)
    assert quadruped.sha256 != biped.sha256


def test_untrusted_checkpoint_is_rejected_before_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "model_2000.pt"
    path.write_bytes(b"not a trusted PyTorch checkpoint")

    with pytest.raises(CheckpointContractError, match="SHA-256 mismatch"):
        load_verified_checkpoint(path, contract_for_task("Isaac-RAMBO-Quadruped-Go2-v0"))
