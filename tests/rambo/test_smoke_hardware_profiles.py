"""CPU-side contracts for the RAMBO smoke hardware profiles."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


_SMOKE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rambo" / "smoke.py"
_SPEC = importlib.util.spec_from_file_location("rambo_smoke_hardware_profiles", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SMOKE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SMOKE)


def _profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _SMOKE._add_hardware_profile_argument(parser)
    return parser


def test_blackwell_accepts_cc_12_0_with_sm_120() -> None:
    assert _SMOKE._validate_hardware_profile("blackwell", (12, 0), ("sm_89", "sm_120")) == "sm_120"


def test_blackwell_rejects_ada_capability() -> None:
    with pytest.raises(
        RuntimeError,
        match=(
            r"hardware_profile=blackwell: expected capability=\(12, 0\); "
            r"actual capability=\(8, 9\)"
        ),
    ):
        _SMOKE._validate_hardware_profile("blackwell", (8, 9), ("sm_89", "sm_120"))


def test_ada_accepts_cc_8_9_with_sm_89() -> None:
    assert _SMOKE._validate_hardware_profile("ada", (8, 9), ("sm_89", "sm_120")) == "sm_89"


def test_ada_rejects_blackwell_capability() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"hardware_profile=ada: expected capability=\(8, 9\); actual capability=\(12, 0\)",
    ):
        _SMOKE._validate_hardware_profile("ada", (12, 0), ("sm_89", "sm_120"))


@pytest.mark.parametrize(
    ("capability", "required_architecture"),
    (((8, 9), "sm_89"), ((12, 0), "sm_120")),
)
def test_compatible_accepts_matching_compiled_architecture(
    capability: tuple[int, int], required_architecture: str
) -> None:
    assert (
        _SMOKE._validate_hardware_profile(
            "compatible", capability, ("sm_89", "sm_120")
        )
        == required_architecture
    )


def test_compatible_rejects_missing_compiled_architecture() -> None:
    with pytest.raises(RuntimeError, match=r"required architecture=sm_89"):
        _SMOKE._validate_hardware_profile("compatible", (8, 9), ("sm_120",))


def test_compatible_rejects_unconvertible_capability() -> None:
    with pytest.raises(RuntimeError, match="Cannot reliably convert CUDA compute capability"):
        _SMOKE._validate_hardware_profile("compatible", (12, 10), ("sm_1210",))


def test_invalid_hardware_profile_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        _profile_parser().parse_args(("--hardware-profile", "hopper"))


def test_default_hardware_profile_remains_blackwell() -> None:
    assert _profile_parser().parse_args(()).hardware_profile == "blackwell"
