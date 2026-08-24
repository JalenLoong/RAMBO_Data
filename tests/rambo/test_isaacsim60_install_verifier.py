"""Pure-Python regression checks for the pinned-install metadata verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_verifier():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "verify_isaacsim60_install.py"
    spec = importlib.util.spec_from_file_location("rambo_isaacsim60_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "requested",
    (
        "isaaclab_newton[all]==6.1.14",
        "isaaclab-newton[all]==6.1.14",
        "newton[sim]==6.1.14",
        "isaaclab_physx[newton]==6.1.14",
        "isaaclab-physx[all]==6.1.14",
        "isaaclab[all]==6.1.14",
    ),
)
def test_optional_newton_extras_are_rejected(tmp_path: Path, requested: str) -> None:
    verifier = _load_verifier()
    source = tmp_path / "requirements.in"
    source.write_text(f"{requested}\n", encoding="utf-8")

    with pytest.raises(verifier.VerificationError, match="optional Isaac Lab Newton extras"):
        verifier._validate_no_optional_newton_extras(("test input", source))


def test_bare_transitive_newton_distribution_remains_allowed(tmp_path: Path) -> None:
    verifier = _load_verifier()
    source = tmp_path / "requirements.in"
    source.write_text("isaaclab_newton==6.1.14\n", encoding="utf-8")

    verifier._validate_no_optional_newton_extras(("test input", source))
