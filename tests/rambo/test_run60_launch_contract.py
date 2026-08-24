"""Shell-only tests for the fail-closed RAMBO simulator wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fake_runtime_venv(tmp_path: Path) -> Path:
    """Make the tiny filesystem shape ``run.sh`` checks without importing torch."""

    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable))
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    package_root = venv / "lib" / f"python{python_version}" / "site-packages"
    torch_lib = package_root / "torch" / "lib"
    nvrtc_lib = package_root / "nvidia" / "cuda_nvrtc" / "lib"
    torch_lib.mkdir(parents=True)
    nvrtc_lib.mkdir(parents=True)
    (torch_lib / "libtorch_cuda_linalg.so").touch()
    (nvrtc_lib / "libnvrtc-builtins.so.12.8").touch()
    return venv


def _run_wrapper(script: str, *arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_root() / "scripts" / "rambo" / script), *arguments],
        cwd=_root(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("placeholder.py",), "exactly one explicit --viz"),
        (("placeholder.py", "--viz", "rerun"), "--viz only accepts none or kit"),
        (("placeholder.py", "--viz", "none", "--viz=kit"), "exactly one explicit --viz"),
        (("placeholder.py", "--visualizer", "none"), "--visualizer is forbidden"),
    ),
)
def test_run60_requires_one_canonical_visualizer(arguments: tuple[str, ...], message: str) -> None:
    result = _run_wrapper("run60.sh", *arguments)

    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    "forbidden",
    (
        "--headless",
        "--headless=true",
        "--physics=newton",
        "--physics-backend=physx",
        "--newton",
        "--use_newton_actuators=true",
        "--use-newton-actuators=true",
        "--physx-manager=override",
        "--backend=physx",
        "--experience=unaudited.kit",
        "--kit_args=--/physics/backend=newton",
    ),
)
def test_run60_rejects_unsafe_launcher_overrides(forbidden: str) -> None:
    result = _run_wrapper("run60.sh", "placeholder.py", "--viz", "none", forbidden)

    assert result.returncode == 2
    assert "RAMBO launch contract violation" in result.stderr


@pytest.mark.parametrize(
    ("viz_arguments", "expected_viz"),
    ((("--viz", "none"), "none"), (("--viz=kit",), "kit")),
)
def test_run60_forwards_only_canonical_viz_to_python(
    tmp_path: Path, viz_arguments: tuple[str, ...], expected_viz: str
) -> None:
    child = tmp_path / "child.py"
    child.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--viz', choices=('none', 'kit'), required=True)\n"
        "parser.add_argument('--artifact-label', required=True)\n"
        "args = parser.parse_args()\n"
        "print(f'child-viz={args.viz};label={args.artifact_label}')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["RAMBO_VENV"] = str(_fake_runtime_venv(tmp_path))

    result = _run_wrapper(
        "run60.sh",
        str(child),
        *viz_arguments,
        "--artifact-label",
        "isaaclab_newton-transitive-package",
        environment=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"child-viz={expected_viz};label=isaaclab_newton-transitive-package"


def test_run_runtime_artifact_rejects_unsafe_arguments_before_python_or_finalization(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    result = _run_wrapper(
        "run_runtime_artifact.sh",
        "placeholder.py",
        "--output-dir",
        str(artifact),
        "--viz",
        "none",
        "--headless",
    )

    assert result.returncode == 2
    assert "--headless is forbidden" in result.stderr
    assert not artifact.exists()


def test_generic_run_wrapper_remains_usable_without_simulator_visualizer(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["RAMBO_VENV"] = str(_fake_runtime_venv(tmp_path))

    result = _run_wrapper("run.sh", "-c", "print('generic-python-ok')", environment=environment)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "generic-python-ok"
