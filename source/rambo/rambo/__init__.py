"""RAMBO external Isaac Lab extension.

The package intentionally degrades to metadata-only import outside an Isaac Lab
runtime.  This keeps static tooling and pure-Python utility tests usable without
Omniverse installed; :func:`register_tasks` performs the actual Gym registration
once Isaac Lab is importable.
"""

from __future__ import annotations

from pathlib import Path

from ._runtime import kit_runtime_ready


def _read_version() -> str:
    """Read package metadata without imposing a runtime dependency on ``toml``."""

    try:
        import tomllib

        with (Path(__file__).resolve().parent.parent / "config" / "extension.toml").open("rb") as file:
            return str(tomllib.load(file)["package"]["version"])
    except (ImportError, OSError, KeyError, ValueError):
        return "0.0.0"


__version__ = _read_version()


def register_tasks() -> bool:
    """Register RAMBO Gym tasks when optional simulator dependencies are present.

    Returns:
        ``True`` when task modules were imported and registration was attempted;
        ``False`` when this interpreter is intentionally missing Isaac Lab's
        optional simulator stack.
    """

    # Do not merely *probe* the installed Isaac Lab package here: importing it
    # from a plain Python process starts Kit.  Task registration is called again
    # by the RAMBO entry points after ``AppLauncher`` has initialized Kit.
    if not kit_runtime_ready():
        return False

    try:
        from . import tasks as _tasks  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".", 1)[0] in {"isaaclab", "isaaclab_assets", "gymnasium", "omni"}:
            return False
        raise
    return True


# This remains useful for extension discovery inside Kit, while normal Python
# tooling receives a metadata-only import with no Isaac Sim side effects.
register_tasks()

__all__ = ["__version__", "kit_runtime_ready", "register_tasks"]
