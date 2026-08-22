"""Minimal Gym registry helpers owned by the RAMBO external extension.

Isaac Lab's pip wheel bundles its example-task extension source, but the
``isaaclab_tasks`` Python package is not installed by the base wheel.  RAMBO
only needs the two small configuration-resolution operations below, so keeping
them here avoids a runtime dependency on that broad, unrelated task package.

The behavior intentionally mirrors Isaac Lab 2.3.2's public task-registry
conventions: a Gym task declares an ``*_cfg_entry_point`` in its registration,
whose value is either a YAML file or a callable/class entry point.
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any


def _task_spec(task_name: str) -> Any:
    """Resolve a Gym task spec while accepting an optional namespace prefix."""

    import gymnasium as gym

    return gym.spec(task_name.split(":")[-1])


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict[str, Any] | object:
    """Load a RAMBO task configuration declared in the Gym registry.

    YAML entry points use the standard ``package.module:file.yaml`` form;
    Python entry points use ``package.module:ConfigClass``.  This deliberately
    stays small because RAMBO does not need Isaac Lab's example-task importer,
    Hydra wiring, or checkpoint-directory helpers.
    """

    cfg_entry_point = _task_spec(task_name).kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        raise ValueError(
            f"Could not find configuration for the environment {task_name!r}. "
            f"Expected Gym registry key {entry_point_key!r}."
        )

    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            try:
                module_name, filename = cfg_entry_point.split(":", maxsplit=1)
            except ValueError as error:
                raise ValueError(
                    "YAML configuration entry points must be absolute paths or "
                    "use 'package.module:file.yaml': "
                    f"{cfg_entry_point!r}"
                ) from error
            module_file = getattr(importlib.import_module(module_name), "__file__", None)
            if module_file is None:
                raise RuntimeError(
                    f"Cannot resolve module-relative YAML configuration from {module_name!r}."
                )
            config_file = os.path.join(os.path.dirname(module_file), filename)

        import yaml

        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as file:
            return yaml.full_load(file)

    if callable(cfg_entry_point):
        cfg_class = cfg_entry_point
    elif isinstance(cfg_entry_point, str):
        try:
            module_name, attribute_name = cfg_entry_point.split(":", maxsplit=1)
        except ValueError as error:
            raise ValueError(
                "Python configuration entry points must use 'package.module:Class': "
                f"{cfg_entry_point!r}"
            ) from error
        cfg_class = getattr(importlib.import_module(module_name), attribute_name)
    else:
        cfg_class = cfg_entry_point

    print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
    # ``inspect`` preserves the error behavior of the upstream helper for a
    # malformed callable while avoiding an unnecessary simulator import here.
    if inspect.isclass(cfg_class) or callable(cfg_class):
        return cfg_class()
    return cfg_class


def parse_env_cfg(
    task_name: str,
    *,
    device: str = "cuda:0",
    num_envs: int | None = None,
    use_fabric: bool | None = None,
) -> object:
    """Instantiate a RAMBO environment config and apply runner overrides."""

    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    if isinstance(cfg, dict):
        raise RuntimeError(
            f"Configuration for the task {task_name!r} is a dictionary; "
            "RAMBO DirectRL tasks require an environment configuration object."
        )

    try:
        cfg.sim.device = device
        if use_fabric is not None:
            cfg.sim.use_fabric = use_fabric
        if num_envs is not None:
            cfg.scene.num_envs = num_envs
    except AttributeError as error:
        raise RuntimeError(
            f"Configuration for the task {task_name!r} is not an Isaac Lab environment config."
        ) from error
    return cfg


__all__ = ("load_cfg_from_registry", "parse_env_cfg")
