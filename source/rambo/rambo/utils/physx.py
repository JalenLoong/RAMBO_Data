"""Fail-closed PhysX selection and runtime verification for RAMBO.

The imports that construct a PhysX configuration intentionally live inside the
functions below.  RAMBO's metadata-only import path must remain safe before
``AppLauncher`` has initialized Kit.
"""

from __future__ import annotations

from typing import Any, Sequence


PHYSX_CFG_FQN = "isaaclab_physx.physics.physx_manager_cfg.PhysxCfg"
PHYSX_MANAGER_FQN = "isaaclab_physx.physics.physx_manager.PhysxManager"
PHYSX_MANAGER_FACTORY_FQN = "isaaclab_physx.physics.physx_manager:PhysxManager"


def _fqn(value: Any) -> str:
    """Return a stable fully-qualified identity for either a class or object."""

    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def configure_physx(cfg: Any) -> Any:
    """Explicitly select PhysX and disable Newton actuators on a RAMBO config.

    This is deliberately not a default/fallback: assigning a new ``PhysxCfg``
    prevents a caller, registry override, or future Isaac Lab default from
    silently selecting another physics manager.
    """

    try:
        from isaaclab_physx.physics import PhysxCfg
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("RAMBO requires the Isaac Lab PhysX package after AppLauncher starts") from error

    sim_cfg = getattr(cfg, "sim", None)
    if sim_cfg is None:
        raise RuntimeError("RAMBO environment config does not expose a simulation configuration")
    sim_cfg.physics = PhysxCfg()
    sim_cfg.use_newton_actuators = False
    return cfg


def assert_physx_runtime(sim: Any) -> dict[str, Any]:
    """Verify the active manager is exactly PhysX, never merely a package import.

    Fully-qualified identities are compared instead of ``isinstance`` because
    ``AppLauncher`` can temporarily unload/re-import Isaac Lab modules while
    Kit starts.  Any manager that is not the exact PhysX manager fails closed.
    """

    cfg_root = getattr(sim, "cfg", None)
    cfg = getattr(cfg_root, "physics", None)
    manager = getattr(sim, "physics_manager", None)
    cfg_fqn = _fqn(cfg)
    manager_fqn = _fqn(manager)
    if cfg_fqn != PHYSX_CFG_FQN:
        raise RuntimeError(f"RAMBO physics config is not PhysxCfg: {cfg_fqn}")
    if manager_fqn != PHYSX_MANAGER_FQN:
        raise RuntimeError(f"RAMBO active physics manager is not PhysxManager: {manager_fqn}")

    configured_manager = getattr(cfg, "class_type", None)
    configured_manager_fqn = _fqn(configured_manager)
    if configured_manager_fqn not in {PHYSX_MANAGER_FQN, PHYSX_MANAGER_FACTORY_FQN}:
        raise RuntimeError(
            "RAMBO PhysX config does not resolve to PhysxManager: "
            f"{configured_manager_fqn}"
        )

    use_newton_actuators = getattr(cfg_root, "use_newton_actuators", None)
    if use_newton_actuators is not False:
        raise RuntimeError("RAMBO requires use_newton_actuators=False")

    physics_dt = getattr(sim, "get_physics_dt", None)
    return {
        "requested_cfg": cfg_fqn,
        "actual_manager": manager_fqn,
        "configured_manager": configured_manager_fqn,
        "use_newton_actuators": False,
        "physics_dt_s": None if physics_dt is None else float(physics_dt()),
        "device": str(getattr(sim, "device", "unknown")),
    }


def assert_physx_environment(env: Any) -> dict[str, Any]:
    """Verify PhysX for a Gym environment or a RAMBO/CRL2 wrapper."""

    base_env = getattr(env, "unwrapped", env)
    sim = getattr(base_env, "sim", None)
    if sim is None:
        raise RuntimeError("RAMBO environment does not expose its SimulationContext")
    return assert_physx_runtime(sim)


def has_cli_option(name: str, argv: Sequence[str]) -> bool:
    """Return whether ``argv`` contains an explicit long option spelling."""

    return any(argument == name or argument.startswith(f"{name}=") for argument in argv)


def validate_rambo_visualizer_args(parser: Any, args: Any, argv: Sequence[str]) -> list[str]:
    """Restrict RAMBO launchers to audited Kit/no-visualizer choices.

    ``AppLauncher`` accepts visualizers that can choose other physics paths.
    RAMBO production, smoke, validation, and teleop launchers intentionally
    expose only the no-visualizer and Kit choices.
    """

    if has_cli_option("--headless", argv):
        parser.error("--headless is forbidden; use --viz none")
    if not (has_cli_option("--viz", argv) or has_cli_option("--visualizer", argv)):
        parser.error("--viz must be explicitly set to none or kit")
    visualizer = getattr(args, "visualizer", None)
    if visualizer is None or visualizer == ["none"]:
        selection = ["none"]
    elif visualizer == ["kit"]:
        selection = ["kit"]
    else:
        parser.error("only --viz none and --viz kit are permitted for RAMBO PhysX execution")
    if getattr(args, "experience", None):
        parser.error("--experience is forbidden for RAMBO PhysX execution")
    if getattr(args, "kit_args", None):
        parser.error("--kit_args is forbidden for RAMBO PhysX execution")
    if getattr(args, "livestream", -1) not in (-1, 0):
        parser.error("--livestream may only be omitted or set to 0 for RAMBO PhysX execution")
    args.fast_shutdown = True
    args.livestream = 0
    return selection


__all__ = [
    "PHYSX_CFG_FQN",
    "PHYSX_MANAGER_FACTORY_FQN",
    "PHYSX_MANAGER_FQN",
    "assert_physx_environment",
    "assert_physx_runtime",
    "configure_physx",
    "has_cli_option",
    "validate_rambo_visualizer_args",
]
