"""The common first-person RGB camera contract for RAMBO task validation."""

from __future__ import annotations

import math
from typing import Any


# The acceptance rollout is 100 Hz and the physics loop is 500 Hz.  Keeping
# this as an explicit contract makes the 30 s / 3000 policy-step run exactly
# 375 RGB frames, rather than relying on a floating-point sensor clock.
FRONT_RGB_UPDATE_PERIOD_S = 0.08

# The Go2 base is rotated -90 degrees around Y in the biped task.  These
# parent-frame offsets keep the physical camera at (+0.30, 0, +0.08) in the
# initial world frame and pointing along world +X, clear of the chassis.
UPRIGHT_BIPED_FRONT_CAMERA_OFFSET_POS = (0.08, 0.0, -0.30)
UPRIGHT_BIPED_FRONT_CAMERA_OFFSET_ROT = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


def camera_update_interval_steps(update_period_s: float, physics_timestep_s: float) -> int:
    """Return the exact integer physics cadence for a RAMBO camera.

    Isaac Lab's generic sensor clock stores its timestamps in float32.  At a
    2 ms physics timestep that clock can eventually turn a nominal 40-step
    camera period into an occasional 41-step period.  RAMBO's RGB acceptance
    contract is intentionally commensurate, so reject non-integral periods
    instead of silently drifting away from the declared configuration.
    """

    update_period_s = float(update_period_s)
    physics_timestep_s = float(physics_timestep_s)
    if not math.isfinite(update_period_s) or update_period_s <= 0.0:
        raise ValueError(f"Camera update period must be finite and positive, got {update_period_s!r}")
    if not math.isfinite(physics_timestep_s) or physics_timestep_s <= 0.0:
        raise ValueError(f"Physics timestep must be finite and positive, got {physics_timestep_s!r}")

    interval_steps = round(update_period_s / physics_timestep_s)
    if interval_steps <= 0 or not math.isclose(
        interval_steps * physics_timestep_s,
        update_period_s,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError(
            "RAMBO front-camera update_period must be an integer multiple of physics dt: "
            f"update_period={update_period_s}, physics_dt={physics_timestep_s}"
        )
    return int(interval_steps)


def advance_camera_cadence(
    elapsed_physics_steps: int,
    *,
    elapsed_step_count: int,
    interval_steps: int,
) -> tuple[int, bool]:
    """Advance an integer cadence clock and report whether it crossed a deadline.

    This pure helper mirrors :class:`RamboExactCadenceCamera` and deliberately
    treats a multi-tick scene update as crossing a deadline when the boundary
    lies *inside* the update, not only when it ends exactly on one.
    """

    if elapsed_physics_steps < 0 or elapsed_step_count < 0:
        raise ValueError("Camera cadence counters cannot be negative")
    if interval_steps <= 0:
        raise ValueError("Camera cadence interval must be positive")
    next_physics_steps = elapsed_physics_steps + elapsed_step_count
    due = next_physics_steps // interval_steps > elapsed_physics_steps // interval_steps
    return next_physics_steps, due


_EXACT_CADENCE_CAMERA_TYPE: type[Any] | None = None


def _exact_cadence_camera_type() -> type[Any]:
    """Build the RAMBO-only Camera subclass after Isaac Sim has launched.

    Importing ``isaaclab.sensors`` launches Kit in this runtime, so the class
    must be created lazily.  The resulting object is still an official Isaac
    Lab ``Camera``: only ``SensorBase.update``'s float32 cadence decision is
    replaced with the integer schedule required by this extension.
    """

    global _EXACT_CADENCE_CAMERA_TYPE
    if _EXACT_CADENCE_CAMERA_TYPE is not None:
        return _EXACT_CADENCE_CAMERA_TYPE

    try:
        import torch
        from isaaclab.sensors import Camera
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Front RGB cameras require an installed Isaac Lab runtime.") from error

    class RamboExactCadenceCamera(Camera):
        """Official Isaac Lab camera with an exact, commensurate RAMBO clock."""

        def _initialize_impl(self) -> None:
            super()._initialize_impl()
            self._rambo_physics_timestep_s = float(self._sim_physics_dt)
            self._rambo_cadence_interval_steps = camera_update_interval_steps(
                self.cfg.update_period,
                self._rambo_physics_timestep_s,
            )
            self._rambo_elapsed_physics_steps = torch.zeros(
                self._num_envs,
                dtype=torch.long,
                device=self._device,
            )

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            if env_ids is None:
                env_ids = self._ALL_INDICES
            self._rambo_elapsed_physics_steps[env_ids] = 0

        def update(self, dt: float, force_recompute: bool = False) -> None:
            """Advance sensor state with an integer number of physics ticks."""

            dt = float(dt)
            if not math.isfinite(dt) or dt < 0.0:
                raise ValueError(f"Camera update dt must be finite and non-negative, got {dt!r}")
            elapsed_step_count = round(dt / self._rambo_physics_timestep_s)
            if not math.isclose(
                elapsed_step_count * self._rambo_physics_timestep_s,
                dt,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise RuntimeError(
                    "RAMBO exact-cadence camera received a scene update that is not an integer "
                    f"multiple of physics dt: dt={dt}, physics_dt={self._rambo_physics_timestep_s}"
                )

            # Keep the parent timestamps available for diagnostics and normal
            # buffer bookkeeping, but never use their float32 difference to
            # decide an RGB deadline.
            self._timestamp += dt
            previous_elapsed_steps = self._rambo_elapsed_physics_steps
            self._rambo_elapsed_physics_steps = previous_elapsed_steps + elapsed_step_count
            elapsed_periods = torch.div(
                previous_elapsed_steps,
                self._rambo_cadence_interval_steps,
                rounding_mode="floor",
            )
            updated_periods = torch.div(
                self._rambo_elapsed_physics_steps,
                self._rambo_cadence_interval_steps,
                rounding_mode="floor",
            )
            self._is_outdated |= updated_periods > elapsed_periods
            if force_recompute or self._is_visualizing or self.cfg.history_length > 0:
                self._update_outdated_buffers()

    _EXACT_CADENCE_CAMERA_TYPE = RamboExactCadenceCamera
    return _EXACT_CADENCE_CAMERA_TYPE


def create_front_rgb_camera(camera_cfg: Any) -> Any:
    """Instantiate RAMBO's exact-cadence front camera after Kit starts."""

    return _exact_cadence_camera_type()(camera_cfg)


def make_front_rgb_camera_cfg(
    prim_path: str = "/World/envs/env_.*/Robot/base/front_camera",
    *,
    width: int = 640,
    height: int = 480,
    update_period: float = FRONT_RGB_UPDATE_PERIOD_S,
    offset_pos: tuple[float, float, float] = (0.30, 0.0, 0.08),
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> Any:
    """Create RAMBO's shared base-mounted front RGB camera configuration.

    Both offsets remain relative to the base prim. ``offset_rot`` uses Isaac
    Lab's ``world`` camera convention. The biped task pitches the Go2 base by
    90 degrees to stand it upright, so it supplies transformed local position
    and rotation offsets to keep the physical camera in front of the chassis.
    """

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
    except ModuleNotFoundError as error:  # pragma: no cover - static-tooling path.
        raise RuntimeError("Front RGB cameras require an installed Isaac Lab runtime.") from error

    # Keep the public Isaac Lab configuration at the physical 0.08 s period.
    # ``create_front_rgb_camera`` consumes the same value using integer
    # physics ticks, so configuration inspection and rendered-frame cadence
    # agree exactly.
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=width,
        height=height,
    )
