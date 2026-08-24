"""The common first-person RGB camera contract for RAMBO task validation."""

from __future__ import annotations

import math
from typing import Any

import warp as wp


# The acceptance rollout is 100 Hz and the physics loop is 500 Hz.  Keeping
# this as an explicit contract makes the 30 s / 3000 policy-step run exactly
# 375 RGB frames, rather than relying on a floating-point sensor clock.
FRONT_RGB_UPDATE_PERIOD_S = 0.08

# The Go2 base is rotated -90 degrees around Y in the biped task.  These
# parent-frame offsets keep the physical camera at (+0.30, 0, +0.08) in the
# initial world frame and pointing along world +X, clear of the chassis.
UPRIGHT_BIPED_FRONT_CAMERA_OFFSET_POS = (0.08, 0.0, -0.30)
# Isaac Lab 3 camera offsets use simulator-facing XYZW quaternions.  This is
# the former WXYZ rotation ``(sqrt(0.5), 0, sqrt(0.5), 0)`` written as XYZW.
UPRIGHT_BIPED_FRONT_CAMERA_OFFSET_ROT = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))


@wp.kernel
def _advance_camera_ticks_kernel(
    elapsed_physics_ticks: wp.array(dtype=wp.int64),
    deadline_crossed: wp.array(dtype=wp.bool),
    elapsed_step_count: wp.int64,
    interval_steps: wp.int64,
) -> None:
    """Advance each camera's exact integer cadence clock."""

    env_id = wp.tid()
    previous_ticks = elapsed_physics_ticks[env_id]
    current_ticks = previous_ticks + elapsed_step_count
    elapsed_physics_ticks[env_id] = current_ticks
    deadline_crossed[env_id] = current_ticks // interval_steps > previous_ticks // interval_steps


@wp.kernel
def _reset_camera_ticks_kernel(
    elapsed_physics_ticks: wp.array(dtype=wp.int64),
    deadline_crossed: wp.array(dtype=wp.bool),
    reset_mask: wp.array(dtype=wp.bool),
) -> None:
    """Reset only the camera clocks selected by an Isaac Lab reset mask."""

    env_id = wp.tid()
    if reset_mask[env_id]:
        elapsed_physics_ticks[env_id] = wp.int64(0)
        deadline_crossed[env_id] = False


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
        from isaaclab.sim import SimulationContext
        from isaaclab.utils.warp import ProxyArray
    except ModuleNotFoundError as error:  # pragma: no cover - target-runtime guard.
        raise RuntimeError("Front RGB cameras require an installed Isaac Lab runtime.") from error

    class RamboExactCadenceCamera(Camera):
        """Official Isaac Lab camera with an exact, commensurate RAMBO clock."""

        def _initialize_impl(self) -> None:
            super()._initialize_impl()
            sim = SimulationContext.instance()
            if sim is None:
                raise RuntimeError("RAMBO exact-cadence camera requires an active SimulationContext")
            self._rambo_physics_timestep_s = float(sim.get_physics_dt())
            self._rambo_cadence_interval_steps = camera_update_interval_steps(
                self.cfg.update_period,
                self._rambo_physics_timestep_s,
            )
            self._rambo_elapsed_physics_ticks = ProxyArray(
                wp.zeros(self.num_instances, dtype=wp.int64, device=self.device)
            )
            self._rambo_deadline_crossed = wp.zeros(self.num_instances, dtype=wp.bool, device=self.device)
            self._rambo_reset_mask = wp.zeros(self.num_instances, dtype=wp.bool, device=self.device)
            # Both Tensor views are zero-copy views of the Warp buffers.  Keep
            # all logical/fancy indexing in Torch, while kernels receive only
            # native Warp arrays and keep their required integer dtypes.
            self._rambo_deadline_crossed_torch = wp.to_torch(self._rambo_deadline_crossed)
            self._rambo_reset_mask_torch = wp.to_torch(self._rambo_reset_mask)

        @property
        def rambo_physics_ticks(self) -> Any:
            """Exact elapsed physics ticks as a public ``ProxyArray`` view.

            Diagnostics must use ``camera.rambo_physics_ticks.torch``.  It is
            deliberately separate from Isaac Lab's internal float timestamp
            buffers, which remain owned by the base sensor implementation.
            """

            return self._rambo_elapsed_physics_ticks

        def reset(self, env_ids=None, env_mask=None) -> None:
            """Reset official Camera state and the independent integer clock."""

            super().reset(env_ids=env_ids, env_mask=env_mask)
            if env_mask is None:
                self._rambo_reset_mask.zero_()
                if env_ids is None:
                    self._rambo_reset_mask_torch.fill_(True)
                else:
                    if isinstance(env_ids, wp.array):
                        reset_ids = wp.to_torch(env_ids)
                    elif isinstance(env_ids, slice):
                        reset_ids = torch.arange(self.num_instances, device=self.device)[env_ids]
                    else:
                        reset_ids = torch.as_tensor(env_ids, device=self.device)
                    # Torch's conventional logical index type is int64.  This
                    # is intentionally not passed into a Warp kernel.
                    reset_ids = reset_ids.to(device=self.device, dtype=torch.long).reshape(-1)
                    self._rambo_reset_mask_torch[reset_ids] = True
                reset_mask = self._rambo_reset_mask
            else:
                reset_mask = env_mask
            wp.launch(
                _reset_camera_ticks_kernel,
                dim=self.num_instances,
                inputs=[
                    self._rambo_elapsed_physics_ticks.warp,
                    self._rambo_deadline_crossed,
                    reset_mask,
                ],
                device=self.device,
            )

        def update(self, dt: float, force_recompute: bool = False) -> None:
            """Advance sensor state with an integer number of physics ticks."""

            if not self.is_initialized:
                return

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
            wp.launch(
                _advance_camera_ticks_kernel,
                dim=self.num_instances,
                inputs=[
                    self._rambo_elapsed_physics_ticks.warp,
                    self._rambo_deadline_crossed,
                    elapsed_step_count,
                    self._rambo_cadence_interval_steps,
                ],
                device=self.device,
            )
            if bool(self._rambo_deadline_crossed_torch.any().item()):
                # Delegate all timestamp/outdated-buffer mutation to the
                # official Camera implementation.  Passing one full sensor
                # period means its float timestamp is never used to decide the
                # cadence; it only records an already-proven due frame.
                super().update(dt=self.cfg.update_period, force_recompute=force_recompute)

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
    offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
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
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
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
        # RAMBO's production camera is attached to an explicitly PhysX-backed
        # scene.  Be equally explicit about the RTX renderer used for its
        # visual contract instead of relying on whichever default renderer a
        # future Isaac Lab build happens to select.
        renderer_cfg=IsaacRtxRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=width,
        height=height,
    )
