"""Spawn the OASIS basket with its source-authored physics unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any


BASKET_ASSET_ROOT = (
    Path(__file__).resolve().parent / "third_party" / "oasis" / "lift_basket"
)
BASKET_USD_PATH = BASKET_ASSET_ROOT / "basket.usd"
BASKET_MASS_KG = 0.53
BASKET_ROOT_HEIGHT_ABOVE_BOTTOM_M = 0.0
# Source Y-up -> task Z-up, followed by +90 degrees of world-Z yaw so the
# arch plane faces the Go2 approaching from the negative-X side.
BASKET_INITIAL_ORIENTATION_XYZW = (0.5, 0.5, 0.5, 0.5)


def spawn_lift_basket_asset(
    prim_path: str,
    position: tuple[float, float, float],
) -> Any:
    """Spawn the audited OASIS USD as one source-authored rigid object.

    Simulator imports stay local so provenance and package-data tests can
    inspect this module without starting Kit.
    """

    if not BASKET_USD_PATH.is_file():
        raise FileNotFoundError(f"LiftBasket visual asset is missing: {BASKET_USD_PATH}")

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    asset_cfg = sim_utils.UsdFileCfg(usd_path=str(BASKET_USD_PATH))
    asset_cfg.func(
        prim_path,
        asset_cfg,
        translation=position,
        # Coordinate conversion only: rotate the source Y-up visual, collision
        # mesh, rigid body, mass properties, and physics material together.
        orientation=BASKET_INITIAL_ORIENTATION_XYZW,
    )

    return RigidObject(
        RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position,
                rot=BASKET_INITIAL_ORIENTATION_XYZW,
            ),
        )
    )


__all__ = [
    "BASKET_ASSET_ROOT",
    "BASKET_MASS_KG",
    "BASKET_ROOT_HEIGHT_ABOVE_BOTTOM_M",
    "BASKET_USD_PATH",
    "BASKET_INITIAL_ORIENTATION_XYZW",
    "spawn_lift_basket_asset",
]
