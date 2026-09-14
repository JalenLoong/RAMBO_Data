"""Static provenance and integration contracts for the LiftBasket asset."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rambo.assets import (
    BASKET_ASSET_ROOT,
    BASKET_MASS_KG,
    BASKET_ROOT_HEIGHT_ABOVE_BOTTOM_M,
    BASKET_USD_PATH,
    BASKET_INITIAL_ORIENTATION_XYZW,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def test_lift_basket_asset_payload_matches_pinned_manifest() -> None:
    manifest = json.loads((BASKET_ASSET_ROOT / "asset_manifest.json").read_text(encoding="utf-8"))

    assert BASKET_USD_PATH.is_file()
    assert manifest["asset"]["runtime_role"] == "visual_and_source_authored_physics"
    assert manifest["asset"]["authored_mass_kg"] == 0.53
    assert manifest["asset"]["runtime_task_mass_kg"] == BASKET_MASS_KG == 0.53
    assert (
        manifest["rambo_wrapper"]["root_height_above_physical_bottom_m"]
        == BASKET_ROOT_HEIGHT_ABOVE_BOTTOM_M
    )
    for relative_path, expected in manifest["files"].items():
        path = BASKET_ASSET_ROOT / relative_path
        assert path.stat().st_size == expected["size_bytes"]
        assert _sha256(path) == expected["sha256"]


def test_lift_basket_uses_source_authored_physics_without_proxies() -> None:
    assert BASKET_ROOT_HEIGHT_ABOVE_BOTTOM_M == 0.0
    x, y, z, w = BASKET_INITIAL_ORIENTATION_XYZW
    assert math.isclose(x*x + y*y + z*z + w*w, 1.0)
    # Physical up is mesh-local -Z. Source orientation must survive the
    # stage-up conversion; the old (.5,.5,.5,.5) left the basket horizontal.
    assert math.isclose(2*(x*x + y*y) - 1, 1.0)
    source = (
        REPOSITORY_ROOT / "source/rambo/rambo/assets/lift_basket.py"
    ).read_text(encoding="utf-8")
    assert "sim_utils.UsdFileCfg" in source
    assert "CuboidCfg" not in source
    assert "CollisionProxy" not in source
    assert "RemoveAPI" not in source
    assert "SetCollisionEnabledAttr" not in source


def test_lift_basket_task_uses_free_source_rigid_body() -> None:
    source = (
        REPOSITORY_ROOT
        / "source/rambo/rambo/tasks/direct/rambo_quadruped/object_tasks_env.py"
    ).read_text(encoding="utf-8")

    assert 'spawn_lift_basket_asset(f"{self._root}/basket", center)' in source
    assert "basket_vertical_guide" not in source
    assert "PrismaticJoint" not in source
    assert "basket_u_handle_crossbar" not in source
    assert "primary_position = (0.75, 0.15, 0.02)" in source
    assert "pos[:, 2]" in source
    assert "self._primary_rest_pose[:, 2]" in source
