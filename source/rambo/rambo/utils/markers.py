"""RAMBO-only colored sphere marker configurations."""

from __future__ import annotations

from rambo._runtime import kit_runtime_ready


if kit_runtime_ready():
    # Importing installed Isaac Lab from a normal Python process starts Kit and
    # can prompt for the EULA. Marker definitions are only useful after
    # AppLauncher, so static utilities intentionally receive inert sentinels.
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkersCfg
    except ModuleNotFoundError:  # pragma: no cover - incomplete runtime install.
        sim_utils = None
        VisualizationMarkersCfg = None
else:  # pragma: no cover - static import without Omniverse.
    sim_utils = None
    VisualizationMarkersCfg = None


def _sphere(color: tuple[float, float, float]):
    if sim_utils is None or VisualizationMarkersCfg is None:
        return None
    return VisualizationMarkersCfg(
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        }
    )


GREEN_SPHERE_MARKER_CFG = _sphere((0.0, 1.0, 0.0))
BLUE_SPHERE_MARKER_CFG = _sphere((0.0, 0.0, 1.0))
YELLOW_SPHERE_MARKER_CFG = _sphere((1.0, 1.0, 0.0))
CYAN_SPHERE_MARKER_CFG = _sphere((0.0, 1.0, 1.0))
