---
id: UPSTREAM-INFRA-001
type: provenance
status: accepted
source_map:
  - scripts/rambo/preview_lift_cameras.py
---
# Simulator preview provenance

The bounded camera preview was selectively migrated from WAM-Policy commit ad4023b,
`src/wam_rambo/preview_lift_cameras.py`, to RAMBO_Data during INFRA-002.
It uses direct simulator imports and a fixed zero-force 9D diagnostic command; old collector and reference-program dependencies are removed.
It publishes run evidence only. Dataset v2 recording and production camera coverage are later work.
The remaining RAMBO controller and task source derives from this repository's main commit 66605a8.
