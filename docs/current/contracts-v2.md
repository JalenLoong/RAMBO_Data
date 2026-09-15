---
id: CURRENT-DATA-INTERFACE-V2
type: current
status: accepted
source_map:
  - source/rambo/rambo/dataset_v2/validation.py
  - source/rambo/rambo/dataset_v2/media.py
  - source/rambo/rambo/dataset_v2/spec/profile.json
---
# Current data interface: LeRobot 50Hz

Work ID: **V2-DATA-CONTRACT**.

[Authoritative dataset contract](adaptation_v2_dataset_contract.md) defines `wam-quadruped-v2.1.0`, LeRobot codebase `v2.1` and `lerobot==0.3.3` compatibility.
Raw/canonical policy RGB are 50Hz; model RGB is deterministically sampled to 12.5Hz by WAM. Observer25Hz is monitor-only.
Canonical is MP4/Parquet/metadata, with N paired rows/N video frames and separate terminal PNG/state; no terminal action.
The current model remains native9, two views, identity high-level filter and zero desired force.

## APIs and dependencies

Use `DataPaths.from_config` with explicit dataset_root/canonical_root/cache_root and ffmpeg/ffprobe paths.
`validate_canonical(root, tools=...)` and `validate_raw(root, tools=...)` check the new format without publishing.
CLI: `python -m rambo.dataset_v2 --config PATHS_JSON` (add `--raw` for raw audit).
PyArrow is an offline-tool dependency. The workspace policy/file-tool interpreter already provides LeRobot/PyArrow;
the Isaac interpreter remains unchanged and does not currently include PyArrow. Simulator CPU regressions still use its locked interpreter.

## Historical boundary

The [2.0.0 PNG reference](../work/archive/V2-CONTRACTS/contracts-v2.0-reference.md) and its original CPU acceptance remain historical evidence.
`contracts_v2` is retained for that explicit audit and the existing pure runtime protocol; do not use its old PNG dataset schema as the new default.
Dataset format versioning does not rewrite the existing runtime handshake/profile bundle. Real 50Hz acquisition, observer creation,
recorder, production conversion, actual sensor cadence, VAE/T5 generation and simulator/server wiring remain not_run.

## Producer-side helpers

`finalize_video` is a single-writer close/validate/rename helper. Failed validation retains partial files and
returns invalid status for the caller to persist. It neither repairs videos nor publishes episodes.
Synthetic fixture construction exists for CPU compatibility tests only; it is not a production Raw-to-Canonical converter.

CPU acceptance evidence: workspace `runs/audit/v2/V2-DATA-CONTRACT/20260915T141714Z`. WAM55 tests, data tools37 tests, simulator CPU198 tests and10 cross-namespace cases passed.
