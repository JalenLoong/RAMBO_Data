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

Work ID: **DATA-002**.

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

The [2.0.0 PNG reference](../work/archive/DATA-001/contracts-v2.0-reference.md) and its original CPU acceptance remain historical evidence.
`contracts_v2` is retained for that explicit audit and the existing pure runtime protocol; do not use its old PNG dataset schema as the new default.
Dataset format versioning does not rewrite the existing runtime handshake/profile bundle. The original DATA-002 attempt did not exercise real acquisition. DATA-004 now validates one real50Hz/25Hz pilot, recorder, canonical conversion and VAE/T5 cache. Learned-model sampler/server wiring and formal release remain not_run.

## Producer-side helpers

`finalize_video` is a single-writer close/validate/rename helper. Failed validation retains partial files and
returns invalid status for the caller to persist. It neither repairs videos nor publishes episodes.
Synthetic fixture construction exists for CPU compatibility tests only; it is not a production Raw-to-Canonical converter.

CPU acceptance evidence: workspace `runs/audit/v2/V2-DATA-CONTRACT/20260915T141714Z`. WAM55 tests, data tools37 tests, simulator CPU198 tests and10 cross-namespace cases passed.

## DATA-003 clarifications

Factor-4 sampling selects canonical rows only. Sampled timestamps inherit authoritative `simulation_time_ns`; a selected terminal uses terminal metadata. Tail action timestamps also inherit source rows. Nominal 20ms is validation only, never a timestamp generator. The actual implementation is WAM `CanonicalReader.alignment` (the fixed-factor resampler role); no separate `FixedFactorResampler` class is required.

Approach-and-Push Box / Box 05 is approved under DATA-004. Geometry and success are fixed in the task profile; contact is diagnostic only.

For approved Push Box, diagnostic pair evidence distinguishes FL-object and body-object only when available. Body-aggregated force alone cannot attribute object motion to the FL manipulation leg. Missing pair observations are unknown, not confirmed non-contact. Success is full footprint in goal and not fallen for 3 consecutive policy ticks; pair evidence does not gate success/collection. Unknown is explicit in the required diagnostic sidecar; existing file schema/profile hashes are unchanged.


## Current DATA-004 valid replacement pilot

The prior 8.04s/402-action center-only pilot is invalid for current DATA-004 and excluded from future training. No legacy/compatibility path or history rewrite was introduced. The DATA-004 valid technical pilot is the straight-push replacement under task profile `push-box-v2-2`.

The reviewed narrow `local_x_min` face is allowed. Actual mesh bounds define its center/normal and all corners. Box dimensions are about30.0763x17.4538x22.6464cm. Nominal geometric center is(0.690382,0.142000,0.133114)m, face center(0.540000,0.142000,0.133114)m and normal(-1,0,0), yaw0. Robot stays at native reset XY origin; FL's neutral lateral command aligns the push line. No yaw randomization, controller replacement or model architecture change.

The scripted expert approaches until the measured face distance is suitable, then approaches normally and advances the body with about0.38m FL forward reference. World-axis references are transformed into the existing projected-com native9 frame. This is demonstration generation using simulator state, not a learned-policy result. Intended face/commanded contact/actual FL EEF/box pose/yaw are recorded in audit-only task-review metadata.

Success requires the entire projected source-box bounding volume inside goal x=[0.95,1.35], y=[-0.10,0.40]m and robot not fallen for3 consecutive50Hz policy ticks. Four XY envelope corners are computed from all8 pose-transformed3D corners, including roll/pitch. Contact remains diagnostic/unknown and never gates success; yaw is reported without a success threshold.

The valid pilot lasts9.08s with454 completed actions. Full containment first appears at physics timestamp9036000000ns and policy timestamp9040000000ns; success ticks are9.04/9.06/9.08s. Terminal footprint X=[0.956747,1.297620],Y=[0.038445,0.257470]m is fully inside. Forward displacement0.436802m, net lateral displacement0.005958m. Yaw range[-12.405,1.875]degrees, final-7.396degrees; residual rotation is disclosed.

Real terminal-before-reset/reset isolation passed on the current implementation and final pilot. Raw/canonical validation and MP4 decode passed:455 dual50Hz Raw boundaries,454 canonical rows/frames plus separate terminal state/RGB,908 controller steps,4540 physics steps/4541 snapshots,909 actual sensor updates,228 observer25Hz frames. Requested/executed native9 and zero desired force/external wrench retained. Observer is never a model input or success ground truth. DATA-002 schema/profile bytes are unchanged.

WAM reads113 sampled RGB frames per view with inherited source timestamps, encodes real frozen VAE/T5, and reloads latents[1,48,29,24,20], actions/mask[1,9,29,16,1], text[1,512,4096]. Initial mask is false.448 actions enter complete groups;6 tail actions and their timestamps remain in Raw/canonical and are explicitly reported by cache metadata. Normalizer is diagnostic identity, not training statistics.

Evidence: workspace `runs/audit/v2/DATA-004/straight-push-20260916T042259Z`. Two failed development collection attempts are excluded (spawn XY/controller reference mismatch; expert overreach and lateral deflection). Within DATA-004, only the replacement is valid. WAM73 and RAMBO220 CPU tests passed; real data/model gates were executed separately. The user accepted this replacement pilot and authorized DATA-004 commit/push. The later DATA-005 batch was separately authorized; formal training/release and model server closed-loop remain not_run.


## DATA-005 completed

Three pilots and ten small-variation demonstrations passed the full data path. The DATA-004 controller/task/cameras remain fixed. Whole-episode splits are8/1/1; no training or publication was performed. See [DATA-005 execution](../work/archive/DATA-005/plan.md), [episode-isolation decision](../decisions/ADR-0009.md) and the [current data contract](adaptation_v2_dataset_contract.md).
