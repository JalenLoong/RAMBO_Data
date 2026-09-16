---
id: DATA-004
type: change
status: completed
source_map: []
---
# DATA-004: replace the center-only pilot with a straight full-footprint push

## Scope
The user invalidated the prior 8.04s/402-action center-only pilot. It is excluded from the current valid DATA-004 pilot and all future training selections. No new archive/compatibility path or Git history rewrite is needed. Reuse DATA-004 and its existing architecture/data contract.

## Required change
Derive the intended face, center, normal and contact reference from actual Box 05 mesh dimensions. Use deterministic front-facing geometry and an expert reference trajectory approaching the face center along its normal, then mainly translating toward the goal. Record commanded contact target, actual FL EEF, box pose/yaw and footprint for review. Existing native9, quadruped RAMBO and force-zero transform remain unchanged.

Success requires the complete projected box XY footprint inside the goal and robot not fallen, for 3 consecutive 50Hz policy ticks. Contact remains diagnostic/unknown and never gates success. Preserve dual50Hz policy RGB, observer25Hz, controller100Hz, physics500Hz, sensor-native contact, authoritative timestamps, N/N+1 and terminal-before-reset. Run the full raw/canonical/real VAE-T5/WAM reload chain on exactly one replacement valid technical pilot. No batch demos/training/remote jobs or automatic commit/push.

## Progress
- [x] Reopened existing work; invalidated old pilot in current acceptance.
- [x] Implement dimension-derived geometry, expert and footprint/hold logic; CPU checks.
- [x] Revalidate actual terminal-before-reset after changes.
- [x] Collect and review the replacement pilot; retain only the new valid pilot in current selection.
- [x] Full raw/canonical/WAM/cache validation and three-view report.

## Evidence
Current attempt directory: workspace `runs/audit/v2/DATA-004/straight-push-20260916T042259Z`.
All behavior and data gates for the replacement have now passed; user visual review accepted; additional collection requires separate authorization. Old pilot success remains only an observation under its discarded criterion, not current acceptance.

User steering: widest face is optional; retain the reviewed narrower side if useful. Straight FL-foot pushing toward the goal is the priority.

## Current DATA-004 valid replacement pilot

The prior 8.04s/402-action center-only pilot is invalid for current DATA-004 and excluded from future training. No legacy/compatibility path or history rewrite was introduced. The only current valid technical pilot is the new straight-push replacement under task profile `push-box-v2-2`.

The reviewed narrow `local_x_min` face is allowed. Actual mesh bounds define its center/normal and all corners. Box dimensions are about30.0763x17.4538x22.6464cm. Nominal geometric center is(0.690382,0.142000,0.133114)m, face center(0.540000,0.142000,0.133114)m and normal(-1,0,0), yaw0. Robot stays at native reset XY origin; FL's neutral lateral command aligns the push line. No yaw randomization, controller replacement or model architecture change.

The scripted expert approaches until the measured face distance is suitable, then approaches normally and advances the body with about0.38m FL forward reference. World-axis references are transformed into the existing projected-com native9 frame. This is demonstration generation using simulator state, not a learned-policy result. Intended face/commanded contact/actual FL EEF/box pose/yaw are recorded in audit-only task-review metadata.

Success requires the entire projected source-box bounding volume inside goal x=[0.95,1.35], y=[-0.10,0.40]m and robot not fallen for3 consecutive50Hz policy ticks. Four XY envelope corners are computed from all8 pose-transformed3D corners, including roll/pitch. Contact remains diagnostic/unknown and never gates success; yaw is reported without a success threshold.

The valid pilot lasts9.08s with454 completed actions. Full containment first appears at physics timestamp9036000000ns and policy timestamp9040000000ns; success ticks are9.04/9.06/9.08s. Terminal footprint X=[0.956747,1.297620],Y=[0.038445,0.257470]m is fully inside. Forward displacement0.436802m, net lateral displacement0.005958m. Yaw range[-12.405,1.875]degrees, final-7.396degrees; residual rotation is disclosed.

Real terminal-before-reset/reset isolation passed on the current implementation and final pilot. Raw/canonical validation and MP4 decode passed:455 dual50Hz Raw boundaries,454 canonical rows/frames plus separate terminal state/RGB,908 controller steps,4540 physics steps/4541 snapshots,909 actual sensor updates,228 observer25Hz frames. Requested/executed native9 and zero desired force/external wrench retained. Observer is never a model input or success ground truth. DATA-002 schema/profile bytes are unchanged.

WAM reads113 sampled RGB frames per view with inherited source timestamps, encodes real frozen VAE/T5, and reloads latents[1,48,29,24,20], actions/mask[1,9,29,16,1], text[1,512,4096]. Initial mask is false.448 actions enter complete groups;6 tail actions and their timestamps remain in Raw/canonical and are explicitly reported by cache metadata. Normalizer is diagnostic identity, not training statistics.

Evidence: workspace `runs/audit/v2/DATA-004/straight-push-20260916T042259Z`. Two failed development collection attempts are excluded (spawn XY/controller reference mismatch; expert overreach and lateral deflection). Only the replacement is the current valid pilot. WAM73 and RAMBO220 CPU tests passed; real data/model gates were executed separately. The user accepted this replacement pilot and authorized DATA-004 commit/push. Further batch demonstrations require separate authorization; formal training/release and model server closed-loop remain not_run.
