---
id: DATA-005
type: change
status: completed
source_map: []
---
# DATA-005: three pilots followed by ten successful demonstrations

## User authorization and boundaries
The user accepted and published DATA-004, then explicitly authorized the annotated next steps: three pilots (nominal repeat, small box-forward offset, small box-lateral offset), and after their behavior/data gates pass, ten successful demonstrations with small geometry variation and episode-level splits. No training, remote jobs, controller/model/camera changes, yaw randomization or automatic commit/push.

Keep the native robot reset XY origin, source Box05 asset, expert, zero-force native9 chain, full-footprint containment for3 policy ticks, fixed policy/observer cameras,50/100/500Hz timing and DATA-002 format. Old center-only and failed episodes are excluded. Contact stays diagnostic/unknown. Every episode must pass terminal-before-reset/reset isolation and raw/canonical validation.

## Implementation
Expose bounded box XY offsets and purpose/work identity in the existing one-environment collector. Freeze three pilot scenarios: (dx,dy)=(0,0),(+0.025,0),(0,+0.010)m. After pilot checks, collect ten distinct predefined offsets within dx=[-0.020,+0.025]m, dy=[-0.010,+0.010]m, fixed seed42/yaw0. Keep failures as failures; only successful reviewed episodes enter the demonstration package.

Merge canonical episodes without re-encoding their videos; preserve source timestamps, terminal sidecars and per-episode unknown-contact metadata. Create an explicit8/1/1 train/validation/test split by whole demonstration episode; pilots are outside it. WAM builds real frozen VAE/T5 caches with reset per episode. Any fitted action normalizer uses train episodes only; no optimizer/training job runs.

## Gates
- [x] Source changes and CPU tests; new terminal gate for current collector.
- [x] Three real pilots: task success, behavior review, raw/canonical/reader/actual cache checks.
- [x] Only after pilots pass: ten successful demonstrations and per-episode QA.
- [x] Multi-episode canonical, episode-level split, WAM frozen cache/reload and review report.

## Recovery and evidence
Use new versioned workspace layers and immutable actual attempt manifests. Refuse overwrites, retain failed attempts, do not silently repair invalid records. Evidence: workspace `runs/audit/v2/DATA-005/20260916T055512Z`. Local W&B offline tracks actual runtime attempts. Additional collection beyond this scope or model training requires a later request.
