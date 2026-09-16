---
id: DATA-005
type: exec-plan
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
- [x] Source changes and CPU tests; new real terminal gate passed.
- [x] Three pilots all passed behavior and full data path; durations9.08/9.12/9.08s. Ten-demo stage authorized by the original user request and pilot-gate.json.
- [x] Only after pilots pass: ten successful demonstrations and per-episode QA.
- [x] Multi-episode canonical, episode-level split, WAM frozen cache/reload and review report.

## Recovery and evidence
Use new versioned workspace layers and immutable actual attempt manifests. Refuse overwrites, retain failed attempts, do not silently repair invalid records. Evidence: workspace `runs/audit/v2/DATA-005/20260916T055512Z`. Local W&B offline tracks actual runtime attempts. Additional collection beyond this scope or model training requires a later request.

## Outcomes and evidence

## DATA-005 bounded collection result

The user authorized three repeatability pilots and, after their gates passed, ten successful small-XY-variation demonstrations. All 3 pilots and 10 demonstrations passed the real acquisition, terminal-before-reset/reset isolation, independent whole-footprint/not-fallen behavior checks, Raw/canonical media validation and real frozen VAE/T5 cache reload. The predefined seed42 scenarios vary only initial box XY; the DATA-004 task profile, expert/controller, robot native origin, asset and cameras are unchanged. No yaw randomization or contact gate was added.

The ten demonstration episodes contain 4567 actions, with durations 9.08–9.20s. Net lateral displacement is 0.56–1.23cm; final box yaw is -11.40–-6.96 degrees. Residual rotation remains disclosed and is not a task-success gate. Contact is explicitly unknown/invalid diagnostic evidence.

The demonstration collection has explicit whole-episode train/validation/test splits of8/1/1. Pilots, old center-only DATA-004 records and failures are absent from this collection. Merging copies videos/terminal images without further encoding, rebases only global row/episode indices and keeps episode timestamps local and authoritative. Per-episode provenance/contact sidecars are validated independently. The single-export train alias is not an SFT split; `meta/split.json` is authoritative for the merged collection.

WAM fits a non-diagnostic affine q01/q99 normalizer on the 3655 actions from the8 train episodes only, with no clipping, scale1 for constant dimensions and force offsets0/scales1. Validation/test episodes do not affect fitting. VAE temporal state resets per episode; observer never enters model data. Sampling inherits canonical/terminal timestamps. Incomplete tail groups remain in source data and are explicitly listed in cache metadata; the initial action mask is false.

Evidence: workspace `runs/audit/v2/DATA-005/20260916T055512Z/README.md` and `acceptance.json`. Versioned path configs are `pilots-paths.json` and `demonstrations-paths.json` there. This is bounded scripted-expert data acceptance, not learned-policy evaluation, generalization statistics or permission to start SFT. Training, remote jobs, learned-policy server integration and formal release remain not_run. DATA-005 is not committed or pushed.

CPU evidence: WAM75 tests; RAMBO220 passed, one PyArrow module skipped in simulator environment and one CUDA test deselected. Actual real-media/GPU integration was validated separately. No failed DATA-005 acquisition attempts.

## Discovery

Single-episode exports inherited a pilot-only descriptive split label. Their original metadata is preserved; the merged collection explicitly binds each episode to meta/split.json and keeps the original text as source_split_semantics. It does not infer train membership from that transport alias. The exporter wording is corrected for future demonstration exports after this batch, while sealed exports are retained.

## Subsequent user acceptance —2026-09-16

The user confirmed that the current ten demonstrations meet the standard. Recorded separately in workspace evidence `user-review-accepted.json`; original acquisition/technical acceptance records remain unchanged. No further collection, training, remote work or commit/push was authorized by this review.
