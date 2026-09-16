---
id: DATA-006
type: change
status: completed
source_map: []
---
# DATA-006: authorized expansion from10 to50 demonstrations

The user explicitly approved retaining DATA-005's ten episodes and acquiring40 additional successful demonstrations, without a training prerequisite or another asset/pilot gate. New episodes use Box05/01/04 counts20/10/10, four quarter-turn headings ten each, four goal-border colors ten each, actual-geometry-derived goal sizes, small initial XY variation and coordinated/leg-finish/body-finish counts14/13/13. Existing ten8/1/1 split is immutable; additions32/4/4 produce40/5/5. No core dataset schema, controller, camera, force-zero or success-algorithm changes. Success remains full footprint inside the resolved goal, not fallen, for3 policy ticks; unknown contact is diagnostic only.

## Execution and recovery
Use reproducible versioned scenario/attempt manifests. Each slot gets at most3 attempts, retaining its asset/orientation/motion/split; preserve failures. Stop short slots after three failures and report them, never lower success. Max20s simulation per attempt. Direct demonstration collection; terminal-before-reset and existing Raw/canonical checks remain mandatory per episode. User authorization supersedes old collector preflight gates for this explicit expansion only. No optimizer, remote job, automatic commit/push or new external asset download.

## Progress
- [x] Scenario/geometry/command parameterization and affected CPU checks.
- [x] Forty new successful episodes, or explicit exhausted-slot report.
- [x] Versioned combined collection/cache/train-only norm and three-view review.

## Evidence
runs/audit/v2/DATA-006/20260916T075451Z

## DATA-006 final collection acceptance

The50-episode collection is complete: original10 plus40 new usable demonstrations, split40/5/5 and24,335 actions. Raw/canonical/terminal checks passed. Actual frozen VAE/T5 encoding and cache reload passed for all50; train-only normalization uses19,340 actions. Independent final audit confirmed authoritative timestamps, pre-reset terminal capture, byte-preserved source media, unchanged26 schema/profile files,24 asset files and original controller. Pilots, failures and demo11-a1 remain local and excluded. No model training or GitHub commit/push. Evidence: workspace runs/audit/v2/DATA-006/20260916T075451Z/acceptance.json and collection-paths.json. QLM-Bench publication is separate DATA-007 work.
