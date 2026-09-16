---
id: DATA-006
type: exec-plan
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

## Discovery and recovery

The first slot exhausted3 attempts on a numpy.float64 command / float32 quaternion dtype mismatch. Corrected the velocity tensor dtype explicitly; failures remain immutable and no fourth attempt was added. Resolved goal centers now follow per-episode goal geometry, with a packager consistency guard. CPU geometry/recorder checks12 passed; collection continues over remaining slots.

## Straight-push reference correction

Later attempts use an oriented-box center-ray intersection instead of dividing by a permanently selected face normal when the object turns. Failed270-degree attempts also show the measured FL EEF persistently on one side of the center ray. A bounded high-level EEF-Y tracking compensation (gain2/s, limit1.5cm, reset per episode) is applied to bring the actual foot closer to that ray. This stays within the authorized small position-command adjustments; no controller, desired-force, schema or success change. Source versions are bound in each attempt; earlier data/failures are not rewritten. Physical effect is reported from the already scheduled attempts, not assumed.

## Source-lineage quarantine

During the early dtype correction, demo_11-a1 bootstrapped across a source edit: its declared expert hash is the pre-fix version, while the executed successful path requires the fixed dtype. Preserve but exclude it from the combined dataset. Reacquire demo_11 using its remaining a2/a3 budget after the main sweep. Added a post-import hash consistency guard to reject startup source drift. Do not edit watched code while captures are starting.

## Explicit retry exception

The user approved one additional attempt for demo_10 only after its3 dtype-error attempts. Queue a4 after the main sweep, with the corrected frozen implementation. All other slot limits remain3. Evidence: extra-demo10-approval.json.

## Orchestrator interruption

The original long-lived collector parent ended with143/SIGTERM; termination source is unknown. demo_31-a1 has a sealed20s task-failure result, but its subprocess exit status was not recorded. Preserve the unfinished attempt plus interruptions/demo_31-a1-capture.json; resume at its remaining attempts. Subsequent orchestration handles at most4 unfinished slots per invocation. Capturing code remains frozen.

Operational continuation, runner/repair commands and active resources are recorded in workspace evidence `runs/audit/v2/DATA-006/20260916T075451Z/CONTINUE.md`; progress.json and quarantine.json are authoritative for usable counts.

## Latest user steering: review before recapture

Finish the original main sweep, then show every exhausted slot and its failed attempts. Hold all recapture, including previously approved demo_10-a4 and lineage replacement, until the user resumes after review. See post-main-review-hold.json. Do not run prepare_repair or any extra-attempt runner now.

## DATA-006 main sweep — user review hold

The main sweep is complete: 36 usable additions plus the original10 yield46 usable demonstrations, not50. Three slots exhausted their original3-attempt budgets (demo_10, demo_13, demo_25); all9 attempts are available in workspace `runs/audit/v2/DATA-006/20260916T075451Z/failures.html`. demo_11-a1 is separately quarantined for startup source-fingerprint inconsistency. The user requested failure review before any recapture; this defers previously approved demo_10-a4 and the lineage replacement as well. No recapture, combined collection/cache, training or publication has been performed after this hold. DATA-006 remains active and the50-episode target is incomplete.

## User-authorized corner recovery trial

User reviewed failures and authorized demo10 recapture plus a trial for13/25: stop forward motion near full containment, shift body left, use FL to nudge the offending side inward. One a4 attempt per named slot is now authorized; demo11 remains on hold. Core contract, success, controller, camera and source assets remain unchanged. Explicit episode flag enables the new high-level recovery only for13/25. Freeze source before launching. Evidence: runs/audit/v2/DATA-006/20260916T075451Z/corner-trial-20260916T110631Z .

## Corner-repair trial completed

The user-authorized a4 trials for demo10/13/25 all passed. demo13 and demo25 actually executed stop-forward, body-left shift and inward-FL-nudge phases; success occurred at13.96/14.42s under unchanged full-footprint/not-fallen/3-tick criteria. Desired force remains zero. Nominal left shift16cm produced actual16.6/19.8cm, disclosed as tracking overshoot. This only validates these two near-goal +Y-side cases. Source was frozen and checked against each capture. Evidence: workspace `runs/audit/v2/DATA-006/20260916T075451Z/corner-trial-20260916T110631Z`. There are now49 usable demonstrations; demo11 remains quarantined/deferred. No further collection, merged cache, training or publication.

## demo11 recapture resumed

User explicitly requested demo11 recapture. Run its original remaining a2/a3 deterministic parameters, without corner recovery; preserve quarantined a1. No other collection or training. Evidence: demo11-recapture-authorization.json in the DATA-006 audit directory.

## demo11 replacement completed

User-authorized demo_11-a2 passed in9.14s/457 actions, with full-footprint/not-fallen/3-tick success, terminal-before-reset, Raw/canonical validation and consistent startup/final source hashes. Original demo_11-a1 remains preserved and quarantined. Usable episodes now total50 (original10 plus new40; planned aggregate split40/5/5). No further collection is needed. Combined DATA-006 collection, frozen model caches and final collection acceptance remain pending; no training, commit or push. Evidence: workspace runs/audit/v2/DATA-006/20260916T075451Z/demo11-recapture-result.json.

## DATA-006 final collection acceptance

The50-episode collection is complete: original10 plus40 new usable demonstrations, split40/5/5 and24,335 actions. Raw/canonical/terminal checks passed. Actual frozen VAE/T5 encoding and cache reload passed for all50; train-only normalization uses19,340 actions. Independent final audit confirmed authoritative timestamps, pre-reset terminal capture, byte-preserved source media, unchanged26 schema/profile files,24 asset files and original controller. Pilots, failures and demo11-a1 remain local and excluded. No model training or GitHub commit/push. Evidence: workspace runs/audit/v2/DATA-006/20260916T075451Z/acceptance.json and collection-paths.json. QLM-Bench publication is separate DATA-007 work.
