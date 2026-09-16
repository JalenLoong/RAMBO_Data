---
id: CURRENT-OVERVIEW
type: current
status: accepted
source_map: []
---
# Adaptation v2 overview

## Research objective and current position

Adapt pretrained LingBot-VA to language-conditioned Go2 quadrupedal manipulation with one FL manipulation leg and native RAMBO9D commands. The research target is a correctly trained checkpoint making stable synchronous closed-loop progress on the approved Push Box task using dual RGB/visual history and confirmed executed-action history.

Model adaptation foundations and the real simulation-to-data-to-cache path are validated. DATA-005's ten demonstrations were user accepted; DATA-006 now has50 usable demonstrations, complete Canonical/cache acceptance and no missing slots. No SFT checkpoint or learned-policy task-success result exists yet. QLM-Bench publication is DATA-007; its current status and fixed revision are in [publication policy](publication-v2.md).

## Fixed baseline

- Go2 quadruped, FL only; original RAMBO controller/checkpoint and native robot reset XY origin.
- Native9 order: base vx/vy/yaw rate, FL position xyz, desired FL force xyz. Desired force remains zero.
- Inputs: text, Go2 ego RGB, D435i task RGB, visual history and confirmed executed9D history. Telemetry/proprio/contact/observer remain diagnostic, not extra model inputs.
- Frozen VAE/T5, native9 projections and ego-above-task dual-view latent composition. First SFT baseline uses existing Transformer trainability policy; no OPD, LoRA-first redesign, proprio encoder or replacement controller.
- Raw/canonical RGB/actions50Hz; modelRGB12.5Hz; observer25Hz monitor-only; controller100Hz/physics500Hz. Factor4 selects rows; timestamps inherit authoritative canonical simulation_time_ns or terminal metadata, never frame-index synthesis.
- CanonicalN action rows/video frames plus separate pre-reset terminal state/two RGB PNGs; RawN+1 boundaries. No dummy terminal action. Preserve source tails; apply explicit initial masks/complete grouping only in derived model preprocessing.

## Responsibilities

WAM-Policy owns model conversion, data consumption, frozen preprocessing/cache, normalization, SFT, checkpoint recovery, inference and model evaluation. RAMBO_Data owns Isaac/controller/task/assets/cameras, scripted demonstrations, recording, task success and canonical publication. Exchange explicit contracts; no cross-repository implementation imports.

## Completed evidence and limits

| Work | Completed | Not established |
|---|---|---|
| INFRA-001/002 | Local workspace, source isolation, runtime foundations | Remote AMD runtime |
| ALG-001 | Native9 conversion, dual-view VAE/T5, full-model T=9 forward/export/reload; small-model backward/single-process recovery | Full-model backward/optimizer, FSDP/distributed resume |
| DATA-001/002/003 | Runtime/data contracts, LeRobot50Hz storage, authoritative timestamps and terminal semantics | Learned-policy runtime integration |
| DATA-004 | Approved task/assets/cameras, original controller wiring, strict terminal-before-reset and valid straight full-footprint pilot | Old center-only pilot is invalid/excluded |
| DATA-005 | Three pilots followed by ten accepted small-XY demonstrations,8/1/1 split and actual frozen caches | Learned task success |
| DATA-006 | Fifty usable episodes,40/5/5 split, real frozen cache reload, train-only normalization and final data acceptance | Dataset sufficiency/generalization or SFT performance |

DATA-004 was previously committed/pushed. Later source publication is recorded by DATA-007's GitHub publication evidence; source HEADs alone do not identify uncommitted historical collection implementations. Each original run retains its actual source hashes.

## Current task and data

Success: full XY envelope of all eight pose-transformed source-box corners inside that episode's goal, robot not fallen, for three consecutive50Hz policy ticks. Roll/pitch are included in geometry. Contact is diagnostic/unknown, not a gate; residual yaw and natural box toppling are reported without inventing upright/contact/yaw success thresholds.

DATA-006 retains DATA-005's ten original files and8/1/1 split, and adds40 episodes with32/4/4 split. Total50 episodes and24,335 actions; train-only affine q01/q99 normalization uses19,340 actions, no clipping, constant dimension scale1, force offsets0/scales1. Validation/test do not affect fitting. VAE state resets per episode; frozen VAE/T5 encoding and cache reload passed on all50.

Added variations: Box05/01/04 counts20/10/10, yaw0/90/180/270 ten each, goal colors/geometry and coordinated/leg-finish/body-finish configuration counts14/13/13. Of13 leg-finish configurations,12 actually triggered the conditional finish phase. demo13/25 successful a4s include stop-forward/left-shift/inward-FL correction. demo11-a2 is valid; a1 is quarantined. Pilots, failures and quarantined attempts remain local and are excluded from usable Canonical and HF publication.

Evidence: workspace runs/audit/v2/DATA-006/20260916T075451Z/acceptance.json, collection-paths.json and collection-entries.json. Original per-episode Raw/canonical sources and source media are immutable. Final audit confirms all pre-reset terminals and inherited timestamps;26 core schema/profile and24 asset files plus original controller remain unchanged. No further collection is required to meet this50-episode scope.

## Next phase: discuss before implementation

The user intends to prepare training code, submission scripts/job orchestration for their AMD server and feasible local RTX5070Ti small-scale verification. The next session must first restore context and confirm understanding, not implement or launch jobs.

Existing flow/trainability/checkpoint utilities are not a complete production training launcher. Resolve real data-window/history/mask/tail handling, sampling and resumable data state, optimizer/scheduler/validation/logging, complete checkpoints and remote execution configuration. Preserve approved model/data semantics. Full-model backward/optimizer/checkpoint-resume on the target backend requires its own canary evidence before formal SFT; small-model success cannot substitute.

The AMD server connection, GPU count/type, scheduler, ROCm/PyTorch/attention backend and runtime/storage are unverified. Do not infer CUDA compatibility or copy prior OPD behavior. Local outside-sandbox GPU work is generally user-authorized, but the new session's first turn is read-only context recovery. No remote jobs, training or new collection have been started.

## Read next

- [Contracts](contracts-v2.md)
- [Publication](publication-v2.md)
- [DATA-005](../work/archive/DATA-005/plan.md)
- [DATA-006](../work/archive/DATA-006/plan.md)
- [Normalization/split decision](../decisions/ADR-0009.md)

The Downloads handoff is historical research context; current contracts and measured evidence take precedence.

- [Authoritative dataset contract](adaptation_v2_dataset_contract.md)

## Published first release

QLM-Bench publication and fixed-revision readback passed. Final Hub revision `4b5e0ed9aa04cbc1143774798f63171b27d838a7`; payload revision `21b69e2d775cd77d6565e25d1b0b99ad725b54b3`. Exactly1616 payload files/2150407479 bytes,50 usable Raw/Canonical demonstrations, split40/5/5. All remote file hashes match;516 Canonical/release files were downloaded at the payload revision and independently validated through RAMBO and WAM/LeRobot0.3.3. Root README/index and exact remote membership were verified at the final revision; public/license unknown unchanged. No pilots/failures/quarantine/model-cache/asset binaries uploaded. Publication and source handoff receipts: workspace runs/audit/v2/DATA-007. No training or remote jobs.
