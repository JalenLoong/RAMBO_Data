# Adaptation v2 agent guide

Read docs/INDEX.md, docs/current/overview.md, docs/current/contracts-v2.md, docs/current/publication-v2.md and .agent/PLANS.md. Follow docs/governance/documentation.md and docs/governance/v2.md. Shared v2 Work IDs are reserved in WAM's docs/governance/work-registry.json and synchronized to RAMBO; v1 numbering is independent. Archive paired ChangeSpec/ExecPlan only after evidence passes. Keep attempt/source/checkpoint identities and historical manifests immutable.

DATA-002 remains the current LeRobot50Hz storage/terminal contract; later DATA-005/006/007 acceptance does not change its schema/profile identities.

Fixed baseline: Go2 quadruped, FL manipulation only, native9, original RAMBO controller, ego + task RGB, desired force zero. Raw/canonical50Hz, modelRGB12.5Hz, observer25Hz diagnostic-only. Inherit authoritative simulation timestamps and pre-reset terminal state/RGB; do not synthesize time from frame indices. Do not add proprio/observer/contact as model inputs or redesign the architecture.

Use workspace.env and workspace.lock.yaml in the workspace container. Necessary local RTX5070Ti GPU/Isaac execution outside the sandbox is user-authorized; sandbox GPU invisibility is not a host-driver diagnosis. Full SFT is intended for the user's AMD server; historical MI355X/Slurm/ROCm OPD execution is documented by DOC-002, while current v2 access/runtime/backend compatibility remain unverified. Read docs/current/overview.md for the evidence link. No OPD, remote job, new collection or training is implied by prior data authorization.

Current data: DATA-005 ten accepted demonstrations; DATA-006 completed50 usable demonstrations,40/5/5 episode split,24,335 actions, actual frozen VAE/T5 caches and train-only normalization on19,340 actions. Read docs/work/archive/DATA-006/plan.md. Pilots, failures and demo11-a1 quarantine remain local/excluded; demo11-a2 is the valid replacement. Success remains full footprint in episode goal and not fallen for3 policy ticks; contact unknown is diagnostic, not a gate.

DATA-007 publishes usable demonstration Raw/Canonical and corresponding evidence/replay to dontKnow23456/QLM-Bench. Never upload pilots/failures/quarantine/model caches/weights/asset source. Public/license unknown unchanged. User explicitly authorized the DATA-005/006/007 source commits/pushes to existing v2 targets, now complete per publication receipts. DOC-005 records the user's explicit authorization and completed source publication of DOC-002/003/004 and TRAIN-001 to the existing three v2 GitHub targets; final readback receipts are in workspace runs/audit/v2/DOC-005. This is not permission for arbitrary later commits or jobs.

TRAIN-001 implementation is explicitly authorized: WAM training/DDP recovery, W&B and administrator Slurm package, plus local GPU diagnostics capped at900 seconds. DOC-002 changes must be preserved. Do not connect to AMD, submit remote jobs or collect data. The user-authorized source commit/push is completed under DOC-005; this does not authorize unrelated future publication. The administrator executes single/eight-card smoke and automatic8-card1000-step SFT; these target-hardware results remain unverified. Read docs/current/overview.md and the TRAIN-001 work record.

RAMBO owns simulator/task/assets/cameras/control/recording/canonical publication. Never import WAM/LingBot-VA/Diffusers/Transformers. Preserve original controller/checkpoint and unrelated quadruped tasks. Run docs, import boundaries and affected tests before handoff.

DATA-007 HF release is published/verified at 4b5e0ed9aa04cbc1143774798f63171b27d838a7. Its completed plan is docs/work/archive/DATA-007/plan.md. Source push receipts and next-session prompt are in workspace runs/audit/v2/DATA-007.
