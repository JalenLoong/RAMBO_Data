---
id: TRAIN-001
type: exec-plan
status: completed
source_map: []
---
# TRAIN-001: Whole-episode SFT, DDP recovery and administrator AMD pipeline

## Authorization and scope
User explicitly approved implementation of the complete plan: fixed QLM-Bench revision4b5e0ed9aa04cbc1143774798f63171b27d838a7, existing40/5/5/cache/normalizer, full Transformer trainability, single/8-rank smoke then automatic8-rank1000-step SFT, W&B junlinlong-the-university-of-sydney/WAM-Policy. Codex implements and validates locally; administrator executes AMD. No AMD login/submission, commit/push, collection, OPD/LoRA/FSDP. Preserve DOC-002 changes and immutable resources. Local GPU budget cumulative900 seconds.

## Implementation and compatibility
WAM owns inspect/train/validate/export, deterministic whole-episode sampler, FP32 master/BF16 compute, AdamW1e-5/betas0.9,0.95/weight_decay0.1/clip1/warmup10, mask-preserving joint flow, strict complete per-rank recovery, isolated validation and W&B attempt lineage. Existing stage2 config and resume APIs remain compatible. RAMBO participates only in governance/routing.

Administrator jobs: CPU preflight30min; single-card smoke30min;8-card smoke60min; formal8-card1000 optimizer steps6h. Default8-card allocation64 CPUs/512GiB. Two smoke gates compare10 continuous versus5+resume5, test longest episode, bind source/data/config/environment/resources, require<85% HBM and1.5x time estimate within6h plus storage admission. Formal starts fresh, never from smoke weights. W&B online required, pending sync distinguished from training failure; no model/cache payload uploads.

## Progress
- [x] Restore scope and preserve pre-existing changes.
- [x] Data/config and deterministic sampler.
- [x] Single/DDP engine, recovery and validation.
- [x] W&B journals, metadata artifacts and bounded synchronization.
- [x] Slurm orchestration, gates and administrator handoff.
- [x] CPU/Gloo/mock tests and bounded local GPU checks.
- [x] Final docs/evidence and archive after applicable checks pass.

## Validation and evidence
Workspace runs/audit/v2/TRAIN-001. CPU deterministic resume must be exact; GPU rtol1e-3/atol1e-5. Test split isolation, gradients, accumulation, rank failure, RNG, corrupt/wrong-world checkpoints, gate rejection, automatic1000-step dependency chain and W&B retry identity. Local actual cache tiny backward/resume/longest plus fullT9 no-grad, at most900s. AMD/runtime/SFT/task success remain not_run until administrator evidence exists.

## Recovery and decision log
New derivatives only. Checkpoints atomic and step-boundary only. No automatic budget expansion or unmasked fallback. Current data and baseline architecture remain unchanged. Details and measured deviations will be recorded here.

## Final implementation evidence —2026-09-17

Implemented explicit whole-episode CLI, rank-disjoint sampler, FP32/BF16 single/DDP updates, complete hashed per-rank checkpoints, validation RNG isolation, W&B attempt/metadata/sync handling, four-stage held Slurm chain, strict gates, portable uncommitted source packaging and administrator runbook. Existing stage-2 flow/policy/checkpoint APIs, dataset implementation/profile and configs/adaptation-v2.json are unchanged. RAMBO implementation is unchanged.

WAM final make check:98 passed, including CPU exact single/Gloo resume, accumulation/nonfinite gates, Slurm submit/rollback/continuation, W&B pending-sync identity and artifact exclusions. RAMBO affected context/governance:16 passed. Both repositories' docs/governance/import boundaries and shell syntax passed. First Gloo invocation was blocked by sandbox loopback permissions; the approved outside-sandbox CPU-only rerun passed with GPUs hidden. Historical evidence was not rewritten.

Actual local GPU smoke01:115.841378509s cumulative of900s. Real-cache full-model T9 forward passed with peak allocated10,830,834,688bytes; diagnostic model real-cache updates, intermediate save/resume and longestT46 pressure passed. Resume data/RNG/LR exact, optimizer compared, maximum parameter error1.4901161193847656e-8 withinrtol1e-3/atol1e-5. All eight parameter groups showed sampled parameter changes; four upstream unused text tensors disclosed. Local W&B remained offline. Final changes after GPU verification were reporting/admission/packaging; final CPU checks cover these without relabeling the original GPU fingerprint.

Formal deployment defaults:8 MI355X,1000 optimizer steps,6h; W&B junlinlong-the-university-of-sydney/WAM-Policy online; automatic preflight->single smoke->eight-card smoke->fresh SFT. Historical AMD package versions are reference facts, not current backend acceptance. AMD connection, submission, full-model optimizer, formal SFT and task success remain not_run for this implementation session. No commit/push. Source delivery and all receipts live in workspace runs/audit/v2/TRAIN-001.

Final shutdown test additionally delivered a real local SIGTERM during a diagnostic update, saved at the safe boundary, and resumed to CPU-exact equality with uninterrupted execution. The stage wrapper now records graceful SIGTERM failure during preflight as well. Final CPU log: workspace runs/audit/v2/TRAIN-001/wam-final-check-with-signals.log.
