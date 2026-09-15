---
id: V2-CLEANUP
type: change
status: completed
source_map: []
---
# V2-CLEANUP

## Authorized scope
Remove retired biped and three-view project surfaces, enforce ownership, and validate the retained quadruped runtime.
User authorization: implement v2 stages 0–1, 2026-09-15.

## Plan and acceptance
Record branch/resource identities; use the two v2 branches; preserve original refs and external historical resources.
WAM owns model-side work; RAMBO_Data owns simulator and canonical data. Keep the 9D quadruped controller unchanged.
Check packaging, imports, documentation, retained tests, 64-step PhysX and bounded dual-camera RTX.
No model conversion, dataset collection, training, remote connection or push in this work.

## Execution and evidence
Completed on 2026-09-15 within the authorized stage 0–1 boundary.

- Both v2 branches created from local main; original main/v1 refs retained.
- Simulator source renamed to RAMBO_Data; current workspace routing and editable sources updated.
- Policy packaging now exposes only the bootstrap namespace; simulator WAM installation removed.
- All distribution versions preserved, except the explicitly removed simulator WAM distribution.
- Minimal WAM governance follows the existing Work-ID/ChangeSpec/ExecPlan/ADR layout.
- Simulator project biped/obsolete camera surfaces and their dedicated tests/documents removed.
- Retained quadruped controller/task files and final dual-camera numerical configuration match the original main exactly.
- Bounded camera diagnostic moved from the old WAM implementation to RAMBO_Data with collector/program dependencies removed.

Evidence root: `runs/audit/v2/V2-BOOTSTRAP/20260915T024558Z` in the workspace.
See its README.md and final-consistency.json for the complete inventory and acceptance results.

## Validation
- passed: WAM compilation, import boundaries, documentation checks and boundary tests.
- passed: retained simulator CPU tests and outside-sandbox CUDA QP test.
- passed: single-environment seed-42 quadruped checkpoint load and 64 real PhysX steps, exit 0.
- passed: 240-step RTX preview, 30 frames per camera, 1280x720, correct intrinsics and mount transforms, exit 0.
- passed: source bindings, original branch refs and protected resource inventory checks (final consistency report).
- not_run: GUI keyboard interaction, new task/asset acceptance, Dataset v2, model conversion, collection, training, remote jobs and push.

## Review notes
An initial CPU run failed four old-camera/test-fixture assertions; they were updated and passed on rerun.
The directory-switch Git LFS post-checkout hook initially lacked git-lfs on PATH; the branch was created, and LFS status was checked with the workspace tool path. Existing asset bytes were retained.
Camera rendering acceptance is not production task-coverage or per-device calibration acceptance.
Changes are delivered in the two v2 worktrees without commits or remote pushes.
