# RAMBO_Data v2 agent guide

Read docs/current/overview.md and .agent/PLANS.md. Work only on the v2 branch for this adaptation.
RAMBO_Data owns Isaac/RAMBO runtime, quadruped tasks, assets, camera geometry, teleoperation, recording, telemetry, success detection and canonical dataset publication.
Never import WAM, LingBot-VA, Diffusers or Transformers. WAM-Policy consumes explicit dataset/runtime contracts and owns model-specific caches, training and inference.
Current scope: quadruped Go2, single FL manipulation, native 9D; Go2 ego + D435i RGB task camera. First-baseline desired force is zero. Retired biped and three-world-view project code/documents are deleted from v2, with history retained in Git.
Preserve existing quadruped controller/checkpoint physics and other quadruped tasks unless a task-specific change is authorized. Editor cameras are debug-only.
Keep datasets, checkpoints and run outputs in versioned workspace layers outside source. Preserve historical resources and manifests.
Local Ubuntu 24.04 / RTX 5070 Ti handles debugging, simulation and data synthesis. The user authorizes outside-sandbox execution for necessary local GPU/Isaac checks; sandbox GPU invisibility is not a host-driver failure.
Full model SFT belongs to the remote AMD server after local debugging and job-script preparation; connection and remote runtime are not yet verified. Do not start remote work in bootstrap.
Use workspace.env for paths and workspace.lock.yaml for installed environments. The old /etc/vast-agents-guide.md is unavailable and is not a dependency.
Use Work IDs, ChangeSpec/ExecPlan and evidence; report passed/failed/not_run accurately. Run project boundary checks and affected tests before handoff.

Stage 3 (V2-CONTRACTS) is authorized: versioned contracts, independent installed validators, CPU command/protocol logic and conformance tests. Keep real simulator/server integration and collection/training not_run. See docs/current/contracts-v2.md. No automatic commit/push.
