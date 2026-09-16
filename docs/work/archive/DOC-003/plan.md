---
id: DOC-003
type: exec-plan
status: completed
source_map: []
---
# DOC-003: GitHub README administrator handoff

## Scope and authorization
User requests the branch main README as the administrator handoff: GitHub/Hugging Face links and privately supplied tokens, clone the approved branch, download inputs and submit jobs. Remove the previously offered source archives. Explicitly document total/video/action losses; training already emits all three, validation receives the corresponding total. No commit/push, Hub upload, training or remote submission.

## Implementation
Use the historical administrator README structure, read from the preserved branch ref9e3447495c8376183dda92c368c9f95d7df40f6a after the supplied remote URL returned404. Rewrite WAM README, retain recovery details in a subordinate document, remove source packaging helper/fallback and delivered archives, and mark old packaging receipts historical. Do not invent HF locations for unpublished pinned model/cache assets; require owner-supplied private resource repo/revision before handoff. Preserve data, models and original execution evidence.

## Validation
Docs/governance/import/shell checks and WAM existing CPU tests, including separate loss consistency and Gloo checks. No GPU tests are required for README/logging arithmetic changes. Record archive deletion hashes and current publication prerequisites in workspace runs/audit/v2/DOC-003.

## Progress
- [x] Read current implementation and historical README reference.
- [x] Update main README and handoff routing; remove source package delivery.
- [x] Verify loss fields and check changes; archive records.

## Completion
Main WAM README now contains GitHub branch checkout, site environment, verified QLM-Bench revision, explicitly pending private HF resource release, W&B three-component loss keys, site config and automatic submission instructions. The secondary administrator doc is recovery-only. Both source tar archives, the packaging helper and no-Git source-receipt fallback were removed; old runtime/packaging receipts remain historical. Training and validation each emit total/video/action losses. WAM98 tests, RAMBO16 affected tests and9 README shell-block syntax checks passed; embedded Python parsed; docs/governance/diff checks passed. No GPU run, remote connection/submission, upload, commit or push.
