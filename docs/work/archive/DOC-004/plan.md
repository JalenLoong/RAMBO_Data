---
id: DOC-004
type: exec-plan
status: completed
source_map: []
---
# DOC-004: Final checkpoint HF publication instructions

## Scope
User requests a final handoff-README section with CLI to upload the completed checkpoint to dontKnow23456/WAM-Policy main under a new independent folder. Documentation only in this session; no upload, training, AMD connection, job submission or Git publication.

## Behavior
Select completed formal step1000 from its actual attempt, validate the full resumable checkpoint, authenticate privately, use checkpoints/adaptation-v2/<logical-run>/<attempt>/step_00001000 on main, and record the upload commit SHA. Read back the manifest at that immutable SHA. Existing payloads, visibility and dataset publication stay unchanged; no Raw/model caches are uploaded by these commands.

## Validation
Read hf-cli skill and installed hf upload/download help. Check README shell and embedded Python syntax, simulate local selection and collision guards without network/uploads, then docs/governance checks. The supplied HF web URL could not be fetched by web tooling; destination is user-specified, not newly claimed as remotely verified.

## Progress
- [x] Inspect checkpoint format and CLI syntax.
- [x] Add final publication section and route recovery reference.
- [x] Verify and archive documentation records.

## Completion
Main README now ends with authenticated hf upload to the user-designated main branch and a run/attempt-specific directory. Full formal step1000 selection, checkpoint verification, remote collision refusal, upload receipt and fixed-revision manifest readback CLI are documented. Secondary recovery README routes to the authoritative commands. Installed CLI flags checked;11 shell blocks and embedded Python parse;6 selection/collision cases passed with mocked Hub access and zero network/uploads. WAM make check98 passed; docs/governance/diff checks passed. No upload, training, remote connection/submission or commit/push performed.
