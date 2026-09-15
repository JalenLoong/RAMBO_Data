---
id: DATA-001
type: change
status: completed
source_map: []
---

> Historical DATA-001 acceptance: PNG storage and direct12.5Hz acquisition. These data clauses are superseded by DATA-002; the recorded test results do not establish new LeRobot/50Hz compatibility.

# Stage 3: canonical data and synchronous runtime contracts

User-approved scope: version 2.0.0 schemas/profiles, independent validators,
CPU command preparation/scheduling/acknowledgement and protocol conformance.
50 Hz native9 commands are held for two 100 Hz RAMBO steps; physics is 500 Hz;
dual RGB is 12.5 Hz. Executed history contains completed native command intervals.
Preserve requested force; zero only the last three dimensions at the new policy boundary.
No modification of controller physics, legacy collection API, model tensors or shared weights.

## Implementation sequence
1. Freeze versioned profiles, frames, episode and message schemas.
2. Implement safe numeric-file validation, timing and release gates.
3. Implement pure command scheduler and protocol state machine with fake backend.
4. Implement WAM-side independent contract validation and action alignment.
5. Exercise golden/negative fixtures and project checks; publish evidence and archive.

## Acceptance
640 ms fixture: nine RGB frames per camera, three latent positions,
32 valid commands plus 16 initial invalid slots, 64 control and 320 physics steps.
Negative cases cover timing, image identity, unsafe arrays, provenance/checksums,
external assistance, invalid force, retries, partial execution, timeout and reset isolation.
Two repositories must agree with explicit expected fixture outcomes.
Evidence: workspace runs/audit/v2/V2-CONTRACTS/20260915T060642Z.

## Not run
Isaac/controller wiring, physical consumption timing, contact-pair sensing,
RTX synchronization, terminal capture, real VAE/server integration, collection, training, remote jobs.
No automatic commit/push. Contract acceptance does not authorize dataset release.

## Final acceptance
WAM make check: 38 tests passed; RAMBO CPU suite: 195 passed, 1 skipped, 1 CUDA deselected.
The skipped legacy-path checkpoint check was rerun with current RAMBO_CHECKPOINT_ROOT: 1 passed.
New contract tests: WAM 23 and RAMBO 47. Cross-repository 23-case conformance passed against
explicit expected outcomes. Import/doc checks and preserved-source comparisons passed.
No real runtime, model changes, data publication, environment installation or commit/push.

One development negative fixture unintentionally mixed camera paths; the new path validator
rejected it before its intended duplicate-tick check. The fixture was isolated and the suite rerun.
Source declarations explicitly record float32 setter conversion before force-zeroing; this is
not clipping/smoothing. Final specification/fixture copies have matching byte hashes.
