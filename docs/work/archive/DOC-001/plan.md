---
id: DOC-001
type: exec-plan
status: completed
source_map: []
---
# DOC-001: Restore v2 documentation governance

## Intent and authorized scope
Restore the complete historical governance protocol; use an independent shared v2 category numbering pool. Migrate current Work IDs and ADRs, reword only v2-exclusive commits, commit accepted DATA-001/DATA-002 work and publish to the three explicitly configured targets with exact leases. User approved this plan on 2026-09-16 (Australia/Sydney).

## Invariants and recovery
Preserve v1/main refs, common ancestors, source/model/controller semantics, versioned schema/profile bytes and historical run payloads. Save backups, original files and old/new SHA mappings. Abort a remote update if its lease changed; record partial publication without automatic rollback.

## Progress
- [x] Saved source snapshots, patches, refs and bundles; created isolated checkouts.
- [x] Reworded existing v2 commits with identical trees and author identities.
- [x] Split DATA-001 and DATA-002 using actual historical backups and CPU checks.
- [x] Restore full governance, registry, templates, routing and tests.
- [x] Validate final snapshots, update local branches and publish with exact leases.

## Evidence threshold
Governance positive/negative tests, paired-work lifecycle, registry and source hashes, reference checks, staged snapshot tests, WAM make check, RAMBO CPU and both cross-repository conformance suites must pass. Each actual attempt gets its own record.

## Evidence and limitations
Workspace evidence: `runs/audit/v2/DOC-001/20260915T152709Z/`. Real collection, simulator/server wiring, VAE/T5 regeneration, training and W&B runs are not_run. This work does not retroactively validate model performance or data publication.

## Discoveries
Original documentation was recovered from immutable Git content. Isolated tests need workspace tool/source bindings; the first RAMBO dataset attempt lacked them, and the first WAM dataset attempt omitted the fixture variable. Failed/skipped attempts are retained and rerun explicitly.

## Local acceptance
WAM make check: 68 passed. RAMBO CPU: 211 passed, 1 PyArrow-dependent module skipped, 1 CUDA test deselected. Offline dataset suite: 37 passed. Cross-repository contract cases: 23/23 and dataset cases: 10/10. Both wheels build offline, and packaged spec bytes match the source snapshot. Real collection/runtime/training/W&B: not_run. All three configured v2 targets were published and read back successfully. Publication rounds and final closure SHA are tracked in the external publication report.

The source-protection audit first used an incorrect fixed count for packaged spec files; it now compares the complete expected filename/hash inventory instead. No schema or implementation change was needed.

## Outcome
The complete governance source is restored byte-for-byte in both repositories, v2 IDs and decisions are migrated, CPU validation passed, and accepted DATA-001/DATA-002 code was committed and published. The completion metadata is included in the DOC-001 closure amendment; its exact lease publication is verified in the external report. v1/main/common history and immutable execution identities remain unchanged. No production data was generated or released.
