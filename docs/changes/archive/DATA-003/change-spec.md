---
id: DATA-003
type: change
status: completed
source_map: []
---
# DATA-003: Authoritative sampling time and first-task approval boundary

## Purpose and approved scope
User requested targeted corrections: Approach-and-Push Box is a proposed first-task candidate / pending user approval; assets, camera coverage, geometry and success/contact criteria freeze only after approval. Factor-4 sampling selects rows, never constructs timestamps. If Push Box is selected, require separate FL-object/body-object contact observability. Separate RGB/VAE, text/T5 and executed-history/action-embedder paths in the diagram. Preserve architecture, timing validation, schema hashes, model interfaces and runtime semantics.

## Implementation and recovery
Actual fixed-factor implementation is CanonicalReader.alignment, not a class named FixedFactorResampler. Read canonical int64 timestamps and terminal metadata directly, including action tail provenance. Reversible code/docs edits only; no commit/push, new collection, task freeze or runtime changes. Existing immutable profiles and historical reports remain unchanged.

## Validation threshold
CPU tests must distinguish timestamp inheritance from grid regeneration using large int64 sentinel times at the reader seam; actual malformed grid data must still fail validation. Exercise selected terminal, unselected terminal/tail, short episode and existing model/cache regressions. Check both repositories' docs, import boundaries and shared registry.

## Progress and evidence
- [x] Inspected current reader, contract, namespace and tests; registered DATA-003.
- [x] Implement corrections and tests.
- [x] Validate, update accepted current docs and archive paired work.
Evidence: `runs/audit/v2/DATA-003/20260916T022525Z/`. Real contact observability, asset/task approval, runtime, collection, VAE/T5 generation and training are not_run.

## Outcome
WAM make check: 73 passed, including int64 source/terminal inheritance, tail and short-episode probes, retained off-grid validation, existing reader/cache/model tests and context guards. RAMBO context/governance: 16 passed; docs/import/cross-registry checks passed. First WAM attempt failed four new test cases because their fixtures assumed list-valued timestamps; actual canonical time is scalar int64. Test construction was corrected; production validation/format was unchanged. Original schema/profile bytes are verified against Git HEAD. Downloads originals and workspace context synchronized. No commit/push or next-stage execution.
