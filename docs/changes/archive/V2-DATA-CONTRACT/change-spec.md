---
id: V2-DATA-CONTRACT
type: change
status: completed
source_map: []
---
# LeRobot 50 Hz dataset contract revision

User-approved: revise the dataset contract, schema, minimum CPU reader/validator/mapping adapters,
synthetic compatibility tests and current documentation. Preserve the existing runtime protocol,
controller and stage-2 model code. No production recorder/converter, simulation, collection, training,
real VAE/T5 caching, remote jobs or automatic commit/push.

Decisions: 50 Hz policy RGB/action canonical rows; 25 Hz monitor-only observer; model RGB remains
12.5 Hz at fixed phase 0. Raw N actions/N+1 boundaries; canonical N rows/video frames plus separate
terminal RGB PNG/state metadata. H.264 CRF18 medium, yuv420p BT.709 limited, GOP50/scenecut off.
No new high-level filters, clamps or coordinate transforms. Actual sensor cadence is separately
recorded, not inferred as exactly 200 Hz from a nominal 5 ms setting.

Sequence: preserve baseline -> versioned schemas and source audit -> minimum adapters -> synthetic
N=16/64/65/short/tail/partial tests, LeRobot and native-reader comparison -> update current docs and
mark prior clauses superseded -> project checks and archive.

Acceptance: correct row/frame/terminal counts; immutable canonical during cache use; explicit
normalizer; false initial masks; observer exclusion; sensor freshness; failed partial video retained.
Documentation distinguishes new targets from actual 12.5 Hz simulator reference.
Evidence: workspace runs/audit/v2/V2-DATA-CONTRACT/20260915T141714Z.

## Completion
User-approved contract supplements and current-context migration completed. Both Downloads documents
were updated after checking their original hashes; baseline copies are preserved. New data tools are
CPU-verified only. No real camera/sensor/controller wiring, production conversion, collection, training
or remote work. Existing model/controller/runtime-protocol source and old specification bundles preserved.
No automatic commit/push.
