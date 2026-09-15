---
id: GOVERNANCE-DOCUMENTATION
type: governance
status: accepted
source_map: []
---
# Evidence and document lifecycle

AGENTS.md routes work; docs/current states accepted interfaces; ADRs record decisions; Git records implementation.
Use one Work ID per multi-file change with paired ChangeSpec and ExecPlan under docs/changes and docs/work.
Archive completed pairs together. Keep research separate from accepted facts. Use source maps and valid relative document links.
Record actual attempts with immutable run evidence; report passed, failed and not_run separately.
Keep provenance for code, config, model, dataset and runtime. Never create a successful-run record for an unexecuted experiment.
