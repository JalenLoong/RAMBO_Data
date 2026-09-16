---
id: DOC-002
type: change
status: completed
source_map: []
---
# DOC-002: Current status and historical AMD evidence

## Intent and observable behavior
User authorized correcting stale Adaptation v2 status documentation and investigating the supplied historical AMD branch and W&B run through W&B MCP. Synchronize current data/publication/source status and distinguish historical AMD execution evidence from unverified v2 training compatibility.

## Compatibility and non-goals
Documentation only: no model/training/config-schema changes, no training, collection, AMD login, job submission, commit or push. Preserve historical runs, archived work and source/model identities. The Downloads research note receives a dated status correction with its original preserved in the workspace evidence layer.

## Evidence threshold and acceptance scenarios
Local HEADs match DATA-007 receipts; current docs agree on 50 episodes/40-5-5/24,335 actions/19,340 normalization rows and final publication. Historical AMD claims cite bounded W&B MCP results, run files and pinned Git source. Mark final-evaluation metadata as such; do not infer FSDP or current v2 compatibility from OPD. Docs/governance/import checks and required existing WAM checks pass; runtime remains not_run.

## Migration and rollback
No semantic migration. Revert only DOC-002 documentation changes if needed. Keep original evidence and new readback observations separate.
