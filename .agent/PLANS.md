# Work-plan protocol

Use one Work ID for every multi-file change. Before implementation, create a ChangeSpec
under `docs/changes/active/<WORK-ID>/` and an ExecPlan under
`docs/work/active/<WORK-ID>/`. Record assumptions, affected contracts, validation, and
explicitly not-run items. On completion, move both documents to the corresponding
`archive/` tree with `status: completed`; do not leave duplicate active copies.

Semantic changes to action, camera, timing, data labels, model interfaces, or evaluation
also require an ADR and an update to the relevant `docs/current/` page only after the
evidence threshold stated in the ChangeSpec is met.

Run manifests are immutable records of actual attempts. Start from
`docs/templates/run-manifest.yaml`; do not create a manifest for a planned or unrun job.

Governance: read docs/governance/documentation.md and docs/governance/v2.md before multi-file work. Reserve CATEGORY-NNN in docs/governance/work-registry.json; v2 numbering starts at 001 independently of v1, shared across both repositories. Use the same Work ID in changes, plans, commit/PR titles and attempts. See docs/work/INDEX.md.
