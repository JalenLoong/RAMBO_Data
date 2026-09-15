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
