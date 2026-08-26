# RAMBO simulator boundary

Read `/etc/vast-agents-guide.md` before working with this repository. RAMBO owns
simulator-native tasks, policy execution, PhysX configuration, and the collection
backend interfaces under `rambo.collection`. It must never import WAM packages.

Do not write datasets, checkpoints, or training output into this repository. WAM
owns collection orchestration and all serializable LingBot v1 data artifacts. Use
workspace paths from `workspace.env`; released checkpoints live under
`/workspace/checkpoints/rambo`, and run evidence lives under `/workspace/runs`.

Keep Gym task IDs, checkpoint contracts, task physics, and success conditions stable
unless an approved task-specific change explicitly changes them.
