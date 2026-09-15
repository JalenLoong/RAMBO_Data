# Isaac Sim 6 container reference

The current acceptance target is the installed native workspace runtime; see [scope](current/overview.md) and [runtime commands](../README.md).
Dockerfile.rambo60 is retained for future deployment work. No container build or runtime acceptance is claimed by v2 bootstrap.
It requires a verified Ubuntu 24.04 base-image digest and compatible host GPU support.
Keep external checkpoints read-only and outputs in separate mounts. Preserve the pinned Python 3.12 / Isaac 6.0.1 / Torch 2.10 cu128 / PhysX configuration.
Use the quadruped checkpoint and explicit visualizer selection from the native runtime instructions when a container is validated later.
