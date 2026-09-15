---
id: ADR-V2-BOUNDARIES
type: decision
status: accepted
source_map: []
---
# v2 source boundaries

Accepted by the user's stage 0–1 implementation request, 2026-09-15.
WAM is model-only; RAMBO_Data owns simulator and canonical data. Keep the existing governance organization.
The new simulator branch deletes project biped and obsolete three-camera surfaces; recover historical implementations from original Git branches.
Preserve other quadruped tasks, controller physics, 9D commands and the independent 18D residual interface.
Preserve the existing Go2 + D435i nominal two-camera geometry. Production dataset, timing and inference contracts follow in later work.
Local GPU sandbox escape is authorized; remote full SFT follows local development. No model conversion, new task, collection or training in bootstrap.
