---
id: ADR-V2-DATA-CONTRACT
type: decision
status: accepted
source_map: []
---
# LeRobot 50Hz dataset, terminal boundary and model-cache separation

Accepted by the user's V2-DATA-CONTRACT implementation request. Raw/canonical policy RGB and actions are
50Hz; WAM samples phase0 at stride4 to 12.5Hz. Use LeRobot v2.1-style MP4/Parquet metadata, preserving native9.
Raw N actions/N+1 boundaries; canonical N paired rows/main video frames plus terminal RGB PNG/state metadata.
No physical terminal action or VAE padding enters canonical. Model cache alone owns grouping and residual tails.
Policy video is libx264 CRF18 medium, GOP50/scenecut0, yuv420p BT.709 limited/tv; this is high-quality lossy.
Observer25Hz remains monitor-only, default CRF23 within explicitly configured 20-23. No new observer geometry is created.

Reuse the existing execution chain and two-control-step confirmation; no new clamp/filter/coordinate retargeting.
Contact sampling is sensor-native. A nominal 5ms period is not proof of actual200Hz under2ms/lazy updates.
External force and torque are separate3D vectors with frame. Sensor torque is unavailable unless a real source exists.

Supersedes only the data-storage/acquisition clauses of ADR-V2-CONTRACTS. Preserve previous evidence and runtime
protocol versions. Current implementation is schema, CPU tooling/minimum reader adaptation and tests; all real
recorder/production conversion/Isaac/VAE/T5/training gates remain deferred.
