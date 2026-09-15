---
id: HISTORICAL-CONTRACTS-V2-0
type: reference
status: historical
source_map:
  - source/rambo/rambo/contracts_v2/validation.py
  - source/rambo/rambo/contracts_v2/runtime.py
  - source/rambo/rambo/contracts_v2/spec/schemas.json
  - source/rambo/rambo/contracts_v2/spec/telemetry.json
  - scripts/rambo/check_contracts_v2_conformance.py
---

> Historical PNG/12.5Hz acquisition contract 2.0.0. Dataset storage/timing is superseded by DATA-002; this is not the current canonical specification. Runtime/action-confirmation behavior remains separately applicable.

# Canonical Dataset v2 and synchronous policy contract

Version **2.0.0**. Action identity remains **rambo-native9-v2**. RAMBO_Data owns the
canonical schema/profile and synthetic conformance fixtures. WAM carries a pinned
copy and a separately installed validator; neither repository imports the other.
Only named `extensions` permit additional fields. Unknown core fields, versions and
profiles fail closed. Frozen JSON bytes and semantic profile digests are in the bundle lock.

## Entry points and boundaries

- `validate_episode(directory, release=False)` reads and checks files; it never publishes.
- `validate_release(manifest_path)` additionally checks each episode's release eligibility.
- `prepare_command(requested, command_id, start_tick)` is pure and does not mutate the input.
- `CommandLedger(actions, chunk_id, start_tick)` schedules two control steps per command;
  only backend confirmations advance its clock.
- `SynchronousSession(cache_resetters=...)` orchestrates pure messages and a supplied backend.
- CLI: `python -m rambo.contracts_v2 EPISODE_DIR [--release-eligible]` or
  `python -m rambo.contracts_v2 RELEASE_JSON --release-manifest`.
  Exit 0 means the requested validation passed; exit 1 returns a structured error code/detail.
- Existing collection dataclasses, generic setters and all task/controller loops are unchanged.
  Real controller/server/cache wiring is not part of this implementation.

The pure contracts extra requires NumPy and Pillow; use the installed interpreter in the
workspace lock. No Isaac or model weights are imported by this package.

## 1. Canonical episode and provenance

An episode directory contains `episode.json`, `checksums.json`, JSONL RGB/command/control/
intervention records, NPZ robot/contact/task snapshots, PNG images and referenced task/evidence JSON.
Image paths are `rgb/<semantic_camera_id>/<frame_name>.png`. All references are relative to the
episode; absolute paths, traversal, symlinks, duplicate JSON keys and nonfinite JSON fail.
`checksums.json` maps every payload file, including `episode.json`, to its byte SHA-256 and excludes
only itself. An outer release manifest stores both the episode JSON and checksums.json byte hashes; no self-referential digest.

The executable `schemas.json` catalog lists every required field in `$defs`; external JSON Schema tools must select a record with a reference such as `#/$defs/episode`, rather than validate against the catalog root. Episode metadata identifies dataset,
episode, reset epoch, seed, task/version, exact UTF-8 instruction/language, simulation time origin,
warm-up end, last tick, profile hashes, file references, outcome and provenance. Provenance requires
RAMBO commit, checkpoint hash, Isaac Sim/Lab identity, asset source/license/hash, task config,
asset approval reference (nullable before release) and runtime evidence references.

Task config declares the object body, named FL/object and base/object contact pairs, measurement
methods, progress unit/description and success definition. Its version/hash freezes task-specific
metrics; the actual Approach-and-Push asset, thresholds and implementation follow asset review.
`proximity_inference` cannot be used to mark a confirmed pair-contact measurement valid.

NPZ files contain only numeric/bool NPY arrays, loaded with `allow_pickle=False`. Exact names,
shapes, dtypes, units and frames are in `telemetry.json`; structured/object arrays are forbidden.
Headers are checked against archive payload sizes before allocation. A numeric group is limited
to 256 MiB uncompressed per episode in this baseline; larger production episodes need a versioned
storage-profile change. Per-field `<name>_valid` masks are mandatory; false values require a
nonempty reason in the group's metadata. Invalid payload values remain finite placeholders.
Snapshots occur at tick 0, each completed 100 Hz control boundary and the final terminal tick.
Contact snapshots separately record their actual sensor `source_tick`; no invented 200 Hz cadence.

Raw aborted/partial episodes remain contract-valid when their provenance and timing are complete.
Schema validity does not establish release eligibility. Ordinary validation reports release as
`not_run`/null. Baseline release checks require a successful non-truncated non-diagnostic episode,
all required telemetry valid (friction and diagnostic QP force/source tick may be unavailable), no external assistance, runtime
conformance and asset approval. Runtime reports must match the profiles/task-config/checkpoint
subject hash and pass control consumption, force boundary, contact pairs, camera capture sync,
terminal-before-reset and asset dual-view coverage. These files reference actual evidence;
constructing a JSON assertion is not a substitute for running those gates or for human asset review.

## 2. Native action semantics and execution boundary

Order: `base_vx, base_vy, base_yaw_rate, fl_ee_x, fl_ee_y, fl_ee_z, fl_ee_fx, fl_ee_fy, fl_ee_fz`.
Units: m/s, m/s, rad/s, m, m, m, N, N, N. One command has nine scalars.
The independent 18D RAMBO residual remains separate telemetry.

Keep the existing gravity-projected controller frame: `R_body_from_projected` is constructed by
`rp_rotation_from_gravity_b(g_body)`. Its inverse is the transpose. At level attitude the axes are
forward/left/up; retain the matrix definition under roll/pitch instead of substituting a yaw-only
approximation. Position origin uses base world x/y projected onto world z=0, with current
`com_offset=(0,0,0)`; it is not a measured whole-body center of mass. FL target is the existing
analytical kinematic endpoint and an absolute position in that frame. `fl_ee_z` is not height
relative to the robot base. Keep measured body pose and actual contact points distinct.

Command rows contain requested, transformed, filtered and executed values plus IDs/intervals/status.
`requested` preserves finite incoming physical JSON numbers. `transformed` records conversion to
the existing setter's float32 dtype followed by exact force zeroing. The first six dimensions have
no clipping, geometric retargeting or smoothing; only explicit float32 representation rounding.
`filtered` is identical (`filter_id=identity`). Record whether requested force was overridden.
Only a fully confirmed 20 ms interval gets `executed` and `status=completed`; partial execution
retains `executed_until_tick` and control trace but `executed=null`. This is a high-level hold
confirmation, not a claim that an EEF target was achieved.

A command has two consecutive 10 ms holds. Record residual-observation and FL-IK source command
IDs plus IK-computation tick. Existing code reads base/force pre-step, updates FL IK post-step,
and its runner uses the previous returned residual observation. Preserve this order; do not
assemble asynchronous downstream components into a new 9D history or shift labels to hide delays.

## 3. Force, contact and external state interventions

The existing QP consumes the negative desired FL force as a reaction target. The current physical
loop also rotates negative desired force into FL body-local axes and writes a permanent wrench.
Without an explicit position that wrench acts at FL body CoM; its direct torque is zero.
This path is documented, not changed. The new policy boundary keeps baseline force zero before
any future setter/wrench consumption.

Record desired force, QP force, sensor force and external injected wrench independently. QP force is a body-frame estimate at its own qp_source_tick; missing diagnostic QP values carry false masks/reasons.
PhysX net contact output is normal force; friction is a separate optional measurement. Neither
per-component maxima over force history nor proximity-plus-net-force are instantaneous pair forces.
The partner-filtered sensor API requires a single sensing body per environment; actual new-task
FL/object and base/object channels still require runtime validation.

Intervention records identify source, target body, coordinate frame, application point, force N,
torque N*m and half-open active physics interval. Direct velocity/pose manipulation is a separate
`state_intervention`, even if all wrench components are zero. Release rejects any such assistance.

## 4. Physical time, RGB, latent and action alignment

Episode-local signed-int64 physics ticks are authoritative. Timestamp ns equals
`origin_sim_ns + capture_tick * 2_000_000`. Reset epoch scopes all ticks and identities.
Intervals are left-closed/right-open.

| Unit | Physics ticks | Duration | Native commands |
| --- | ---: | ---: | ---: |
| PhysX step | 1 | 2 ms | — |
| RAMBO step | 5 | 10 ms | — |
| Native command | 10 | 20 ms | 1 |
| RGB sampling interval | 40 | 80 ms | 4 |
| Subsequent latent interval | 160 | 320 ms | 16 |
| Regular chunk of two latents | 320 | 640 ms | 32 |

Initial actual RGB at tick 0 yields L0 with 16 zero/false-mask action slots. L1 adds RGB at ticks
40/80/120/160 and corresponds to actions on [0,160); L2 adds 200/240/280/320 and actions on [160,320).
Causal VAE state persists; groups are not independent clips. Two views occupy spatial height,
not a doubled time axis. A model chunk of C latents has 16C action slots. In the first C=2 chunk,
L0/L1 use five RGB images per camera and only 16 valid actions. Subsequent complete C=2 chunks
add eight RGB per camera and 32 actions. No production inference chunk size is fixed here.

Canonical storage preserves all frames and partial actions. WAM alignment returns the maximal
complete `N=1+4k` prefix, exact command/source indices and explicitly listed unencoded tail.
No interpolation, duplicated frames, silent truncation, index-only temporal inference or
cross-episode cache reuse. Capture the terminal state before auto-reset; warm-up and the first
real image are separately identified. Contract tests do not establish actual renderer timing.

## 5. Cameras and hashes

Semantic mapping: existing `ego` -> `go2_ego`, `task` -> `d435i_rgb_task`; source setup remains
`robot-dual-v3`. Both streams are RGB uint8 1280x720 at 12.5 Hz. Mount convention is +X forward,
+Y left,+Z up; optical convention is +X right,+Y down,+Z forward. The declared
`R_mount_from_optical` maps optical vectors into mount axes. Quaternions use XYZW.

Ego nominal base-relative position is (0.32715,-0.00003,0.04297), identity mount rotation;
its 120 degree diagonal FOV is an explicit assumption. D435i mount is (0.30,0,0.34), pitched down
65 degrees; H-FOV 69.4 degrees, simulated V-FOV about 42.561 degrees (published nominal 42.5).
Profiles retain exact source parameters, nominal intrinsics, clipping, source URLs and unknowns.
The visual bracket has no modeled mass/collision. These are not device calibrations.

Individual semantic configuration hashes and a paired-camera hash are distinct from WAM's
preprocessing hash. JSON canonicalization uses sorted keys, compact separators, UTF-8 and no NaN.
Measured per-frame intrinsics/pose are separate data; K tolerance is 1e-3 pixels and moving-mount
pose/rotation tolerance is 1e-5 against valid base telemetry. Synchronized simulation requires
identical capture ticks; real 15 FPS capture/resampling requires a separate future profile.

## 6. Synchronous runtime protocol

All messages carry version, session/episode/reset epoch, message ID, profile hashes and extensions.

| Message | Body |
| --- | --- |
| ResetRequest | mandatory finite positive timeout_s, task/instruction, origin_sim_ns |
| Ready | tick=0 and cache generation |
| Observation | tick, actual dual RGB increment, full cumulative confirmed command history, instruction |
| ActionChunk | observation_id, chunk_id, start_tick, action_period_ticks=10, physical actions [K,9] |
| ActionAccepted | chunk ID/start and command count; no execution claim |
| ExecutionAck | chunk/start/end, completed count, full command statuses and reason |
| End | last tick, success/terminated/truncated and reason |
| Error | tick, stable code and reason |

K must be a positive multiple of 16. Exactly one observation/chunk is outstanding. Waiting for
inference never advances simulation time. A caller-supplied monotonic wall clock controls the
mandatory timeout; polling the deadline enters Error with no fallback command. The driver must
poll while waiting. Identical message-ID retries return the stored result, without another step;
conflicting contents, stale epoch/observation, reused chunk IDs and skipped/overlapping ticks fail.
A full acknowledgement permits the next actual RGB increment; partial stop terminates the session.
Call `finish(reason=...)` before End when a chunk is partially executed.

The backend receives one schedule per control step and must return actual confirmation with IDs,
held values, residual action and consumption metadata. Exceptions do not manufacture confirmations.
A failure after physics advanced but before confirmation needs recovery evidence in a future driver;
this state machine keeps only confirmed time. No automatic controller reset or retry is attempted.

Reset advances epoch within a session, empties pending commands/history and invokes supplied cache
reset callbacks. Real adapters must clear both causal VAE streams, pending RGB and KV state; CPU
fixtures exercise callbacks only. WAM PolicyHistory independently checks requested-vs-acknowledged
commands and accepts only completed intervals. Predicted video and unexecuted actions cannot be
substituted for actual Observation history. Proprio/contact/task telemetry is not model input.

## Acceptance boundary

640 ms synthetic and negative cases test schemas, numeric/time consistency, force processing,
idempotence, partial execution, cache callbacks and two-interpreter agreement. They are not
runtime/contact/camera calibration evidence, demonstrations, SFT data or task success.
Real Isaac timing, controller wiring, force enforcement, contact pairs, terminal capture, RTX
synchronization, VAE/server integration, collection, training and remote work remain not_run.

Accepted CPU contract evidence: workspace `runs/audit/v2/V2-CONTRACTS/20260915T060642Z`. Real runtime integration remains not_run.
