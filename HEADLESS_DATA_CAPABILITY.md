# RAMBO Isaac Sim Headless Data Capability Audit

Audit time: 2026-08-25 UTC  
Scope: existing source, existing recorders, existing Isaac Sim 6.0.1 / PhysX runtime, and one fresh deterministic Button probe. No simulator, policy, controller, or source changes were made.

## 1. Executive Summary

The production headless path is a real PhysX + RTX synthesis pipeline, not a schema-only claim. The fresh Button run uses the released quadruped checkpoint, the 405-D policy observation and 18-D native policy output, one public contact sensor, the physical spring button, and the production front RGB camera. It streams aligned policy actions, post-step raw observations, post-step robot/button state, and RGB PNG frames.

The immediately usable data product is therefore a **scripted-command, state/action/video trajectory** for the Button task. It is suitable as the foundation for a LingBot-VA adaptation dataset, but it is not yet language-conditioned: the current `prompt` is a fixed recorder-provenance string, not a task instruction paired with a semantically varying trajectory. Depth and calibrated camera data have been separately exercised in the current runtime but are not written by the production episode recorder. Full contact tensors, desired actuator targets/torques, high-level task commands for generic locomotion, and QP internals are runtime-accessible but not dataset fields today.

Status categories used below:

| Status | Meaning |
|---|---|
| A | Current Button recorder saves it. |
| B | Current source/runtime can expose it; current recorder does not save it. |
| C | Reliably derivable from saved fields/configuration. |
| D | Not demonstrated by the current pipeline, or requires a different sensor/configuration. |

## 2. Current Production Host

| Item | Observed value |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Compute capability | 8.9 |
| NVIDIA driver | 595.71.05 (R595) |
| Vulkan | NVIDIA proprietary Vulkan device visible (`apiVersion 1.4.329`) |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128; `torch.version.cuda=12.8` |
| CUDA reality check | `torch.cuda.is_available()=True`; CUDA tensor reduction followed by synchronize succeeded |
| Isaac Sim | 6.0.1.0 |
| Isaac Lab source | `v3.0.0-beta2.patch1`, `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| qpth | 0.0.18 |
| RAMBO branch / HEAD | `adapt/isaacsim60-isaaclab30b2` / `70879f06869793f59cfdb55ab91c4b76dde3ee7c` (clean) |
| Native validated commit/tag | `5d7745994358280b001b78ed6b5484248f630c8c`, `rambo-isaac60-native-r1` |

The fresh short Button gate completed headlessly with the public `PhysxManager`, released checkpoint strict load, QP execution, finite 405-D observation/18-D action, RTX RGB, and clean Kit exit.

**PRODUCTION SYNTHESIS HOST: READY**

## 3. Existing Recorder Schema

The existing recorder is `scripts/rambo/record_button_physx_episode.py`. It is a deterministic single-environment Button episode recorder, not a general dataset writer. For each policy action step it writes four streams:

| File | Current payload | Status |
|---|---|---|
| `actions.jsonl` | `action_step`, `[t,t+10 ms]`, `policy_action` `(1,18)` float JSON numbers, scripted `command` | A |
| `observations.jsonl` | post-step `action_step`, `t+10 ms`, raw environment `policy_observation` `(1,405)`, reward | A |
| `post_states.jsonl` | post-step root pose, FL-foot world position, 12 joint positions and velocities, contact force scalar summary, button telemetry, terminal | A |
| `rgb_frames.jsonl` + `rgb/*.png` | fresh 640x480 RGB image identity, action-step/time/ticks/frame counter/path/mean/std | A |
| `manifest.json`, `summary.json`, `checksums.sha256` | task/provenance/schema/timing/checkpoint/PhysX/acceptance/memory evidence | A |

It does **not** store a general natural-language instruction, task ID separate from task name, initial-state vector, randomization parameters, full contact tensors, desired motor targets, applied torques, or QP matrices.

## 4. Data Capability Matrix

| Field | Source | Shape / dtype | Units / frame | Rate | Status / saved today | LingBot recommendation |
|---|---|---|---|---:|---|---|
| Task identifier | manifest | string | N/A | episode | A | Keep |
| Fixed recorder prompt | manifest/summary | string | provenance instruction, not language task label | episode | A | Do not treat as task prompt |
| Scripted high-level command | Button schedule | 3+3+3 float vectors | base velocity m/s + rad/s convention; FL target/force in projected COM frame | 100 Hz | A | Keep |
| Native policy action | PPO output | `(1,18)`, float32 before JSON | first 6 base acceleration residuals, last 12 joint-position residuals; projected COM/controller frame | 100 Hz | A | Keep |
| Raw policy observation | environment output | `(1,405)`, float32 before JSON | mixed task state; five-frame history | 100 Hz | A | Keep for debugging/teacher alignment |
| Root link position/quaternion | public articulation data | `(1,3)`, `(1,4)` | m, world; quaternion XYZW | 100 Hz | A | Keep |
| Joint position/velocity | public articulation data | `(1,12)` each | rad, rad/s; RAMBO ordered joints | 100 Hz | A | Keep |
| FL foot link position | public articulation data | `(1,3)` | m, world | 100 Hz | A | Keep |
| Contact force summary | ContactSensor history | `(1,)` max norm from `(1,3,19,3)` | N, world-vector norm | 100 Hz | A (summary only) | Keep summary; add per-foot force only if needed |
| Button state | physical RigidObject + task logic | `(1,)` each | displacement m; boolean success/released/ready | 100 Hz | A | Keep |
| Terminal | Gym post-step `dones` | `(1,)` | boolean/int | 100 Hz | A | Keep |
| RGB | RTX front camera | `(1,480,640,3)`, uint8 | RGB 0–255 | 12.5 Hz | A | Core modality |
| Depth | same front camera with tested output type | `(1,480,640,1)`, float32 | `distance_to_image_plane`, m | 12.5 Hz | B | Optional, add only after throughput test |
| Camera intrinsics/pose | public camera data | `(1,3,3)`, `(1,3)`, `(1,4)` | pixels; m/world; XYZW | camera rate or static-per-episode | B | Add once per episode + pose per frame |
| Desired joint position/torque, applied torque | RAMBO/Articulation public buffers | `(1,12)` each | rad, N m | 100 Hz | B | Useful teacher/debug fields; not required V1 |
| Foot contact mask / per-foot force | ContactSensor public data | `(1,4)`, `(1,4,3)` derived from history | bool, N world | 100 Hz | B | Add contact mask; force optional |
| COM / body poses / velocities | articulation public data / properties | body-batched tensors | m/world, m/s, rad/s | 100 Hz | B | Debug/robotics supervision only |
| QP inputs/solution | controller/QP call | matrices/vectors detailed in §8 | projected COM/body frame | 100 Hz | B for sampled debug capture | Do not put full matrices in V1 |
| Segmentation / normals / optical flow | no current recorder/smoke evidence | N/A | N/A | N/A | D | Do not promise; needs explicit sensor configuration and validation |

## 5. Visual Modalities

The production front camera configuration is a base-mounted RTX `PinholeCamera`: 640x480, 18 mm focal length, horizontal aperture 20.955, clip range 0.1–20 m, offset `(0.30, 0, 0.08)`, world convention, 80 ms exact cadence. Production `data_types=["rgb"]`.

| Modality | Real runtime evidence | Shape / dtype / value convention | Current recorder | Alignment |
|---|---|---|---|---|
| RGB | fresh 30 s Button frames and prior RGB/RGBD smoke | `(1,480,640,3)`, `torch.uint8`, RGB 0–255; recorder rejects black/low-variance images | A, PNG | exact actions `8,16,…,3000`; `t=k*10 ms` |
| Depth | prior current-runtime PhysX RGBD smoke | `(1,480,640,1)`, `torch.float32`, `distance_to_image_plane` in m; finite/positive geometry verified | B | same 80 ms camera cadence if enabled |
| RGBD | same camera can emit RGB+depth together | as above | B | same frame/tick counter |
| Intrinsics | prior public camera smoke | `(1,3,3)`; observed fx=fy=549.74945, cx=320, cy=240 | B | static config; pose may be read with each frame |
| Extrinsics / pose | public `camera.data.pos_w`, `quat_w_world` | `(1,3)` m world, `(1,4)` XYZW | B | frame-aligned on camera read |
| Segmentation | no recorder or smoke evidence | not audited as working | D | N/A |
| Normals | no recorder or smoke evidence | not audited as working | D | N/A |
| Optical flow / motion vectors | no recorder or smoke evidence | not audited as working | D | N/A |

The prior depth smoke is strong evidence that depth is directly available from the already existing front camera after adding its output type, but it is not evidence that a 30 s RGBD dataset has yet been produced. A third-person RGBD diagnostic camera exists only for migration validation; it is not the production video stream.

## 6. RAMBO Action Hierarchy

```text
semantic task / future text prompt              [not a real trajectory field today]
        ↓
Button scripted high-level command              [A: 3-D base velocity, 3-D FL position, 3-D FL force]
        ↓
raw 405-D policy observation (5 x 81)           [A, post-step; PPO normalizes internally]
        ↓
18-D native policy output                       [A, pre-step]
  [0:6] base linear/angular acceleration residual, controller projected-COM frame
  [6:18] joint-position residual, rad
        ↓ scale, clip and joint-position target
QP desired acceleration + desired EE force       [B]
        ↓ QP solved 12-D GRF, desired joint torque [B]
        ↓ position target + effort target to articulation
PhysX five 2-ms ticks                             [actual execution]
```

Do not conflate the 9-D Button high-level command with the 18-D native policy action. The recorder supplies the Button command before each policy call; the action is produced from the observation by the strict-loaded checkpoint. In the quadruped configuration, `action_scale` supplies six base and twelve joint scales; actions are clipped to ±two scales. The controller outputs 12 desired motor positions and 12 desired torques, and Physics applies the commands during the five physics substeps.

## 7. Robot State and Contact / Manipulation State

### Robot state

| Signal | Available / shape | Frame / units | Saved now |
|---|---|---|---|
| Root link pose | `(1,3)` + `(1,4)` | world m + XYZW | A |
| Root linear/angular velocity | `(1,3)` each public articulation tensor | world m/s, rad/s; body-frame values also computed in env | B |
| Joint position/velocity | `(1,12)` each | rad, rad/s | A |
| Desired joint position/velocity/torque | `(1,12)` each | rad, rad/s, N m | B |
| Applied joint torque | `(1,12)` | N m | B |
| Foot position/velocity | body-batched public tensors; FL position singled out | world m / m/s | FL position A; rest B |
| Body pose / COM | public body pose/mass tensors and environment COM property | world m, XYZW | B |

### Contact/manipulation

`ContactSensor` is a real PhysX sensor configured on `/World/envs/env_.*/Robot/.*`, history length 3, 5 ms update. Its full net-force history is real sensor state—not a gait-command estimate. The current recorder stores only its per-step global maximum norm and the history shape `(1,3,19,3)`.

| Signal | Meaning | Saved now |
|---|---|---|
| Foot contact mask | `norm(ContactSensor net forces)>0`, four feet | B |
| Per-foot contact force | max over sensor history, world vector | B |
| Gait phase/mode | controller command/estimate (`desired_contact_phase/mode`), not measured contact | C from 405-D observation; B separately |
| Button displacement | physical cap world-X relative to rest, clamped `[0,0.02]` m | A |
| Button pressed/success | task-side logic: >=12 mm for five policy steps | A (success) / C (pressed from displacement) |
| Button released | task-side threshold: displacement <=2 mm | A |
| End-effector/button contact pair force | not instrumented as a named contact pair | D; existing robot ContactSensor does not establish it |

## 8. QP / Controller Internals

The code constructs and solves a 12-variable ground-reaction-force QP every policy step. It exposes/retains the solved `grf` `(N,12)`, desired acceleration `(N,6)`, solved acceleration `(N,6)`, and QP cost. `P`/quadratic term is `(N,12,12)`, `q`/linear term `(N,12)`, `G` `(N,18,12)`, `h` `(N,18)`; equality `A,b` are intentionally empty in this qpth formulation. The mass/centroidal map is `(N,6,12)`. Jacobians used for torque conversion are `(N,12,12)`.

One canonical QP snapshot is already recorded and validated. It contains input/output snapshots and residual metrics; qpth exposes the primal solution but RAMBO does not retain dual variables. The normal Button recorder intentionally does not persist QP tensors.

Recommendation: a sampled, opt-in QP trace is useful for regression/debugging, solver audit, and occasional teacher analysis. It is **not** a good default LingBot training field: full matrices add substantial 100 Hz JSON/binary bandwidth, encode solver implementation detail, and do not directly improve a language-conditioned video-action learner. Retain action, high-level command, robot state, success and (optionally) solved torque/GRF instead.

## 9. Policy Observation

The recorder saves the raw environment observation, before the PPO empirical normalizer. The strict-loaded PPO policy then applies its checkpoint `EmpiricalNormalization` internally before inference. The checkpoint normalizer count is `100003840`. Thus the saved 405-D values are interpretable physical/controller quantities; the normalized network input is reproducible only with the checkpoint normalizer.

### Quadruped / Button: 405-D = 5 x 81-D

Every temporal slice has these exact offsets:

| Slice offsets | Block | Dimension |
|---|---|---:|
| `[0:1]` | base height | 1 |
| `[1:4]` | projected gravity, body frame | 3 |
| `[4:7]` | base linear velocity, body frame | 3 |
| `[7:10]` | base angular velocity, body frame | 3 |
| `[10:22]` | joint position minus default | 12 |
| `[22:34]` | joint velocity | 12 |
| `[34:38]` | desired contact phase | 4 |
| `[38:42]` | desired contact mode | 4 |
| `[42:54]` | desired joint position minus default | 12 |
| `[54:57]` | base velocity command | 3 |
| `[57:60]` | FL EE position command | 3 |
| `[60:63]` | FL EE force command | 3 |
| `[63:81]` | prior native policy action | 18 |

The flattened policy observation is five chronological 81-D slices: slice `j` is `[81*j:81*(j+1)]`, with the newest slice `[324:405]`. On reset, history is initialized to zero; after action step `t`, the recorder writes the post-step history whose newest slice is the state/commands/action at `t`.

### Biped: 435-D = 5 x 87-D

The first 57 dimensions are the same through base velocity command. Then: `[57:60]` FL EE position, `[60:63]` FR EE position, `[63:66]` FL EE force, `[66:69]` FR EE force, `[69:87]` prior action. The newest biped slice is `[348:435]`.

## 10. Timing & Synchronization

| Clock | Actual contract |
|---|---|
| Physics | 2 ms, 500 Hz |
| Policy/action/state/observation | decimation 5 → 10 ms, 100 Hz |
| RGB | 80 ms, 12.5 Hz, exactly 40 physics ticks = 8 policy steps |
| Full episode | 3000 actions = 30.0 s; expected 375 RGB frames |

For policy step `t` (1-indexed): the policy receives the previously returned raw observation `obs_(t-1)`, emits `action_t`, then `env.step(action_t)` executes five Physics ticks. The recorder writes `observations[t]` and `post_states[t]` at `t*10 ms`. Therefore both are **post-action** `state_t`/`obs_t`, not the pre-action policy input. The action record explicitly covers `[ (t-1)*10 ms, t*10 ms ]`.

RGB frame `k` is captured only when the public camera frame counter advances; it maps to action step `8*k`, physics ticks `40*k`, timestamp `80*k ms`. The exact integer Warp cadence rejects missed/advanced counters (`last+1` required) and asserts the expected physics tick, so the recorder is designed to fail rather than silently create a dropped-frame sequence. The initial post-reset image is baseline only, not frame 1. There is no current off-by-one ambiguity in the stored mapping; consumers must nevertheless join RGB to the matching **post-step** state/action end time.

## 11. Headless Task Matrix

| Task | Task ID | Fully headless | Command source | Keyboard needed | RGB path | Action/state | Success label | Randomizable today |
|---|---|---|---|---|---|---|---|---|
| Quadruped locomotion | `Isaac-RAMBO-Quadruped-Go2-v0` | yes (policy smoke) | policy + configured/sampleable commands | no | validated RGB/RGBD smoke; normal task camera opt-in | yes | safety/terminal, no semantic task success | config supports command/randomization flags; recorder disables them |
| Biped locomotion | `Isaac-RAMBO-Biped-Go2-v0` | yes (policy and RGBD smoke) | policy + configured/sampleable commands | no | validated RGBD smoke | yes | safety/terminal, no semantic task success | config supports flags; no dataset recorder |
| Button / Press Button | `Isaac-RAMBO-Quadruped-Button-Go2-v0` | yes | deterministic scripted 9-D loco-manip schedule | no | production 12.5 Hz RGB recorder | yes | physical press/hold/rebound | deterministic current recorder; no episodic randomization saved |
| GUI teleop Button | same Button task | not fully automatic by design | native keyboard callbacks | yes | GUI evidence path | yes | physical Button state | not an automatic synthesis path |

Only the scripted Button recorder is presently an end-to-end, completely automatic **successful-trajectory** generator. It uses no keyboard and accepts success only after physical displacement/hold/rebound evidence. Locomotion rollouts are automatic but currently prove stability rather than a task-specific success predicate.

## 12. Current Data Gaps

### Must add before a language-conditioned dataset

- A real `task_instruction`/prompt field and an instruction schema/version. The existing fixed M10 prompt is recorder provenance, not a semantic command.
- Episode metadata: task ID/name, seed, initial state and goal/target specification, command-program ID, terminal reason and success definition.
- Explicit stream alignment metadata that makes the post-action semantics and `RGB k → action 8k` mapping machine-readable.
- Store the high-level command and 18-D native action in a stable binary/columnar format rather than only high-volume JSONL for scale.

### Recommended additions

- Camera intrinsics once per episode, camera pose per frame, and optionally the already tested depth output.
- Full root velocities, per-foot contact mask, and desired/applied actuator targets if downstream control diagnostics need them.
- Task randomization parameters whenever randomization is enabled; otherwise mark the episode deterministic.

### Do not default to saving

- Full 12x12/18x12 QP matrices and solver transient tensors every step.
- Unvalidated segmentation/normals/optical flow.
- Third-person diagnostic outputs as if they were a stable production observation stream.

## 13. Minimal LingBot Adaptation Dataset V1

Design only; no implementation in this audit:

1. **Episode metadata:** episode UUID, task ID, canonical instruction text, seed, environment/config hash, initial state/goal, success, terminal reason. This makes language-to-trajectory supervision auditable.
2. **RGB:** 640x480 front-camera frames plus camera intrinsics/extrinsics and the explicit 12.5 Hz index mapping. This is LingBot-VA's visual input.
3. **High-level RAMBO command:** base velocity plus EE position/force command at 100 Hz. This distinguishes task intent from low-level residual control.
4. **18-D native RAMBO action:** policy action at 100 Hz, with declared scaling/version/frame semantics. This is the most direct action supervision target.
5. **Robot state and raw policy observation:** root/joint/foot state plus 405-D raw observation at 100 Hz. These support state-conditioned analyses, re-simulation, and distillation; the normalizer checkpoint must accompany observations if exact policy-space reconstruction is needed.
6. **Timestamps and outcome:** action intervals, post-state timestamp, RGB timestamps/frame IDs, success/terminal. These prevent a video/action off-by-one error.

The V1 should begin with deterministic scripted Button data plus a small, explicitly labeled set of automatic locomotion command programs. Add domain randomization only after emitting its per-episode parameters and preserving the same timing/schema contract.

## Fresh Probe Evidence

Fresh artifact: `/workspace/headless-data-audit/20260825T105300Z-button-30s-probe/`.

| Check | Measured result |
|---|---|
| Recorder result | `passed=true`; public PhysX before/after evidence recorded |
| Strict checkpoint | released quadruped checkpoint accepted; 405-D / 18-D contract |
| Action records | 3,000, contiguous steps 1–3,000, finite |
| Observation records | 3,000, contiguous steps 1–3,000, finite; `(1,405)` |
| Post-state records | 3,000, contiguous steps 1–3,000, finite; 12 joint positions + 12 velocities |
| RGB records/PNGs | 375, action steps 8–3,000; `(1,480,640,3)` `torch.uint8` |
| RGB health | first PNG `(480,640,3)` `uint8`, min/max 35/226, mean 139.34, std 16.64; non-black; 375 PNGs present |
| Contact telemetry | public history shape `(1,3,19,3)` used each post-step |
| Outcome | success at action 765; rebound at 795; terminal count 0 |
| Physical evidence | max button travel 19.994 mm, 17 consecutive pressed action steps; base / FL-foot movement 0.588 / 0.832 m |
| Timestamp boundary | final action/observation/post-state time = 30,000,000,000 ns; final RGB maps to step 3,000 |
| Memory gate | GPU allocation slope 0 MiB/100 steps; RSS slope 0.0455 MiB/100 steps; both within limits |
| Offline verification | `RAMBO_BUTTON_PHYSX_EPISODE_VALID`; 381 checksummed files |

This full episode is a fresh runtime probe, not a bulk dataset generation run.
