# RAMBO → Isaac Sim 5.1 / Isaac Lab 2.3.2 Migration

## Objective

This migration delivers a reliable **dual-mode playback loop** on RTX 50-series hardware. Quadruped and biped are equal first-class targets: each keeps its own task registration, QP topology, controller semantics, checkpoint, observation schema, and acceptance result.

The first milestone is complete only when both tasks load their original checkpoints, run a deterministic 30-second rollout, and produce clean GUI and offscreen RGB on the following runtime:

| Component | Pinned version |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchvision | 2.7.0 / 0.22.0, CUDA 12.8 |
| Isaac Sim | 5.1.0 (via official Isaac Lab pip package) |
| Isaac Lab | 2.3.2 |
| qpth | 0.0.18 |
| NVIDIA driver | Linux production branch >= 580.65.06 |

The validation host is Ubuntu 22.04.5 with an RTX 5080 and driver 580.159.03.

## Scope and non-goals

In scope:

- `Isaac-RAMBO-Quadruped-Go2-v0`: 405-dimensional observation and 18-dimensional action.
- `Isaac-RAMBO-Biped-Go2-v0`: 435-dimensional observation and 18-dimensional action.
- Existing pretrained checkpoint playback, QP/actuator compatibility, a native front RGB camera for both modes, and GUI/offscreen rendering validation.
- A standalone RAMBO external extension installed over the official Isaac Lab pip distribution.

Out of scope for this milestone:

- WAM/LingBot-VA integration, 15-dimensional high-level actions, three-camera contracts, synchronized trajectory datasets, and data synthesis.
- Training, logger/video restoration, checkpoint resume, and policy retraining.
- Button, Manipulator, teleoperation, and the legacy trajectory recorder. They are preserved but not ported.
- Legacy 4.5-vs-5.1 numerical parity. The old Torch 2.5.1 CUDA 12.4 stack cannot execute qpth on RTX 5080 (`sm_120`), so it is historical evidence rather than a test gate.

## Baseline and Git preservation

The original vendored Isaac Lab extension reports version `0.36.6`; its closest upstream source baseline is commit `09590912792d...`, not Isaac Lab 2.0.2. RAMBO also contains custom patches over that baseline.

The dirty Isaac Sim 4.5 work has been frozen separately:

```text
branch: codex/legacy-isaacsim-4.5-local
tag:    legacy-isaacsim-4.5-local
```

The migration proceeds on `new_IsaacSim_IsaacLab`. Checkpoints, datasets, logs, and validation images stay outside Git. Do not modify the legacy Python 3.10 environment. After both modes passed the acceptance gates, vendored `source/isaaclab*` was removed from the adaptation branch; use `legacy-isaacsim-4.5-local` for the recoverable 4.5 source snapshot.

## Runtime and package layout

Create an isolated Python 3.11 environment at `/workspace/venvs/rambo51` with [scripts/setup_isaacsim51.sh](scripts/setup_isaacsim51.sh). The installer pins `setuptools<81` and `flatdict==4.0.1` before installing the official bundle:

```bash
python -m pip install 'isaaclab[isaacsim,all]==2.3.2' \
  --extra-index-url https://pypi.nvidia.com
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

`requirements/isaacsim51.in` is the human-maintained intent file and
`requirements/isaacsim51.lock` captures the current validation-host package
set with NVIDIA and PyTorch extra-index headers retained. After a fresh
environment passes `pip check`, qpth GPU checks, and an Isaac Sim smoke test,
regenerate the lock with `python -m pip freeze --all --exclude-editable`.
The official Isaac Lab wheel bundles, but does not automatically install, its
`isaaclab_assets` extension. The setup script installs that exact bundled
official source after the wheel is present; it never installs this repository's
vendored `source/isaaclab_assets`. RAMBO owns its small Gym configuration
resolver, so it does not need the unrelated `isaaclab_tasks` example-task
package. Install `source/crl2` and `source/rambo` separately in editable mode;
they must not be captured as machine-specific editable entries in the lock.

The RAMBO package follows the Isaac Lab 2.3.2 external-extension pattern:

```text
source/rambo/
├── config/extension.toml
├── rambo/
│   ├── actuators/
│   ├── rl/
│   ├── tasks/direct/{rambo_quadruped,rambo_biped}/
│   ├── utils/
│   └── validation/
└── setup.py
```

RAMBO owns its custom math, marker definitions, delayed actuator, and PhysX compatibility adapter; it uses the upstream 2.3.2 COM randomizer rather than patching installed Isaac Lab packages.

On the validation host, PyTorch 2.7's lazy CUDA linear-algebra load needs its
own `torch/lib` directory discoverable. The RAMBO entry points preload that
verified wheel library without changing the simulator packages;
`scripts/rambo/run.sh` provides the same loader environment for standalone
qpth and pytest commands.

## Porting rules

- Preserve the legacy `dt=0.002`, `decimation=5`, and QP → actuator → PhysX → reward/reset/observation ordering on the first port. Bring the custom full `DirectRLEnv.step()` into line with 2.3.2 event, noise, render counter, and reset-rerender semantics without a broad refactor.
- Preserve each controller's topology: quadruped retains its single front end-effector path; biped retains its two-front-end-effector path. Do not reuse quadruped assumptions for biped.
- Reimplement `DelayedDCMotor` as a RAMBO-local actuator: delay 0–10 steps, `Kp=40`, `Kd=1`, friction 0; hip/thigh 21.33 Nm at 30.1 rad/s, calf 40.887 Nm at 15.70 rad/s.
- Replace numeric link/body/Jacobian indices with a validated runtime name mapping in the order base, then FL/FR/RL/RR hip/thigh/calf/foot. Head links are excluded from QP but remain in collision termination checks.
- Keep qpth and its formulation. If Torch 2.7 requires a compatibility fix, patch only the compatibility surface and retain `QPFunction` behavior.
- Move CRL2's wrapper to modern Gymnasium `single_action_space`, `single_observation_space`, and `flatdim`, while retaining its tensor tuple API.
- Cache the reset/step observation in the CRL2 wrapper. Contract checks and PPO setup must not call a RAMBO QP environment's mutating `_get_observations()` and advance its five-frame history before the first policy action.

## Checkpoints and commands

The loader uses `map_location` and explicit `weights_only=False` only after verifying a trusted local checkpoint. It restores the observation normalizer count and rejects a task/schema/hash mismatch.

| Mode | Expected checkpoint | SHA256 | Observation / action |
| --- | --- | --- | --- |
| Quadruped | `model_2000.pt` | `1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706` | 405 / 18 |
| Biped | `model_4000.pt` | `c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6` | 435 / 18 |

The public runner interface is common to both modes:

```bash
python scripts/rambo/play.py --task <task-id> --checkpoint <path>
python scripts/rambo/validate.py --task <task-id> --checkpoint <path> \
  --steps 3000 --enable_cameras --output-dir <directory>
```

Validation uses `seed=42`, one environment, disabled observation noise/domain randomization/random initial state/random episode progress, and a temporary 31-second episode limit. Quadruped uses its zero gait-phase offset. Biped uses the target-runtime/checkpoint-validated fixed `contact_phase_offset_s=19.6`: it advances only the contact/gait clock, not `episode_length_buf`, so it is neither randomized progress nor a warm-up and leaves the full 31-second budget intact. The copied biped contact sequence is extended to cover `19.6 + 31` seconds. Each mode owns one base-mounted front camera at 640×480 with a 0.08-second update period; only a new timestamped frame is consumed.

## Acceptance gates

1. Fresh Python 3.11 shell: `torch.cuda.is_available()` is true, Blackwell `sm_120` is compiled, `pip check` succeeds, and `isaacsim` launches.
2. qpth CPU and CUDA forward/backward tests return finite values, satisfy the expected residual tolerance, and are repeatable.
3. Each task registers, creates, resets, and accepts zero/scripted actions. Tests cover task registration, controller mapping, delay/saturation, CRL2 wrapper dimensions, and checkpoint metadata for both modes.
4. Each checkpoint completes 3,000 control steps without early termination/truncation, NaN/Inf in observations/actions/QP values/GRFs/targets/torques, or torque beyond the actuator limits. The native safety gates are height/orientation ≥ 0.1 m / ≤ 0.75 rad for quadruped and ≥ 0.3 m / ≤ 0.8 rad for biped.
5. Each validation writes 375 fresh RGB frames with timestamps and `summary.json`. Automatic checks reject black, fixed, malformed, or missing frames; a contact sheet/video from both GUI and offscreen output receives a final no-snow/no-corruption visual check.

Only after these gates pass may later work reconnect WAM/data generation, port training, or delete the vendored Isaac Lab source.
