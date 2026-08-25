![RAMBO](docs/source/_static/rambo.png)

# RAMBO: RL-augmented Model-based Whole-body Control for Loco-manipulation

[Jin Cheng](https://jin-cheng.me/)<sup>1</sup>, [Dongho Kang](https://donghokang.net/)<sup>1</sup>, [Gabriele Fadini](https://www.zhaw.ch/en/about-us/person/fadi)<sup>1</sup>, [Guanya Shi](https://www.gshi.me/)<sup>2</sup>, [Stelian Coros](https://crl.ethz.ch/people/coros/index.html)<sup>1</sup>

<sup>1</sup> ETH Zurich, <sup>2</sup> Carnegie Mellon University

Accepted to IEEE Robotics and Automation Letters (RA-L) 2025.

[Paper](https://arxiv.org/pdf/2504.06662) | [ArXiv](https://arxiv.org/abs/2504.06662) | [Video](https://youtu.be/VdZxhLNG6wQ) | [Website](https://jin-cheng.me/rambo.github.io/)

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.0.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v3.0.0--beta2.patch1-silver.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)

## Supported migration target

This branch is the RAMBO migration target for Isaac Sim `6.0.1.0` and Isaac
Lab `v3.0.0-beta2.patch1` at commit
`ffff603eafc6b74264a5261cc0183d6a65390d78`. It deliberately does not follow
an Isaac Lab release branch, `main`, `develop`, nightly builds, or a floating
PyPI package.

The validated host profile is Ubuntu 24.04, Python 3.12.3, PyTorch
`2.10.0+cu128`, and an NVIDIA RTX 4090 (24 GiB, driver 595.71.05, compute
capability 8.9). Treat that as the acceptance profile, not as permission to
silently substitute simulator, Python, Torch, or checkpoint versions.

RAMBO's supported tasks are:

- `Isaac-RAMBO-Quadruped-Go2-v0` — 405 observations / 18 actions.
- `Isaac-RAMBO-Biped-Go2-v0` — 435 observations / 18 actions.
- `Isaac-RAMBO-Quadruped-Button-Go2-v0` — the 405/18 quadruped policy plus
  task-side locomotion and front-left button commands.

The current evidence matrix, command history, known deferred work, and artifact
locations are maintained in [MIGRATION_STATUS.md](MIGRATION_STATUS.md). The
complete migration contract is in
[rambo-isaacsim60-isaaclab30b2-adaptation-plan.md](rambo-isaacsim60-isaaclab30b2-adaptation-plan.md).

## Non-negotiable physics and launcher contract

Every RAMBO configuration, smoke test, validation run, recorder, and production
entry point explicitly selects `PhysxCfg`, sets
`use_newton_actuators=False`, and verifies the live manager as
`isaaclab_physx.physics.physx_manager.PhysxManager` before and after execution.
Actual Newton or OvPhysX manager use is a failure.

The exact Isaac Sim / Isaac Lab dependency graph can install bare
Newton-related packages (including `isaaclab_newton`). Their presence is not a
failure and does not authorize their use. Do not install Newton optional extras
and do not select or run a Newton backend for RAMBO.

Each RAMBO simulator command must explicitly state exactly one audited
visualizer choice:

```text
--viz none     # non-interactive Kit visualizer, suitable for automated RTX sensor runs
--viz kit      # interactive Kit viewport
```

Do not use `--headless`, custom `--experience`, `--kit_args`, or unaudited Kit
arguments for RAMBO. The dedicated launchers reject them before Kit starts. The
generic `scripts/reinforcement_learning/crl2/{play,train}.py` scripts also
reject `Isaac-RAMBO-*` before launching Kit; use only `scripts/rambo/` for
RAMBO.

## NVIDIA EULA

The operator has accepted NVIDIA's EULA for this migration. Pass that consent
only to the individual Isaac Sim process that needs it:

```bash
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh <RAMBO-launcher> --viz none
```

Do not `export` this variable, place it in a shell profile or `.env`, write an
EULA-accepted marker, set `PRIVACY_CONSENT`, or bake it into a Docker image.
Every example below keeps the acknowledgement process-scoped.

## Install the exact native stack

Start from a clean shell without legacy Isaac Sim paths. Prepare the exact
Isaac Lab source checkout, then let the setup script verify its detached commit:

```bash
git clone --branch v3.0.0-beta2.patch1 https://github.com/isaac-sim/IsaacLab.git \
  /workspace/IsaacLab-3.0.0-beta2.patch1
git -C /workspace/IsaacLab-3.0.0-beta2.patch1 checkout --detach \
  ffff603eafc6b74264a5261cc0183d6a65390d78
git -C /workspace/IsaacLab-3.0.0-beta2.patch1 rev-parse HEAD

bash scripts/setup_isaacsim60.sh
source /workspace/venvs/rambo60/bin/activate
```

`scripts/setup_isaacsim60.sh` installs only the pinned official path:

- `isaacsim[all,extscache]==6.0.1.0`;
- `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, and `numpy==2.3.1`;
- the Isaac Lab extensions from the exact checkout, including the pinned tag's
  official core bare nodes (such as `isaaclab_newton`) but no Newton optional
  extras;
- `qpth==0.0.18`, `cvxpy==1.6.7`, `clarabel==0.11.1`, and `scs==3.2.11`.

The direct dependency intent is recorded in
[requirements/isaacsim60.in](requirements/isaacsim60.in). The installer then
consumes [requirements/isaacsim60.lock](requirements/isaacsim60.lock) with
`--no-deps` after the official staged transaction, pinning all 242
non-editable resolved wheels without a second solver decision. Isaac Lab,
CRL2, and RAMBO remain explicit editables: the verifier checks the selected
Isaac Lab extensions' `direct_url` paths against the exact checkout and checks
every lock pin. Never replace either file with a floating or nightly package
transaction. The setup script does not accept the EULA and does not persist it.

`scripts/rambo/run60.sh` is the supported wrapper for simulator launches. It
keeps the pinned PyTorch CUDA shared libraries visible, requires exactly one
`--viz none|kit`, and rejects visualizer/Kit/physics/Newton override arguments
before Python starts. The audited launchers then assert the actual PhysX
manager before and after the runtime.

## Verify the host and official PhysX gate

Use a fresh artifact directory for every run; do not overwrite evidence. The
following Cartpole gate does not import RAMBO and isolates the pinned simulator
stack from RAMBO task issues:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/workspace/migration-output/isaac60/M2/${STAMP}-official-cartpole-direct-16"

OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh \
  scripts/rambo/official_physx_smoke.py \
  --scenario cartpole-direct --num-envs 16 --steps 16 --output-dir "$OUT" --viz none
```

This finite wrapper uses the official `Isaac-Cartpole-Direct-v0` task but is
not the pinned tag's literal `zero_agent.py` command: that upstream script
defines but never applies its `MAX_STEPS` limit, so `--viz none` has no finite
completion condition. The same gate supports `--scenario cartpole`, `go2`, and
`camera`. A successful summary must record `PhysxCfg`, the actual `PhysxManager`,
`use_newton_actuators=false`, and RTX GPU evidence. Do not infer backend use
from installed package names.

## Accepted M2.3 manual Cartpole Kit gate

This is intentionally the only accepted GUI path for M2.3. Do **not** start it
until an operator is ready to watch the visible desktop: it requires `DISPLAY`,
uses the exact official `Isaac-Cartpole-Direct-v0` task with one environment,
and fixes `--viz kit`, explicit PhysX, and a finite 15–300-second dwell. It
does not use xdotool, pyautogui, replayed input, or a synthetic UI substitute.

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
M2_GUI_OUT="/workspace/migration-output/isaac60/M2/${STAMP}-cartpole-direct-kit"
M2_GUI_ATTEST="/workspace/migration-output/isaac60/M2/${STAMP}-cartpole-direct-kit-attestation"

# Run from the visible desktop session (for example DISPLAY=:20). During the
# 45-second window, personally observe the viewport and exercise Play/Stop.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_m2_cartpole_gui_gate.sh \
  --output-dir "$M2_GUI_OUT" --attestation-dir "$M2_GUI_ATTEST" \
  --operator-name "<your-name>" --gui-observation-seconds 45
```

Before Kit closes, the runtime records two read-only `nvidia-smi` samples for
the live Kit PID, its nonzero GPU memory, the selected RTX 4090, and the live
Carb renderer active-GPU/RTX-extension state. It still does not claim that
those records prove the visual observation. After the real child exit is
sealed, the wrapper requires the named operator at an interactive TTY to type
an exact acknowledgement, then validates the sidecar's checksum binding to the
sealed runtime. The attestation statement covers the GUI window, normal
viewport, manual Play/Stop, absence of Vulkan swapchain errors, Kit GPU memory,
and an RTX 4090 rather than llvmpipe. A failed or interrupted run is retained
for diagnosis but cannot receive an attestation; use fresh runtime and sidecar
directories for a genuine retry.

The accepted run is
`/workspace/migration-output/isaac60/M2/20260824T235256Z-cartpole-direct-kit/`
with its sibling `*-attestation/` sidecar. It completed 45.04 seconds and 1083
steps with two live RTX 4090 samples, exact PhysX evidence, exit status zero,
and a checksum-bound post-close operator acknowledgement.

For the separate, non-simulator CUDA/Ada compatibility record, run:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
/workspace/venvs/rambo60/bin/python scripts/rambo/verify_cuda_m2.py \
  --output-dir "/workspace/migration-output/isaac60/M2/${STAMP}-cuda-ada-compatible-contract"
```

This checks the actual CUDA device, capability, finite CUDA arithmetic, and
the wheel's compiled architecture list without launching Kit. The user-approved
M2.1 criterion is successful synchronized CUDA execution on the actual RTX 4090
Ada device; `get_arch_list()` need not literally contain `sm_89`. The artifact
still records the pinned wheel's actual compatible `sm_86` entry and
`native_sm89_compiled=false` without relabelling either fact.

Every acceptance artifact produced by a Kit child must use
`run_runtime_artifact.sh`, not `run60.sh` directly. The wrapper records the
actual post-close shell exit in `process_exit.json` and refreshes
`checksums.sha256`; a passed `summary.json` without a zero, checksum-covered
exit sidecar is only pre-close workload evidence. Verify a non-RGB artifact
without starting Kit:

```bash
/workspace/venvs/rambo60/bin/python scripts/rambo/validate_runtime_artifact.py \
  --artifact-dir "$OUT"
```

Pure Python contract tests do not start Kit or need EULA consent:

```bash
PYTHONPATH=source/rambo:source/crl2 \
  /workspace/venvs/rambo60/bin/python -m pytest -q tests/rambo
```

## Checkpoints and finite policy validation

Only the released checkpoint hashes below are accepted by the RAMBO launchers:

| Mode | Checkpoint | SHA-256 | Iteration |
| --- | --- | --- | --- |
| Quadruped | `model_2000.pt` | `1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706` | 2000 |
| Biped | `model_4000.pt` | `c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6` | 4000 |

The acceptance rollout has 3,000 policy steps at 100 Hz (15,000 physics ticks)
and 375 RGB frames at 640×480 / 12.5 Hz. The biped rollout also fixes the
19.6-second contact/gait phase. All long runs check finite values, terminal
states, height/orientation/torque and joint-speed contracts, and post-warm-up
GPU/RSS growth.

```bash
# Quadruped: finite PhysX-only policy replay without a camera.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh \
  scripts/rambo/physx_quadruped_policy_smoke.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --num-envs 1 --steps 3000 --seed 42 \
  --output-dir /workspace/migration-output/isaac60/M6/$(date -u +%Y%m%dT%H%M%SZ)-quadruped \
  --viz none

# Quadruped: 3,000 steps plus the required 375 RTX RGB frames.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh \
  scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --steps 3000 \
  --output-dir /workspace/migration-output/isaac60/M7/$(date -u +%Y%m%dT%H%M%SZ)-quadruped-rgb \
  --viz none

# Biped: its dedicated gate fixes the 19.6-second validation phase.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh \
  scripts/rambo/physx_biped_policy_smoke.py \
  --checkpoint /workspace/rambo-go2-policies/biped/model_4000.pt \
  --num-envs 1 --steps 3000 --seed 42 \
  --output-dir /workspace/migration-output/isaac60/M9/$(date -u +%Y%m%dT%H%M%SZ)-biped \
  --viz none
```

Run the same `validate.py` command with `Isaac-RAMBO-Biped-Go2-v0` and the
biped checkpoint for its RGB acceptance run. `validate.py` requires a positive
step count divisible by eight so that the camera cadence remains exact. For the
M7/M9 visual-review contract, use the opt-in final third-person diagnostic; it
does not alter the production front-camera mount or cadence and only runs after
the same 3,000-step/375-frame policy rollout has passed.

```bash
# M7: checkpoint rollout plus a final independent robot-and-scene RGBD view.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
M7_OUT="/workspace/migration-output/isaac60/M7/${STAMP}-quadruped-3000-rgb-thirdperson-final"
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --steps 3000 --third-person-diagnostic final \
  --output-dir "$M7_OUT" \
  --viz none

# M9: the same evidence for the released biped checkpoint.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
M9_OUT="/workspace/migration-output/isaac60/M9/${STAMP}-biped-3000-rgb-thirdperson-final"
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/biped/model_4000.pt \
  --seed 42 --steps 3000 --third-person-diagnostic final \
  --output-dir "$M9_OUT" \
  --viz none
```

The successful diagnostic writes `third_person_final_robot_scene_rgb.png`, its
depth array, `third_person_diagnostic_manifest.json`, and a checksum manifest.
Its summary must still show 15,000 physics ticks, 375 production front RGB
frames, the final action step 3,000, unchanged front-camera tick counter, and
actual `PhysxManager` evidence before and after capture.
For the `final` mode, the RAMBO worktree must be clean before `AppLauncher`
starts and when the manifest is written; the artifact records the exact RAMBO
commit, full argv, pinned Isaac Lab tag/commit/version, Isaac Sim version, and
Torch/CUDA/GPU provenance. This prevents a dirty-source technical run from
being represented as final acceptance evidence.
After the wrapper returns zero, verify this stronger final contract offline:

```bash
PYTHONPATH=source/rambo:source/crl2 /workspace/venvs/rambo60/bin/python \
  scripts/rambo/validate_final_third_person_artifact.py --artifact-dir "$M7_OUT"
PYTHONPATH=source/rambo:source/crl2 /workspace/venvs/rambo60/bin/python \
  scripts/rambo/validate_final_third_person_artifact.py --artifact-dir "$M9_OUT"
```

## Button loco-manipulation and evidence recorder

The Button task retains the quadruped checkpoint contract. Its fixed seed-42
program walks during steps 110–269, moves FL vertically during 300–549, presses
along FL x during 550–809, then retracts during 810–909. Acceptance requires a
12 mm press held for five steps, rebound at or below 2 mm, and both base and
joint motion of at least 0.05.

Use `--viz kit` only for the genuine interactive keyboard check. Click the
viewport first; arrows/numpad command the base, `Z`/`X` yaw, `W`/`S`, `A`/`D`,
and `R`/`F` move the FL target, `L` clears commands, and `C` clears a released
success latch. `SPACE` is deliberately not a RAMBO loco-manip mapping and is
recorded as the GUI/timeline conflict during the audited run. Do not claim a
GUI keyboard check from a synthetic input trace.

```bash
# Exploratory teleoperation only; it does not create M8 acceptance evidence.
# Use the dedicated audited command below for the actual GUI gate.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/teleop_loco_manip.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --viz kit

# The only accepted M8 GUI path. Run from the visible desktop session (for
# example DISPLAY=:20). A named operator must focus the viewport and use their
# physical keyboard through the live Selkies WebRTC session: issue a base
# command, move FL, press SPACE once, press the button to at least 12 mm, then
# retract until it reports released.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
M8_GUI_OUT="/workspace/migration-output/isaac60/M8/${STAMP}-gui-keyboard"
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_gui_keyboard_artifact.sh \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --max-steps 3000 --viz kit \
  --gui-artifact-dir "$M8_GUI_OUT" \
  --operator-name "<your-name>" \
  --gui-arm-timeout-s 120

# The dedicated runner has already run this offline-only validator after the
# post-close TTY acknowledgement. It reads immutable evidence only and never
# launches Kit, so it is safe to repeat separately when reviewing an artifact.
PYTHONPATH=source/rambo:source/crl2 /workspace/venvs/rambo60/bin/python \
  scripts/rambo/validate_gui_keyboard_artifact.py --artifact-dir "$M8_GUI_OUT"

# Non-interactive deterministic Button evidence and independent offline audit.
M10_OUT="/workspace/migration-output/isaac60/M10/$(date -u +%Y%m%dT%H%M%SZ)-button"
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run_runtime_artifact.sh \
  scripts/rambo/record_button_physx_episode.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --steps 3000 --output-dir "$M10_OUT" --viz none

PYTHONPATH=source/rambo:source/crl2 /workspace/venvs/rambo60/bin/python \
  scripts/rambo/validate_button_physx_episode.py \
  --artifact-dir "$M10_OUT" --require-acceptance

/workspace/venvs/rambo60/bin/python scripts/rambo/validate_runtime_artifact.py \
  --artifact-dir "$M10_OUT"
```

The audited GUI mode refuses scripted smoke flags and requires the process-scoped
EULA prefix plus a non-empty `DISPLAY` before Kit can start. It records native
Carb callback observations, every control-step state, and explicit PhysX
evidence before and after the run. Only after the Kit child has exited with
status zero does the dedicated runner seal `process_exit.json`, require the
named operator at an interactive TTY to type an exact acknowledgement, then run
the offline validator. Missing, redirected, interrupted, or mismatched
attestations leave the artifact unaccepted. A passing offline validator proves
internal consistency but intentionally does not independently prove the
physical origin of keyboard events. Live human operation of the operator's
physical keyboard through Selkies WebRTC is the intended remote-desktop path;
never use xdotool, pyautogui, scripted or replayed events, automation, or any
injected input source. Retain a failed artifact for diagnosis and use a new
output directory for the next genuine attempt.

The recorder writes actions, observations, post-step state, 375 RGB frames,
timestamps, runtime/checkpoint/PhysX evidence, memory samples, and a checksum
manifest. The offline validator never launches Kit and therefore never selects
any physics backend.

## Container status

The native stack is the acceptance path. Docker and NVIDIA Container Toolkit
are unavailable on the validation host, so no container build or container
runtime claim is made here. [Dockerfile.rambo60](Dockerfile.rambo60) and
[docs/rambo-isaacsim60-docker.md](docs/rambo-isaacsim60-docker.md) are a
deferred, fail-closed build specification: they require an operator-verified
immutable Ubuntu 24.04 base digest, an exact Isaac Lab commit, GPU/RTX checks,
and a process-scoped EULA variable at runtime only.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{cheng2025rambo,
  title={RAMBO: RL-augmented Model-based Optimal Control for Whole-body Loco-manipulation},
  author={Cheng, Jin and Kang, Dongho and Fadini, Gabriele and Shi, Guanya and Coros, Stelian},
  journal={arXiv preprint arXiv:2504.06662},
  year={2025}
}
```

## License

This codebase is under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g. to make demos to advertise commercial products.

## Acknowledgements

- [CAJun](https://github.com/yxyang/cajun/): Our QP based optimization module is inspired by the `cajun` project.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): Our RL framework `crl2` is based on the PPO implementation in rsl_rl.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): The simulator and robot-learning framework used by RAMBO.

## Contact

For collaborations, feedback, or questions, contact Jin Cheng at [jin.cheng@inf.ethz.ch](mailto:jin.cheng@inf.ethz.ch).
