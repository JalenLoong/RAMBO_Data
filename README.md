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
- `torch==2.10.0+cu128` and `torchvision==0.25.0+cu128`;
- the Isaac Lab extensions from the exact checkout, with bare transitive graph
  nodes only and no Newton optional extras;
- `qpth==0.0.18`, `cvxpy==1.6.7`, `clarabel==0.11.1`, and `scs==3.2.11`.

The direct dependency intent is recorded in
[requirements/isaacsim60.in](requirements/isaacsim60.in). Never replace it
with a floating or nightly package transaction. The setup script does not
accept the EULA and does not persist it.

`scripts/rambo/run60.sh` is the supported wrapper for simulator launches. It
keeps the pinned PyTorch CUDA shared libraries visible while delegating
fail-closed visualizer and PhysX checks to the launchers.

## Verify the host and official PhysX gate

Use a fresh artifact directory for every run; do not overwrite evidence. The
following Cartpole gate does not import RAMBO and isolates the pinned simulator
stack from RAMBO task issues:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="/workspace/migration-output/isaac60/M2/${STAMP}-official-cartpole"

OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/official_physx_smoke.py \
  --scenario cartpole --steps 16 --output-dir "$OUT" --viz none
```

The same gate supports `--scenario go2` and `--scenario camera`. A successful
summary must record `PhysxCfg`, the actual `PhysxManager`,
`use_newton_actuators=false`, and RTX GPU evidence. Do not infer backend use
from installed package names.

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
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/physx_quadruped_policy_smoke.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --num-envs 1 --steps 3000 --seed 42 \
  --output-dir /workspace/migration-output/isaac60/M6/$(date -u +%Y%m%dT%H%M%SZ)-quadruped \
  --viz none

# Quadruped: 3,000 steps plus the required 375 RTX RGB frames.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --steps 3000 \
  --output-dir /workspace/migration-output/isaac60/M7/$(date -u +%Y%m%dT%H%M%SZ)-quadruped-rgb \
  --viz none

# Biped: its dedicated gate fixes the 19.6-second validation phase.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/physx_biped_policy_smoke.py \
  --checkpoint /workspace/rambo-go2-policies/biped/model_4000.pt \
  --num-envs 1 --steps 3000 --seed 42 \
  --output-dir /workspace/migration-output/isaac60/M9/$(date -u +%Y%m%dT%H%M%SZ)-biped \
  --viz none
```

Run the same `validate.py` command with `Isaac-RAMBO-Biped-Go2-v0` and the
biped checkpoint for its RGB acceptance run. `validate.py` requires a positive
step count divisible by eight so that the camera cadence remains exact.

## Button loco-manipulation and evidence recorder

The Button task retains the quadruped checkpoint contract. Its fixed seed-42
program walks during steps 110–269, moves FL vertically during 300–549, presses
along FL x during 550–809, then retracts during 810–909. Acceptance requires a
12 mm press held for five steps, rebound at or below 2 mm, and both base and
joint motion of at least 0.05.

Use `--viz kit` only for the genuine interactive keyboard check. Click the
viewport first; arrows/numpad command the base, `Z`/`X` yaw, `W`/`S`, `A`/`D`,
and `R`/`F` move the FL target, `L` clears commands, and `C` clears a released
success latch. Do not claim a GUI keyboard check from a synthetic input trace.

```bash
# Interactive operator check; this is deliberately not --headless.
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/teleop_loco_manip.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --viz kit

# Non-interactive deterministic Button evidence and independent offline audit.
M10_OUT="/workspace/migration-output/isaac60/M10/$(date -u +%Y%m%dT%H%M%SZ)-button"
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh \
  scripts/rambo/record_button_physx_episode.py \
  --checkpoint /workspace/rambo-go2-policies/quadruped/model_2000.pt \
  --seed 42 --steps 3000 --output-dir "$M10_OUT" --viz none

PYTHONPATH=source/rambo:source/crl2 /workspace/venvs/rambo60/bin/python \
  scripts/rambo/validate_button_physx_episode.py \
  --artifact-dir "$M10_OUT" --require-acceptance
```

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
