![RAMBO](docs/source/_static/rambo.png)

# RAMBO: RL-augmented Model-based Whole-body Control for Loco-manipulation

[Jin Cheng](https://jin-cheng.me/)<sup>1</sup>, [Dongho Kang](https://donghokang.net/)<sup>1</sup>, [Gabriele Fadini](https://www.zhaw.ch/en/about-us/person/fadi)<sup>1</sup>, [Guanya Shi](https://www.gshi.me/)<sup>2</sup>, [Stelian Coros](https://crl.ethz.ch/people/coros/index.html)<sup>1</sup>

<sup>1</sup> ETH Zurich, <sup>2</sup> Carnegie Mellon University

Accepted to IEEE Robotics and Automation Letters (RA-L) 2025.

[Paper](https://arxiv.org/pdf/2504.06662) | [Arxiv](http://arxiv.org/abs/2504.06662) | [Video](https://youtu.be/VdZxhLNG6wQ) | [Website](https://jin-cheng.me/rambo.github.io/)

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/v2.3.2/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)

## Current milestone

This branch targets Isaac Sim 5.1 / Isaac Lab 2.3.2 on Linux for **checkpoint playback and rendering only**. It supports the two RAMBO modes as separate first-class tasks:

- `Isaac-RAMBO-Quadruped-Go2-v0`
- `Isaac-RAMBO-Biped-Go2-v0`

The milestone validates original checkpoints, QP/controller execution, a 30-second deterministic rollout, and clean GUI/offscreen RGB for both modes. Training, WAM integration, 15-D high-level actions, trajectory data generation, Button/Manipulator tasks, and teleoperation remain outside this migration milestone. The Isaac Sim 4.5 experiments are preserved at `legacy-isaacsim-4.5-local`; do not use that Python 3.10 runtime for this branch.

The supported entry points on this branch are `scripts/setup_isaacsim51.sh` and `scripts/rambo/{play,validate,smoke}.py`. Historical `isaaclab.sh`, `apps/`, `scripts/environments/`, and `scripts/reinforcement_learning/` remain only as legacy references and are intentionally unsupported after the vendored Isaac Lab packages were removed; use the legacy tag for those workflows.

## Runtime requirements

- Ubuntu 22.04 x86_64, Python 3.11, and at least 16 GB GPU VRAM.
- NVIDIA production driver 580.65.06 or newer. The validation machine uses an RTX 5080 with driver 580.159.03.
- A dedicated virtual environment. Isaac Sim 5.x requires Python 3.11; it cannot share the legacy Isaac Sim 4.5/Python 3.10 environment.

The dependency intent is recorded in [requirements/isaacsim51.in](requirements/isaacsim51.in). It pins Isaac Lab 2.3.2, Isaac Sim 5.1 through the official Isaac Lab pip extra, PyTorch 2.7.0 CUDA 12.8, `numpy<2`, and `qpth==0.0.18`. The qpth CVXPY stack is pinned to the versions compatible with Isaac Sim's own `packaging==23.0` and `osqp==0.6.7.post3` requirements.

## Install

Install Python 3.11 and its venv module if they are not already available, then run the setup script from the repository root:

```bash
bash scripts/setup_isaacsim51.sh
source /workspace/venvs/rambo51/bin/activate
```

Set `RAMBO_VENV` to choose another venv location:

```bash
RAMBO_VENV=/opt/venvs/rambo51 bash scripts/setup_isaacsim51.sh
```

The script installs the official pip bundle—`isaaclab[isaacsim,all]==2.3.2`—rather than this repository's vendored `source/isaaclab*` packages. Isaac Lab's pip wheel bundles its own `isaaclab_assets` extension source without automatically installing it; the script installs that exact bundled official asset package for `UNITREE_GO2_CFG`. RAMBO does not install or import the broad `isaaclab_tasks` example-task package: its two configuration-registry operations live in the RAMBO extension. It also applies the required `setuptools<81` and `flatdict==4.0.1` compatibility pins before installing Isaac Lab. The first `isaacsim` launch may take several minutes to download extensions and asks you to accept NVIDIA's EULA.

For a clean shell, do not export a legacy Isaac Sim `PYTHONPATH`; the setup script verifies the new interpreter and runs `pip check` after installing RAMBO and CRL2 in editable mode.

The RAMBO playback and validation entry points preload the pinned PyTorch CUDA
linear-algebra and NVRTC libraries themselves. For a standalone CUDA qpth
command from a fresh shell, use `scripts/rambo/run.sh` so the wheel's
`torch/lib` directory is present in the dynamic-loader path. The wrapper also
uses a writable Python-bytecode cache, which avoids incompatible bundled
extension bytecode from another Python 3.11 micro release:

```bash
scripts/rambo/run.sh -m pytest -q tests/rambo/test_qpth_contract.py
```

## Preflight

Before running RAMBO, confirm that the new environment can see the GPU and launch Isaac Sim:

```bash
python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__.startswith("2.7.0"), torch.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("compiled architectures:", torch.cuda.get_arch_list())
PY

isaacsim
```

The RTX 5080 build should report Blackwell capability `(12, 0)` and include `sm_120` in the compiled Torch architectures. Resolve any Isaac Sim GUI/offscreen rendering problem before debugging RAMBO.

Run the finite official-Go2 smoke test next. It does not import RAMBO, so it
separates the Isaac Sim/Isaac Lab/driver gate from task-migration failures. The
second command also exercises the camera-capable headless Kit experience and
saves one 640×480 RGB frame as a NumPy array:

```bash
scripts/rambo/run.sh scripts/rambo/smoke.py --steps 100
scripts/rambo/run.sh scripts/rambo/smoke.py --headless --steps 100 \
  --enable-camera --camera-output /tmp/rambo-go2-smoke-rgb.npy
```

The smoke script intentionally does not accept NVIDIA's EULA; the first Kit
launch uses the normal operator-controlled prompt.

## Playback and validation

Both commands require a trusted local checkpoint. The validator verifies the task-specific checkpoint SHA256 and dimensions, uses one environment with fixed seed 42, and writes RGB frames plus `summary.json` to the output directory. Its biped acceptance configuration uses a fixed 19.6-second contact/gait phase while keeping random episode progress disabled and the episode clock at zero; this is recorded as `validation_config.contact_phase_offset_s` in `summary.json`. Do not pass `--contact-phase-offset-s` for the standard acceptance run; it is a diagnostic override.

```bash
# Quadruped: 405 observations, 18 actions
scripts/rambo/run.sh scripts/rambo/play.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint /workspace/rambo/logs/crl2/rambo_quadruped/hf_quadruped/model_2000.pt

scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint /workspace/rambo/logs/crl2/rambo_quadruped/hf_quadruped/model_2000.pt \
  --steps 3000 --enable_cameras \
  --output-dir /workspace/rambo-validation/quadruped

# Biped: 435 observations, 18 actions
scripts/rambo/run.sh scripts/rambo/play.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/biped/model_4000.pt

scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint /workspace/rambo-go2-policies/biped/model_4000.pt \
  --steps 3000 --enable_cameras \
  --output-dir /workspace/rambo-validation/biped
```

See [rambo-isaacsim5-adaptation-plan.md](rambo-isaacsim5-adaptation-plan.md) for migration scope, checkpoint hashes, and acceptance criteria.

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

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.

## Acknowledgements

- [CAJun](https://github.com/yxyang/cajun/): Our QP based optimization module is inspired by the `cajun` project.
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl): Our RL framework `crl2` is based on the PPO implementation in rsl_rl.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): The simulator and robot-learning framework used by RAMBO.

## Contact

For collaborations, feedback, or questions, contact Jin Cheng at [jin.cheng@inf.ethz.ch](mailto:jin.cheng@inf.ethz.ch).
