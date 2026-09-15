![RAMBO](docs/source/_static/rambo.png)

# RAMBO: RL-augmented Model-based Whole-body Control for Loco-manipulation

[Jin Cheng](https://jin-cheng.me/)<sup>1</sup>, [Dongho Kang](https://donghokang.net/)<sup>1</sup>, [Gabriele Fadini](https://www.zhaw.ch/en/about-us/person/fadi)<sup>1</sup>, [Guanya Shi](https://www.gshi.me/)<sup>2</sup>, [Stelian Coros](https://crl.ethz.ch/people/coros/index.html)<sup>1</sup>

<sup>1</sup> ETH Zurich, <sup>2</sup> Carnegie Mellon University

Accepted to IEEE Robotics and Automation Letters (RA-L) 2025.

[Paper](https://arxiv.org/pdf/2504.06662) | [ArXiv](https://arxiv.org/abs/2504.06662) | [Video](https://youtu.be/VdZxhLNG6wQ) | [Website](https://jin-cheng.me/rambo.github.io/)

## RAMBO_Data v2

This branch owns the quadruped simulator, assets, camera geometry, teleoperation, telemetry, task success and canonical dataset publication for native 9D adaptation.
WAM-Policy owns model-side preprocessing/cache, training and inference. No WAM or model stack is required by this simulator package.
Read [current scope](docs/current/overview.md), [agent guide](AGENTS.md), and [work index](docs/INDEX.md).

### Runtime

Use the installed workspace simulator environment through `scripts/rambo/run.sh`; use `run60.sh` with explicit `--viz none` or `--viz kit` for Isaac.
After sourcing workspace.env:

```bash
cd "$RAMBO_SIM_ROOT"
scripts/rambo/run.sh -m pytest -q tests/rambo
scripts/rambo/run.sh scripts/rambo/check_import_boundaries.py
OMNI_KIT_ACCEPT_EULA=Y scripts/rambo/run60.sh scripts/rambo/physx_quadruped_policy_smoke.py --checkpoint "$RAMBO_CHECKPOINT_ROOT/quadruped/model_2000.pt" --num-envs 1 --steps 64 --seed 42 --viz none --output-dir "$RUNS_ROOT/<new-run-directory>"
```

The quadruped checkpoint has 405 observations and 18 residual actions, SHA-256 `1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706`.
Native high-level commands remain 9D; the bootstrap does not alter controller physics or train a controller.
Existing Button, Lift-basket, Pull-object and Shoot-ball tasks remain available as runtime references.

### Mounted RGB reference

`bash scripts/rambo/launch_lift_dual_camera.sh` launches the retained `robot-dual-v3` pair.
Go2 ego uses the existing nominal URDF mount and explicit 120-degree diagonal-FOV approximation.
D435i task uses the existing (0.30, 0, 0.34) m mount, downward 65 degrees and 69.4-degree horizontal FOV.
Both streams are 1280x720 at 12.5 Hz. F6/F7 switch ego/task; F8 is an editor debug view.
This is a nominal simulator reference, not per-device calibration or production Dataset v2 acceptance.
Use `scripts/rambo/preview_lift_cameras.py` through run60.sh for bounded RTX diagnostics; it writes evidence, not dataset episodes.

### Compute and evidence

Local Ubuntu 24.04 / RTX 5070 Ti handles debugging, simulation and data synthesis. Necessary local GPU work outside the sandbox is user-authorized.
Full SFT belongs to the user's remote AMD server after local model debugging and job preparation; this branch does not execute model training.
Keep all datasets, checkpoints and run evidence in versioned workspace directories. Prior migration history remains in original Git refs and external restoration/run records.
Native runtime tests are the acceptance target. Docker definitions are retained as unvalidated deployment references; no image build or release is claimed.
