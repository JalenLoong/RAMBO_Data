# RAMBO Isaac Sim 6 迁移状态

最后更新：2026-08-24 UTC。详细验收见 [migration-report.md](migration-report.md)，
环境与 checkpoint 冻结见 [host-manifest.json](host-manifest.json) 和
[checkpoint-manifest.json](checkpoint-manifest.json)。

## 固定目标与边界

- Isaac Sim `6.0.1.0`；Isaac Lab `v3.0.0-beta2.patch1` /
  `ffff603eafc6b74264a5261cc0183d6a65390d78`；Python 3.12.3；Torch
  `2.10.0+cu128`；RTX 4090 / driver 595.71.05。
- 所有 RAMBO execution 显式选择 PhysX，并在运行前后断言
  `isaaclab_physx.physics.physx_manager.PhysxManager` 与
  `use_newton_actuators=false`。不选择或执行 Newton；精确依赖图中的裸 Newton
  package 允许存在，optional Newton extras 未请求。
- EULA 仅作为每个 Isaac Sim 命令的
  `OMNI_KIT_ACCEPT_EULA=Y` 前缀传递；不 export、不写 marker、不进入 Docker image。

## 里程碑

| 阶段 | 状态 | 主要证据 |
| --- | --- | --- |
| P0、M0、M1 | 完成 | baseline 已保护；精确 venv/checkout 已安装 |
| M2 官方 PhysX/RTX/Kit | 通过 | `M2/20260824T103300Z-cartpole-physx/` 至 `M2/20260824T104800Z-camera-physx-warm-restart/` |
| M3–M5 API/QP | 通过 | `MIGRATION_API_INVENTORY.md`；`M4/20260824T111439Z-*-space-contract-r2/`；`M5/20260824T105302Z-qpth-contract/` |
| M6 四足 3000 steps | 通过 | `M6/20260824T113722Z-quadruped-policy-longrun-n1-schedulefixed/` |
| M7 四足 RGB 3000 steps | 通过 | `M7/20260824T115012Z-quadruped-3000-rgb-stagingfixed/` |
| M8 Button non-interactive | 通过 | `M8/20260824T112955Z-teleop-loco-manip/` |
| M8 GUI 真实键盘 | 待人工 | 必须以 `--viz kit` 做实际物理键盘操作；未以合成输入冒充通过 |
| M9 双足 3000 steps / RGB | 通过 | `M9/20260824T115916Z-biped-policy-longrun-n1/`、`M9/20260824T120253Z-biped-3000-rgb/` |
| M10 30 秒 Button recorder | 通过 | `M10/20260824T122000Z-button-episode-3000-provenance-v2/`；381 checksum 全部通过 |
| M11 native freeze | 完成 | `.in`、242-wheel `.lock`、manifest、报告、默认/metadata-only verifier 均已复核 |
| M11 Docker | 延后 | 当前 host 无 Docker Engine/NVIDIA Container Toolkit；未 build/run/push image |

## 最终 M10 摘要

运行验收 commit：`b0bc68eb32c99ba326bde9a447a898951e774b71`。最终 artifact 包含
3000 actions、3000 observations、3000 post states、375 RGB，terminal 为 0；按压
19.99354 mm 连续 17 steps，成功后回弹符合 ≤2 mm，base/FL foot 均超过 0.05 m。
前后 manager 均为 PhysX，离线 `--require-acceptance` 通过，381 文件 checksums
全部通过。

## 当前剩余项

1. 由真实操作者完成 GUI physical-keyboard teleoperation，并保存独立 artifact。
2. 在另一台具有 Docker 与 NVIDIA Container Toolkit 的主机按
   [docs/rambo-isaacsim60-docker.md](docs/rambo-isaacsim60-docker.md) 执行 build、
   GPU verifier 与 PhysX runtime gates；当前没有容器成功声明。
3. GitHub 凭证可用后再推送本地迁移分支；此前的 push 因无凭证失败，不影响本地
   commit 和原生验收。
