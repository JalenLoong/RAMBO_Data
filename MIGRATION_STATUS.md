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
  package（其中 `isaaclab_newton` 是该 tag 官方 core install 的裸节点）允许存在，
  optional Newton extras 未请求。
- EULA 仅作为每个 Isaac Sim 命令的
  `OMNI_KIT_ACCEPT_EULA=Y` 前缀传递；不 export、不写 marker、不进入 Docker image。

## 里程碑

| 阶段 | 状态 | 主要证据 |
| --- | --- | --- |
| P0、M0、M1 | 完成 | baseline 已保护；精确 venv/checkout 已安装 |
| M2 官方 PhysX/RTX/Kit | **正式未完成** | M2.1 仅有 CUDA/Ada **运行兼容性**证据：`M2/20260824T135749Z-cuda-ada-compatible-contract/`；fixed wheel 的 arch list 无 `sm_89`，故原计划的字面断言未通过，须正式修订验收标准后才能计入 M2.1。M2.2 的有限 Direct Cartpole 16-env 替代 workload 通过：`M2/20260824T135518Z-official-cartpole-direct-16-physx/`，但旧 summary 无运行前 backend 和 post-close exit sidecar；已修代码，待 sealed rerun。pinned `zero_agent.py --viz none` 无有限退出路径，不能伪称字面命令通过。M2.4 Go2+RGB 的旧双启动也是 pre-close workload evidence，待 sealed rerun；M2.3 GUI 已按本轮范围延期。 |
| M3 API inventory | 待最终 clean commit 后重扫 | 现有账本和 `/workspace/migration_logs/{quaternion-scan,api-inventory}.txt` 是当前新增源码之前的快照；不可作为最终 source freeze。 |
| M4 API/空间 contract | 旧 workload 技术通过、最终 provenance/exit 待绑定 | `M4/20260824T111439Z-*-space-contract-r2/`；最终 clean commit 后需以 sealed runtime artifact 重跑。 |
| M5.1 qpth contract | 通过（范围受限） | CPU/CUDA toy-qpth forward/backward、KKT、残差与确定性：`M5/20260824T105302Z-qpth-contract/`；这不是旧版 RAMBO QP snapshot 对比 |
| M5.2 RAMBO 5.1 QP 数值对比 | **已批准延期** | 计划要求的旧 baseline mass/Jacobian/contact/desired-force/constraint 及 solution/torque/contact/residual snapshot 未保存，当前主机不能诚实重建或以新端数据代替；按当前用户范围，它不阻断本轮后续自动化工作。 |
| M6 四足 | 当前 workload 技术子门禁大体通过；sealed rerun 待执行 | 405/18、16-step 与 production walking 3000-step/0 terminal：`M6/20260824T112247Z-*`、`M6/20260824T113722Z-quadruped-policy-longrun-n1-schedulefixed/`；独立的全支撑零 action 100/1000 诊断：`M6/20260824T134233Z-*`、`M6/20260824T134254Z-*`（不是 production walking contact schedule）；首个真实 policy→QP transition：`M6/20260824T133438Z-quadruped-first-policy-qp-transition/`。这些旧 artifacts 无 post-close exit sidecar，待 clean commit 后以 wrapper 重跑。production schedule 的旧零 action 1000-step 在约 152 step terminal，必须保留且不计通过。 |
| M7 四足 RGB 3000 steps | 技术子门禁通过；最终 clean-source/exit 绑定待重跑 | `M7/20260824T131656Z-quadruped-3000-rgb-thirdperson-final/` 的 rollout/RGB/第三人称技术证据通过；其发生在 dirty source 且无 post-close exit sidecar，新的 final provenance/exit gate 已要求 clean tree 后重跑。 |
| M8 Button non-interactive | 通过 | `M8/20260824T112955Z-teleop-loco-manip/` |
| M8 GUI 真实键盘 | 待人工 | 必须以 `--viz kit` 做实际物理键盘操作；未以合成输入冒充通过 |
| M9 双足 | 技术子门禁通过；sealed clean rerun 待执行 | 3000/RGB/第三人称技术 artifact：`M9/20260824T132054Z-biped-3000-rgb-thirdperson-final/`（dirty-source、无 post-close exit sidecar，待 clean rerun）；FL/FR + QP contact override 两步运行时独立性：`M9/20260824T134918Z-biped-front-leg-independence-physx/`（旧 provenance/schema v1，待 clean source/schema v2 sealed rerun）。M2.3/M8 GUI 已按本轮范围延期。 |
| M10 30 秒 Button recorder | workload 技术子门禁通过、sealed rerun 待执行 | `M10/20260824T122000Z-button-episode-3000-provenance-v2/`；381 checksum 全部通过，但旧 artifact 无 post-close exit sidecar；计划要求在完整 simulator migration 后才将其作为最终数据合成验收 |
| M11 native freeze | 阶段性可复现快照、最终冻结待当前范围的 sealed runtime gate | `.in`、242-wheel `.lock`、manifest、默认/metadata-only verifier 已复核；最终 host/commit freeze 必须等待 M3/M4、M6/M7/M9/M10 sealed clean rerun 与最终源码 commit；M2.3/M8 GUI、M5.2 和 Docker 已按本轮范围延期 |
| M11 Docker | **已批准延期** | 当前 host 无 Docker Engine/NVIDIA Container Toolkit；本轮不 build/run/push image |

## 最终 M10 摘要

运行验收 commit：`b0bc68eb32c99ba326bde9a447a898951e774b71`。最终 artifact 包含
3000 actions、3000 observations、3000 post states、375 RGB，terminal 为 0；按压
19.99354 mm 连续 17 steps，成功后回弹符合 ≤2 mm，base/FL foot 均超过 0.05 m。
前后 manager 均为 PhysX，离线 `--require-acceptance` 通过，381 文件 checksums
全部通过。

## 补充的 M2 可复现证据

M2.1 的非模拟 CUDA artifact
`/workspace/migration-output/isaac60/M2/20260824T135749Z-cuda-ada-compatible-contract/`
已复核：Torch `2.10.0+cu128` / CUDA 12.8 在 RTX 4090 capability `(8,9)` 上完成
同步 CUDA 算术。该精确 wheel 的 `get_arch_list()` 没有 native `sm_89`，但有
`sm_86`；artifact 同时保存这两个事实和成功算术，按 NVIDIA 的 Ampere→Ada
forward-compatibility 解释为**运行兼容**，绝不把 native `sm_89` 伪报为存在。
因此它不能满足原计划字面 `assert "sm_89" in torch.cuda.get_arch_list()`，除非验收
标准被正式修订。

M2.2 的有限官方任务 wrapper artifact
`/workspace/migration-output/isaac60/M2/20260824T135518Z-official-cartpole-direct-16-physx/`
已复核：官方 `Isaac-Cartpole-Direct-v0`、16 environments、16 steps、显式
`PhysxCfg`、前后 `PhysxManager`、`use_newton_actuators=false` 和 checksum 均通过。
固定 tag 的 `zero_agent.py` 被保留为非验收的上游限制：
`135150Z-*` 因它选到无 gymnasium 的宿主 venv 失败；`135239Z-*` 在正确 venv 中
确认 `--viz none` 无限循环后被人工中止，只有 console log，不能算成功。

新建的 `scripts/rambo60/official_go2_smoke.py` 不导入 RAMBO，只使用官方
`UNITREE_GO2_CFG`。在 RTX 4090 上以显式 `PhysxCfg` 和
`use_newton_actuators=false` 分别运行 1000 physics ticks 两次：

- warm-up：`/workspace/migration-output/isaac60/M2/20260824T125356Z-official-go2-rgb-1000-warmup/`；
- clean restart：`/workspace/migration-output/isaac60/M2/20260824T125434Z-official-go2-rgb-1000-restart/`。

两份 `summary.json` 都记录单个 Go2、640×480 Isaac RTX RGB、11,000 次有限 state
检查，且运行前后 manager 均为
`isaaclab_physx.physics.physx_manager.PhysxManager`。每个目录的
`go2_rgb.png` 与 `summary.json` 均由各自的 `checksums.sha256` 覆盖并已复核。

M3 的旧原始扫描仍保留在 `/workspace/migration_logs/quaternion-scan.txt` 和
`/workspace/migration_logs/api-inventory.txt`，但在最终 clean commit 后必须重新生成；
前者旧快照中两处 identity 都是已审计的 **XYZW** `[0,0,0,1]`，不得误标为 WXYZ。

## 当前剩余项

1. 由真实操作者完成 M2.3 GUI 观察：以 M2.2 的**同一 Direct Cartpole task**完成 window/viewport/play-stop、无 Vulkan swapchain error、Kit RTX GPU memory、active RTX 4090 而非 llvmpipe；不得用自动化或日志替代。已有 `Isaac-Cartpole-v0` 的短 Kit 启动 artifact 只能证明该任务的 PhysX/Kit 启动，不能替代此门禁。
2. 提交最终源码后，使用 `scripts/rambo/run_runtime_artifact.sh` 重跑非 GUI 的 M2.2/M2.4、M4、M6、M7、M9 和 M10；每个接受 artifact 必须有 checksum-covered `process_exit.json` 且为 0。
3. 重跑 M3 扫描，并以 clean-source provenance + sealed exit 重跑 M7/M9 final 3000-step RGB diagnostics；M9 两步独立性使用 schema v2 clean-source artifact。
4. M2.3/M8 GUI、M5.2 历史 QP snapshot 与 M11 Docker 已按当前用户范围批准延期；它们保留为后续工作，不写成已完成。
5. 在上述当前范围 runtime gate 和最终源码 commit 后重生 M11 host/commit freeze；GitHub 凭证可用后再推送本地迁移分支，此前的 push 因无凭证失败，不影响本地
   commit 和原生验收。
