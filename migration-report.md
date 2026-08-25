# RAMBO Isaac Sim 6 / Isaac Lab 3 迁移报告

生成时间：2026-08-24（UTC）
状态来源：[MIGRATION_STATUS.md](MIGRATION_STATUS.md)。本报告是证据状态记录，
不是“全部迁移完成”的声明。

## 当前结论

RAMBO 的目标运行路径已迁移为 Isaac Sim `6.0.1.0`、Isaac Lab
`v3.0.0-beta2.patch1` (`ffff603eafc6b74264a5261cc0183d6a65390d78`)、
Python 3.12.3 和 Torch `2.10.0+cu128`。所有已运行的 RAMBO simulation、
smoke、validation 和 recorder 都显式配置 `PhysxCfg`、
`use_newton_actuators=false`，并在运行前后记录实际
`isaaclab_physx.physics.physx_manager.PhysxManager`。没有任何 RAMBO runtime
选择或执行 Newton。

精确官方安装图中的裸 Newton package（包括 pinned tag 官方 core install 的
`isaaclab_newton`）允许存在；这既不是失败条件，也绝不授权 RAMBO 使用 Newton。
安装输入和 canonical setup 均不请求可选 Newton extras。EULA 只由每条 Kit
启动命令的 `OMNI_KIT_ACCEPT_EULA=Y` 前缀传递，不写入环境、marker 或镜像。

本轮已批准延期两项：M5.2 的 Isaac Sim 5.1 历史 QP snapshot 比较，以及 M11
Docker 验收。它们保留为后续工作，不作为当前自动化 PhysX 交付的阻断条件。非 GUI
long-runtime 已在 clean source `358daa5` 上封存，新增短门禁绑定 `078bfa2`；M11
current-scope native freeze 已绑定完整 canonical input list，所有这些路径都不选择或执行
Newton。

新增的 `run_runtime_artifact.sh` 会在 Kit 子进程真正退出后写入
`process_exit.json` 并重算 checksum。此前所有 Kit artifact 的 `summary.json`
均在 close 前写出，因此只能称为 pre-close workload evidence；本轮已用 sealed
artifact 取代需要接受的非 GUI runtime 证据。

## 证据矩阵

| 门禁 | 严格状态 | 证据与限定 |
| --- | --- | --- |
| M0–M1 | 完成 | legacy baseline 已保护；目标 venv、固定 Isaac Sim/Isaac Lab checkout、host/checkpoint manifests 均已建立。 |
| M2.1 CUDA | 按用户修订标准通过 | `M2/20260824T135749Z-cuda-ada-compatible-contract/` 证明 Torch CUDA 12.8 在 RTX 4090 `(8,9)` 完成同步 CUDA 算术。用户已明确接受实机成功，不强制 `get_arch_list()` 字面包含 `sm_89`；artifact 仍如实记录 native `sm_89=false`、兼容 arch `sm_86`。 |
| M2.2 官方 Cartpole | sealed 通过 | `M2/20260824T142949Z-official-cartpole-direct-16-physx-sealed/`：官方 `Isaac-Cartpole-Direct-v0`、16 env、16 step、显式 PhysX、前后 `PhysxManager` 与 post-close exit 0。literal `zero_agent.py --viz none` 仍无有限退出路径，不能伪称 literal 命令通过。 |
| M2.3 Kit GUI | sealed + 人工声明通过 | `M2/20260824T235256Z-cartpole-direct-kit/` + `*-attestation/`：真实操作者完成 45.04 秒观察；runtime 记录 1083 steps、两次 live RTX 4090/`rtx`/`omni.hydra.rtx` evidence、PhysX 前后 FQN 与 exit 0，post-close TTY 声明已 checksum 绑定。 |
| M2.4 官方 Go2 + RGB | sealed 通过 | warm-up `M2/20260824T143007Z-official-go2-rgb-1000-warmup-sealed/` 与 clean restart `M2/20260824T143029Z-official-go2-rgb-1000-restart-sealed/`：单 Go2、1000 tick、640×480 Isaac RTX RGB、PhysX 前后 manager、checksums 与 exit 0。 |
| M3 API inventory | 完成（clean-source） | `M3/20260824T142802Z-api-inventory-clean/` 绑定 `e449b1d`，源码 clean、checksums 已复核；扫描仅保留两处已审计 XYZW `[0,0,0,1]` identity 候选。 |
| M4 API/空间合同 | sealed 通过 | 四足 `M4/20260824T143107Z-quadruped-space-contract-sealed/` 与双足 `M4/20260824T143117Z-biped-space-contract-sealed/`：各 1 env/5 step，PhysX 前后 FQN 与 exit 0。 |
| M5.1 qpth | 通过（范围受限） | `M5/20260824T105302Z-qpth-contract/` 覆盖 CPU/CUDA toy qpth forward/backward、KKT、残差与确定性；不等于历史 RAMBO QP 对比。 |
| M5.2 RAMBO 5.1 QP 数值比较 | **已批准延期** | 所需旧端 snapshot 未保存；本轮不以其阻断后续自动化工作。 |
| M6 四足 | sealed 通过 | `M6/20260824T143140Z-quadruped-policy-3000-first-transition-sealed/` 完成 3000-step production policy→QP rollout 和 first transition；`M6/20260824T151140Z-quadruped-policy-16env-16step-sealed/` 在 current source 上补齐 16-env/16-step（405/18、256 env-steps、PhysX 前后 FQN、exit 0）；独立 `143407Z-*`/`143422Z-*` 100/1000-step 全支撑零动作诊断也通过，但不替代 production schedule。 |
| M7 四足 RGB | sealed clean-source 通过 | `M7/20260824T143537Z-quadruped-3000-rgb-thirdperson-final-sealed/`：3000 policy step、15,000 physics tick、375 base-mounted 前视 RGB、终态独立 RGBD、PhysX 前后 FQN、checksum 与 exit 0。前视流证明生产 sensor/cadence/scene；绑定同一 final state 的 third-person RGBD 证明 robot + scene，不伪称机器人在每个前视像素帧中可见。 |
| M8 Button 非交互 | sealed 通过 | M10 current artifact 覆盖 deterministic Button press/hold/release 且满足 30 秒数据合同。 |
| M8 GUI 真实键盘 | sealed + 真人声明通过 | `M8/20260825T000552Z-gui-keyboard/`：3000 states、5324 Carb callbacks、239 base-command states、363 FL-velocity states、18.06 mm max press、成功后回弹、2 次 SPACE、PhysX 前后 FQN、exit 0、Selkies 真人声明与 checksum 均通过。 |
| M9 双足 | sealed clean-source 通过 | `M9/20260824T143926Z-biped-3000-rgb-thirdperson-final-sealed/` 完成 3000/15,000/375/RGBD；`M9/20260824T151210Z-biped-policy-16env-16step-sealed/` 在 current source 上补齐 16-env/16-step（435/18、256 env-steps、PhysX 前后 FQN、exit 0）；`144321Z-biped-front-leg-independence-physx-schema-v2-sealed/` 通过 FL/FR 独立性。RGB visibility 的前视流 + 同 rollout third-person 解释与 M7 相同。 |
| M10 Button recorder | sealed 通过 | `M10/20260824T144344Z-button-episode-3000-sealed/`：3000 action/observation/post-state、375 RGB、Button acceptance、382 文件 checksum 与 exit 0。 |
| M11 native freeze | current-scope 最终完成 | `M11/20260825T002300Z-native-freeze-final-gui-v7/`：gpu-required verifier、242 wheel lock、无 optional Newton extras、31 项 canonical inputs、17 项接受工件索引和 M2.1/M2.3/M8 hashes 全部绑定。 |
| M11 Docker | **已批准延期** | 当前 host 无 Docker Engine/NVIDIA Container Toolkit；本轮不 build/run/push。 |

## 关键技术说明

### M2.1 的 CUDA/Ada 结果

artifact 记录 Torch `2.10.0+cu128`、CUDA 12.8、RTX 4090 capability `(8,9)`、
成功 CUDA 算术和 `sm_86` compiled arch。PyTorch 的
[`get_arch_list`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html)
返回 wheel 编译时包含的架构，而不是设备实时 capability；NVIDIA 的
[Ada compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.0/ada-compatibility-guide/index.html)
说明 Ampere binary 可向前兼容 Ada。用户已明确接受 RTX 4090 实机 CUDA 成功作为
验收标准，因此 M2.1 通过；这不改变或掩盖 wheel 未列出 native `sm_89` 的事实。

### 四元数和 QP 边界

目标 Isaac Lab 3 public simulator-facing 数据在本迁移涉及的 articulation/camera
边界为 **XYZW**。RAMBO 的历史 QP/checkpoint math 保留 WXYZ，只允许经
`xyzw_to_wxyz` / `wxyz_to_xyzw` 做显式、局部转换；不得对 405/435D observation、
18D action 或 checkpoint 作全局重排。

### M7/M9 provenance 修复

新的 `--third-person-diagnostic final` 会在创建 `AppLauncher` **之前**捕获并要求：
RAMBO Git HEAD、porcelain clean status、完整 Python argv、Isaac Lab exact tag/HEAD/
package version、Isaac Sim package version、Torch/CUDA/GPU。manifest 写入时再次检查
clean source。当前 M7/M9 sealed artifacts 均满足此合同，并分别保存 375 张生产前视
RGB、终态独立 RGBD、完整 provenance、checksum 与 post-close exit evidence。

### M7/M9 的前视与机器人可见性

RAMBO 的生产 RGB 是单个挂在 Go2 base 前方的前视相机；因此 contact sheet 中主要是
前方场景网格，不能诚实地说机器人本体出现在每张前视图里。两个 final artifact 在同一次
通过的 3000-step rollout 结束后，保持前视 camera tick 和 robot state 不变，额外捕获对准
最终 robot root 的 third-person RGBD。故 375 张前视 RGB 证明生产 sensor、12.5 Hz、scene
与连续性；绑定的 third-person RGBD 证明 robot + scene 可见。这保持单前视相机的既有语义，
不以改相机或重跑来掩盖证据边界。

## 尚需完成的工作

当前用户批准的 native migration scope 已完成。M5.2 历史 snapshot 和 M11 Docker
已批准延期，保留其现有证据与 runbook，待后续
   具备输入/主机条件时恢复。

所有 artifact 保留历史失败和非验收尝试，以便审计；它们不会被删除或改写为成功。
