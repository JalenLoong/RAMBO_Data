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
Docker 验收。它们保留为后续工作，不作为当前自动化 PhysX 交付的阻断条件。当前
已形成 clean local commit `e449b1d`，并已完成绑定该 commit 的 M3
quaternion/API inventory 扫描；后续只需在干净源码上完成 M7/M9 的 sealed runtime
provenance 证据。

新增的 `run_runtime_artifact.sh` 会在 Kit 子进程真正退出后写入
`process_exit.json` 并重算 checksum。此前所有 Kit artifact 的 `summary.json`
均在 close 前写出，因此只能称为 pre-close workload evidence；本轮会用 sealed
artifact 重新取代需要接受的非 GUI runtime 证据。

## 证据矩阵

| 门禁 | 严格状态 | 证据与限定 |
| --- | --- | --- |
| M0–M1 | 完成 | legacy baseline 已保护；目标 venv、固定 Isaac Sim/Isaac Lab checkout、host/checkpoint manifests 均已建立。 |
| M2.1 CUDA | **未按原计划通过** | `M2/20260824T135749Z-cuda-ada-compatible-contract/` 成功证明 Torch CUDA 12.8 可在 RTX 4090 `(8,9)` 做同步 CUDA 算术；但该 fixed wheel 的 `torch.cuda.get_arch_list()` 没有 `sm_89`（有 `sm_86`），故原计划的字面 `assert "sm_89" ...` 未满足。它只能称运行兼容性证据，除非正式修改验收标准。 |
| M2.2 官方 Cartpole | workload 技术替代 gate 通过；sealed rerun 待执行 | `M2/20260824T135518Z-official-cartpole-direct-16-physx/`：官方 `Isaac-Cartpole-Direct-v0`、16 env、16 step、显式 PhysX 和后端 evidence。该旧 artifact 没有运行前 backend 与 post-close exit sidecar；修复后需 sealed rerun。固定 tag 的 literal `zero_agent.py --viz none` 没有有限退出路径，不能伪称 literal 命令通过。 |
| M2.3 Kit GUI | 待人工 | 必须在与 M2.2 相同的 Direct Cartpole task 上，由真实操作者观察 window、viewport、play/stop、Vulkan、Kit GPU memory 和 RTX 4090（非 llvmpipe）。现有 `Isaac-Cartpole-v0` 短 Kit 启动仅支持 Kit/PhysX 启动事实，不能替代此门禁。 |
| M2.4 官方 Go2 + RGB | workload 技术通过；sealed rerun 待执行 | warm-up `M2/20260824T125356Z-official-go2-rgb-1000-warmup/` 与 clean restart `M2/20260824T125434Z-official-go2-rgb-1000-restart/`：单 Go2、1000 tick、640×480 Isaac RTX RGB、前后 PhysX manager、`false` 和 checksums；旧 artifact 无 post-close exit sidecar。 |
| M3 API inventory | 完成（clean-source） | `M3/20260824T142802Z-api-inventory-clean/` 绑定 `e449b1d`，源码 clean、checksums 已复核；扫描仅保留两处已审计 XYZW `[0,0,0,1]` identity 候选。 |
| M4 API/空间合同 | workload 技术通过；sealed clean rerun 待执行 | `M4/20260824T111439Z-*-space-contract-r2/` 覆盖四足/双足空间合同和 PhysX，但需绑定最终 clean revision 与 post-close exit。 |
| M5.1 qpth | 通过（范围受限） | `M5/20260824T105302Z-qpth-contract/` 覆盖 CPU/CUDA toy qpth forward/backward、KKT、残差与确定性；不等于历史 RAMBO QP 对比。 |
| M5.2 RAMBO 5.1 QP 数值比较 | **已批准延期** | 所需旧端 snapshot 未保存；本轮不以其阻断后续自动化工作。 |
| M6 四足 | workload 技术证据成立；sealed rerun 待执行 | 405/18、16 step 与 production walking 3000 step / 0 terminal：`M6/20260824T112247Z-*`、`M6/20260824T113722Z-quadruped-policy-longrun-n1-schedulefixed/`。`M6/20260824T134233Z-*` 与 `134254Z-*` 是全支撑零 action 100/1000 的独立静止诊断，**不是** production walking contact schedule；后者的旧 1000-step 尝试约在 152 step terminal，保留为失败。首个真实 policy→QP transition：`M6/20260824T133438Z-quadruped-first-policy-qp-transition/`。旧 Kit artifacts 无 post-close exit sidecar。 |
| M7 四足 RGB | workload 技术通过；sealed clean provenance 待重跑 | `M7/20260824T131656Z-quadruped-3000-rgb-thirdperson-final/` 有 3000 action、15,000 physics ticks、375 前视 RGB 与第三人称技术证据，但在 dirty source 上运行，旧 manifest 缺完整 exact-pinned provenance 和 post-close exit。 |
| M8 Button 非交互 | 通过 | `M8/20260824T112955Z-teleop-loco-manip/` 覆盖 deterministic Button press/hold/release。 |
| M8 GUI 真实键盘 | 待人工 | 已实现可审计 artifact/离线验证器；仍需真实操作者在 `--viz kit` 下完成物理键盘操作。合成输入不计通过。 |
| M9 双足 | workload 技术通过；sealed clean provenance 待重跑 | 旧 3000/RGB artifact `M9/20260824T132054Z-biped-3000-rgb-thirdperson-final/` 发生在 dirty source。独立性 runtime `M9/20260824T134918Z-biped-front-leg-independence-physx/` 证明 FL/FR 受不同命令且 QP 六个逻辑 contact slot 对应 override；该旧 schema 无 clean-source/post-close exit evidence，需 schema-v2 sealed rerun。 |
| M10 Button recorder | workload 技术子门禁通过；sealed rerun 待执行 | `M10/20260824T122000Z-button-episode-3000-provenance-v2/`：3000 action/observation/post-state、375 RGB、381 checksums 与离线 acceptance；旧 artifact 无 post-close exit sidecar。 |
| M11 native freeze | 阶段性可复现；最终冻结待自动化 chain | pinned `.in`/lock、metadata verifier 和 manifests 已有；最终 commit/host freeze 应等待 M3、M4 provenance 和 M7/M9 clean artifacts。 |
| M11 Docker | **已批准延期** | 当前 host 无 Docker Engine/NVIDIA Container Toolkit；本轮不 build/run/push。 |

## 关键技术说明

### M2.1 的 CUDA/Ada 结果

artifact 记录 Torch `2.10.0+cu128`、CUDA 12.8、RTX 4090 capability `(8,9)`、
成功 CUDA 算术和 `sm_86` compiled arch。PyTorch 的
[`get_arch_list`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html)
返回 wheel 编译时包含的架构，而不是设备实时 capability；NVIDIA 的
[Ada compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.0/ada-compatibility-guide/index.html)
说明 Ampere binary 可向前兼容 Ada。因此运行兼容有证据，但计划中的 native
`sm_89` 断言仍然没有通过。

### 四元数和 QP 边界

目标 Isaac Lab 3 public simulator-facing 数据在本迁移涉及的 articulation/camera
边界为 **XYZW**。RAMBO 的历史 QP/checkpoint math 保留 WXYZ，只允许经
`xyzw_to_wxyz` / `wxyz_to_xyzw` 做显式、局部转换；不得对 405/435D observation、
18D action 或 checkpoint 作全局重排。

### M7/M9 provenance 修复

新的 `--third-person-diagnostic final` 会在创建 `AppLauncher` **之前**捕获并要求：
RAMBO Git HEAD、porcelain clean status、完整 Python argv、Isaac Lab exact tag/HEAD/
package version、Isaac Sim package version、Torch/CUDA/GPU。manifest 写入时再次检查
clean source。下一批 M7/M9 3000-step artifact 只有满足此合同才可作为 final clean
evidence。

## 尚需完成的工作

1. 重跑 M7 quadruped 和 M9 biped 的 clean-source final PhysX diagnostics，并绑定
   M4 provenance；M3 inventory/quaternion scan 已完成。
2. 由真实操作者完成 M2.3 的 Direct Cartpole GUI 检查和 M8 physical-keyboard
   teleoperation；不得使用 xdotool、pyautogui、重放或注入事件。
3. M5.2 历史 snapshot 和 M11 Docker 已批准延期，保留其现有证据与 runbook，待后续
   具备输入/主机条件时恢复。
4. 在上述自动化和人工 GUI 条件满足后再生成 M11 final native freeze。由于本地分支
   尚无 remote tracking，push 仍须具备 GitHub 凭据/授权。

所有 artifact 保留历史失败和非验收尝试，以便审计；它们不会被删除或改写为成功。
