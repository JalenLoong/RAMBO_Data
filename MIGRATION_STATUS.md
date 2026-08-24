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
| M2 官方 PhysX/RTX/Kit | 非 GUI sealed 门禁通过；M2 整体仍正式未完成 | M2.2 `M2/20260824T142949Z-official-cartpole-direct-16-physx-sealed/`（官方 Direct Cartpole、16 env/16 step）与 M2.4 warm-up/restart `M2/20260824T143007Z-*`、`143029Z-*`（官方 Go2、各 1000 ticks、RTX RGB）均为 PhysX 前后 FQN、checksum-covered `process_exit.json=0`。M2.1 仍不满足原字面 `sm_89` 断言；M2.3 GUI 按本轮范围延期。 |
| M3 API inventory | 完成（clean-source 扫描） | `M3/20260824T142802Z-api-inventory-clean/` 绑定 `e449b1d`，源码 porcelain clean，quaternion scan 仅有两处已审计的 XYZW `[0,0,0,1]` identity 候选；checksums 已复核。 |
| M4 API/空间 contract | 完成（sealed） | 四足 `M4/20260824T143107Z-quadruped-space-contract-sealed/` 与双足 `M4/20260824T143117Z-biped-space-contract-sealed/` 均为 1-env/5-step、显式 PhysX、前后 manager FQN 与 exit 0。 |
| M5.1 qpth contract | 通过（范围受限） | CPU/CUDA toy-qpth forward/backward、KKT、残差与确定性：`M5/20260824T105302Z-qpth-contract/`；这不是旧版 RAMBO QP snapshot 对比 |
| M5.2 RAMBO 5.1 QP 数值对比 | **已批准延期** | 计划要求的旧 baseline mass/Jacobian/contact/desired-force/constraint 及 solution/torque/contact/residual snapshot 未保存，当前主机不能诚实重建或以新端数据代替；按当前用户范围，它不阻断本轮后续自动化工作。 |
| M6 四足 | 完成（sealed） | `M6/20260824T143140Z-quadruped-policy-3000-first-transition-sealed/` 完成 3000 步真实 policy→QP rollout 与首转换；`151140Z-*` 补齐 current-source 的 16-env/16-step（256 env-steps、405/18）sealed gate；`143407Z-*`/`143422Z-*` 是额外 100/1000-step 全支撑零动作诊断，明确不替代 production walking。所有列出的 gate 均 exit 0/PhysX 前后 FQN。 |
| M7 四足 RGB 3000 steps | 完成（sealed clean-source） | `M7/20260824T143537Z-quadruped-3000-rgb-thirdperson-final-sealed/` 绑定 clean source `358daa5`：3000 policy steps、15,000 physics ticks、375 base-mounted 前视 RGB、终态独立 RGBD、checksum 与 exit 0 均通过。前视流证明 cadence/scene；同一封存 rollout 的 third-person RGBD 证明 robot + scene，绝不伪称机器人在每张前视帧内可见。 |
| M8 Button non-interactive | 完成（由 M10 sealed recorder 覆盖） | 30 秒 deterministic Button 运行在 `M10/20260824T144344Z-button-episode-3000-sealed/` 中完成并通过 acceptance。 |
| M8 GUI 真实键盘 | **本轮已延期** | 仅接受真实操作者在 `--viz kit` 下的物理键盘；不使用 xdotool、pyautogui、重放或任何注入输入。 |
| M9 双足 | 完成（sealed clean-source） | `M9/20260824T143926Z-biped-3000-rgb-thirdperson-final-sealed/` 完成 3000/15,000/375/RGBD；`151210Z-*` 补齐 current-source 的 16-env/16-step（256 env-steps、435/18）sealed gate；`144321Z-biped-front-leg-independence-physx-schema-v2-sealed/` 在 clean source 上完成 FL/FR 两步独立性。前视流/third-person RGBD 的 visibility 解释与 M7 相同；所有列出的 gate 均 PhysX 前后 FQN、checksum 与 exit 0。 |
| M10 30 秒 Button recorder | 完成（sealed） | `M10/20260824T144344Z-button-episode-3000-sealed/`：3000 action/observation/post-state、375 RGB、Button success/rebound 与 motion acceptance、382 个封存文件、exit 0 均通过。 |
| M11 native freeze | 完成（本轮 current scope） | `M11/20260824T151639Z-native-freeze-final-current-scope-v3/` 已绑定干净 commit、gpu-required verifier、242-wheel lock、无 optional Newton extras、20 项 canonical source inputs 与接受工件索引的 checksum。它取代旧 6-input freeze；M2.3/M8 GUI、M5.2 和 Docker 不在本轮 scope。 |
| M11 Docker | **已批准延期** | 当前 host 无 Docker Engine/NVIDIA Container Toolkit；本轮不 build/run/push image |

## 最终 M10 摘要

运行验收源码 commit：`358daa52260841cdfa8263adeee930144379c5fe`。最终 artifact 包含
3000 actions、3000 observations、3000 post states、375 RGB，terminal 为 0；按压
19.99354 mm 连续 17 steps，成功后回弹符合 ≤2 mm，base/FL foot 均超过 0.05 m。
前后 manager 均为 PhysX，离线 `--require-acceptance` 通过，381 文件 checksums
加上 post-close exit sidecar 后共 382 个封存文件全部通过：
`/workspace/migration-output/isaac60/M10/20260824T144344Z-button-episode-3000-sealed/`。

## 补充的 M2 可复现证据

M2.1 的非模拟 CUDA artifact
`/workspace/migration-output/isaac60/M2/20260824T135749Z-cuda-ada-compatible-contract/`
已复核：Torch `2.10.0+cu128` / CUDA 12.8 在 RTX 4090 capability `(8,9)` 上完成
同步 CUDA 算术。该精确 wheel 的 `get_arch_list()` 没有 native `sm_89`，但有
`sm_86`；artifact 同时保存这两个事实和成功算术，按 NVIDIA 的 Ampere→Ada
forward-compatibility 解释为**运行兼容**，绝不把 native `sm_89` 伪报为存在。
因此它不能满足原计划字面 `assert "sm_89" in torch.cuda.get_arch_list()`，除非验收
标准被正式修订。

M2.2 的 sealed 官方任务 wrapper artifact
`/workspace/migration-output/isaac60/M2/20260824T142949Z-official-cartpole-direct-16-physx-sealed/`
已复核：官方 `Isaac-Cartpole-Direct-v0`、16 environments、16 steps、显式
`PhysxCfg`、前后 `PhysxManager`、`use_newton_actuators=false`、checksum 与 post-close
exit 0 均通过。固定 tag 的 `zero_agent.py` 被保留为非验收的上游限制：
`135150Z-*` 因它选到无 gymnasium 的宿主 venv 失败；`135239Z-*` 在正确 venv 中
确认 `--viz none` 无限循环后被人工中止，只有 console log，不能算成功。

新建的 `scripts/rambo60/official_go2_smoke.py` 不导入 RAMBO，只使用官方
`UNITREE_GO2_CFG`。在 RTX 4090 上以显式 `PhysxCfg` 和
`use_newton_actuators=false` 分别运行 1000 physics ticks 两次：

- warm-up：`/workspace/migration-output/isaac60/M2/20260824T143007Z-official-go2-rgb-1000-warmup-sealed/`；
- clean restart：`/workspace/migration-output/isaac60/M2/20260824T143029Z-official-go2-rgb-1000-restart-sealed/`。

两份 `summary.json` 都记录单个 Go2、640×480 Isaac RTX RGB、11,000 次有限 state
检查，且运行前后 manager 均为
`isaaclab_physx.physics.physx_manager.PhysxManager`。每个目录的
`go2_rgb.png`、`summary.json` 与 `process_exit.json` 均由各自的 `checksums.sha256`
覆盖并已复核。

旧原始扫描仍保留在 `/workspace/migration_logs/quaternion-scan.txt` 和
`/workspace/migration_logs/api-inventory.txt`，但它们已由 clean-source M3 artifact
`/workspace/migration-output/isaac60/M3/20260824T142802Z-api-inventory-clean/` 取代。
新扫描中的两处 identity 都是已审计的 **XYZW** `[0,0,0,1]`，不得误标为 WXYZ。

## 当前剩余项

1. M2.1 仍未满足原计划的 native `sm_89` 字面断言；现有 artifact 只证明 CUDA/Ada 运行兼容，若要计为完整 M2.1，必须先正式修订该验收标准。
2. M2.3 GUI 与 M8 真实键盘 GUI 均按当前用户范围延期，后续只能由真实操作者在 `--viz kit` 完成，绝不以日志、合成或注入输入替代。
3. M5.2 历史 Isaac Sim 5.1 QP snapshot 比较和 M11 Docker 已批准延期；它们保留为后续工作，不写成已完成。
