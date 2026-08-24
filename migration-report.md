# RAMBO Isaac Sim 6 / Isaac Lab 3 迁移报告

生成时间：2026-08-24（UTC）
运行验收代码：`b0bc68eb32c99ba326bde9a447a898951e774b71`

## 结论

自动化的原生 PhysX 迁移门禁已通过：固定的 Isaac Sim `6.0.1.0`、Isaac Lab
`v3.0.0-beta2.patch1` / `ffff603eafc6b74264a5261cc0183d6a65390d78`、Python
3.12.3、Torch `2.10.0+cu128` 在 RTX 4090 / driver 595.71.05 上完成了官方门禁、
四足与双足策略、RGB、Button 交互和 30 秒 recorder 验收。所有已接受的 RAMBO
runtime artifact 都在运行前后记录了 `PhysxCfg`、
`isaaclab_physx.physics.physx_manager.PhysxManager` 和
`use_newton_actuators=false`；没有 RAMBO runtime 选择或执行 Newton。

裸 Newton 相关分发包是精确官方依赖图允许的传递节点，不是失败条件；安装输入和
canonical setup 均不请求 Isaac Lab Newton optional extra。该结论不能用包存在与否
替代实际 backend 断言。

尚有两项不能伪造为通过的外部工作：真实操作者的 GUI 物理键盘检查，以及在具备
Docker Engine/NVIDIA Container Toolkit 的独立主机上实际 build/run 容器验收。

## 已接受的原生证据

| 门禁 | 状态 | 主要 artifact / 结果 |
| --- | --- | --- |
| M0–M1 | 通过 | host 清理与精确安装：`/workspace/migration-output/isaac60/M0/20260824T100222Z-host-and-cleanup/`、`M1/` |
| M2 官方门禁 | 通过 | Cartpole、Go2、RTX RGB、Kit GUI 和 clean restart：`/workspace/migration-output/isaac60/M2/20260824T103300Z-cartpole-physx/` 至 `M2/20260824T104800Z-camera-physx-warm-restart/` |
| M4 API/空间 contract | 通过 | 四足/双足各 5 control steps、PhysX 前后断言：`M4/20260824T111439Z-quadruped-space-contract-r2/`、`M4/20260824T111439Z-biped-space-contract-r2/` |
| M5 qpth | 通过 | CPU/CUDA forward/backward、KKT、残差、确定性：`M5/20260824T105302Z-qpth-contract/` |
| M6 四足 3000 步 | 通过 | 405/18、3000 steps、0 terminal、15,000 physics ticks：`M6/20260824T113722Z-quadruped-policy-longrun-n1-schedulefixed/`；summary SHA-256 `fc748621e4b2746d2b25ab8369fb0c5d61ca3e64b267ba165122cf0c2e227251` |
| M7 四足 RGB | 通过 | 3000 steps、375 × 640×480 RGB、12.5 Hz：`M7/20260824T115012Z-quadruped-3000-rgb-stagingfixed/`；summary SHA-256 `ba271c314973e2fec34e4d23a6f2801b68707b3eb42a0529b42ab7f7504114af` |
| M8 Button 非交互 smoke | 通过 | 15.8 mm 按压、至少 5 步、1.6 mm 回弹、base 0.520 m、joint 0.976 rad：`M8/20260824T112955Z-teleop-loco-manip/` |
| M9 双足 3000 步与 RGB | 通过 | 435/18、3000 steps、0 terminal、19.6 s phase；`M9/20260824T115916Z-biped-policy-longrun-n1/`；RGB：`M9/20260824T120253Z-biped-3000-rgb/` |
| M10 Button recorder | 通过 | 3000 action/observation/post-state、375 RGB、381 checksums、离线 `--require-acceptance` 通过：`M10/20260824T122000Z-button-episode-3000-provenance-v2/` |

M10 最终验收的按钮位移为 19.99354 mm，连续按下 17 步；第 765 action step 首次成功，
第 795 step 后回弹，成功后最小位移约 `2.98e-7 m`；base/FL foot 最大位移分别为
0.58812 m/0.83201 m。GPU 分配增长为 0，RSS 斜率为 0.01317 MiB/100 steps。其
`summary.json` SHA-256 为
`5305be4b3a6e0038b8381d80eb49c33503fa4d1535253715b96025d852e8d2fd`，
`manifest.json` SHA-256 为
`74317adadc99bacfe8e6ccfa83b1f8dd55b890c4be7ea1098be02aadf9e9ca00`。

## 完整性说明

M10 最终 artifact 自带 `checksums.sha256`，381 个文件均已复核。早期已接受的
M7 最终 RGB、M8 teleop 和 M9 最终 RGB artifact 没有持久 checksum manifest，故
不将它们说成具有 manifest；其独立摘要/图像哈希如下，供追溯：

- M7 summary：`ba271c314973e2fec34e4d23a6f2801b68707b3eb42a0529b42ab7f7504114af`；contact sheet：`00a7d054461db1153b4cff8c1981d15248636510dde1c7c93935ab461df755db`。
- M8 summary：`9650ca85d8b6fa1f7d53f535a975cef7454daed7883cf88ab54d87196a7b86ff`；teleop log：`2e591e3f10001d158507c2ea922d30260f65a761f4ab4a1717a55ca6ba2a566c`。
- M9 RGB summary：`a42d99f6fa50aac78ed55aa9d35981ac5b906922f539ee142a909eee1647ab20`；contact sheet：`c6a68ff68c8b158dba04c321e820e34700ebc212c10cbac2c500f891985cb5df`。

保留而不计为验收的诊断 artifact 包括 M6 的旧 10 秒 contact schedule 耗尽、M7
的旧 RGB RSS 斜率超阈值，以及 M10 的 schema-v1 recorder。它们分别由 31 秒
schedule、可复用 CPU staging buffer 和 schema-v2 provenance/离线 validator
取代；未删除诊断证据。

## 冻结与容器边界

`requirements/isaacsim60.in` 表示直接意图；`requirements/isaacsim60.lock`
包含 10 个直接 pin 与 242 个非 editable 精确 wheel pin。setup 先执行官方固定分段路径，再以
`--no-deps` 消费 lock，不引入第二次依赖解析；verifier 会逐一比对 lock，验证
Isaac Lab editable `direct_url` 来自固定 checkout，并严格检查已审阅的 8 项
`pip check` metadata 差异。

Dockerfile 是 fail-closed 的**延后规范**：它在没有 GPU 的 build 中只运行
`--docker-build-metadata-only`，不得将它解释为 CUDA/RTX/PhysX 通过。未来容器
必须在 `docker run --gpus all` 中先运行默认 verifier，再用 process-scoped
`OMNI_KIT_ACCEPT_EULA=Y` 执行独立 PhysX artifact gate。当前 host 没有 Docker
Engine 或 NVIDIA Container Toolkit，因此没有执行 Docker build/run/push。

## 仍需人工完成的 GUI 检查

唯一未认证的原生行为是 GUI 中的真实物理键盘 teleoperation。请按 README 的
`--viz kit` 命令启动、点击 viewport，并实际操作 arrows/numpad、`Z/X`、
`W/S`、`A/D`、`R/F`、`L`、`C`。不得用合成输入 trace 替代该检查或将其标记为通过。
