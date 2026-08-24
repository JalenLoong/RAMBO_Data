# RAMBO Isaac Sim 6 / Isaac Lab 3 最终矩阵

这是 M11 native freeze 使用的**唯一 canonical final matrix**。它汇总当前
当前 native scope 的 gate 状态；详尽命令、版本、工件和风险仍以
[`migration-report.md`](migration-report.md) 与
[`MIGRATION_STATUS.md`](MIGRATION_STATUS.md) 为准。任何 current-scope M11 native
freeze 都必须同时 checksum 本文件、这两个文档和 runtime launch/sealing wrapper，
不能只冻结依赖 lock。Docker 定义继续延期，不纳入这次 native freeze 的闭合范围。

| Gate | 当前结论 | Canonical evidence / 限制 |
| --- | --- | --- |
| M0–M1 | 完成 | 精确 target checkout、venv、requirements lock 与 host/checkpoint manifests 已建立。 |
| M2.1 CUDA | **未按原字面标准通过** | Ada CUDA runtime contract 成功，但固定 Torch wheel 不列出 `sm_89`；不可写成完整通过。 |
| M2.2 官方 Cartpole | sealed 通过 | `M2/20260824T142949Z-official-cartpole-direct-16-physx-sealed/`；有限 16 env/16 step、显式 PhysX、post-close exit 0。 |
| M2.3 Kit GUI | sealed + 人工声明通过 | `M2/20260824T235256Z-cartpole-direct-kit/` 与 `*-attestation/`：45.04 秒、1083 steps、两次 live RTX 4090/`rtx`/`omni.hydra.rtx` sample、显式 PhysX、post-close exit 0 和真人 TTY 声明。 |
| M2.4 官方 Go2 + RGB | sealed 通过 | warm-up/restart `M2/20260824T143007Z-*`、`M2/20260824T143029Z-*`；均为 RTX RGB、PhysX FQN 与 exit 0。 |
| M3 | 完成 | clean-source inventory `M3/20260824T142802Z-api-inventory-clean/`。 |
| M4 | sealed 通过 | 四足/双足 space contract：`M4/20260824T143107Z-*`、`M4/20260824T143117Z-*`。 |
| M5.1 | 通过（范围受限） | qpth contract 仅证明 toy qpth，不等于历史 RAMBO QP 数值对比。 |
| M5.2 | **已延期** | Isaac Sim 5.1 历史 QP snapshot 缺失；不以新端数据伪造比较。 |
| M6 | sealed 通过 | 3000-step/first-transition `M6/20260824T143140Z-*` 与 current-source 16-env/16-step `M6/20260824T151140Z-*`；零动作诊断仅补充，不能替代 production walking。 |
| M7 | sealed clean-source 通过 | `M7/20260824T143537Z-quadruped-3000-rgb-thirdperson-final-sealed/`：生产前视流证明 sensor/cadence/scene，绑定同 rollout final third-person RGBD 证明 robot + scene。 |
| M8 non-interactive | sealed 通过 | M10 Button recorder 覆盖 30 秒 deterministic episode。 |
| M8 GUI 真实键盘 | **待执行** | `run_gui_keyboard_artifact.sh` 仅接受带 EULA 前缀且有 `DISPLAY` 的 `--viz kit`；操作者须用本人实体键盘经当前 Selkies WebRTC 会话实时输入。Kit 关闭后写入 exit sidecar、要求真人 TTY 精确声明再离线验收；禁止 xdotool、pyautogui、脚本、重放、自动化和输入注入。 |
| M9 | sealed clean-source 通过 | 3000-step RGB、current-source 16-env/16-step 与 FL/FR independent artifacts：`M9/20260824T143926Z-*`、`M9/20260824T151210Z-*`、`M9/20260824T144321Z-*`；RGB visibility 采用 M7 的组合证据。 |
| M10 | sealed 通过 | `M10/20260824T144344Z-button-episode-3000-sealed/`：3000 action/observation/post-state、375 RGB 与 acceptance。 |
| M11 native freeze | 先前 current-scope freeze 完成；最终刷新待 M8 | `M11/20260824T234827Z-native-freeze-viewport-rtx-v6/` 生成于 M2.3 人工验收之前；待 M8 GUI 完成后一次性刷新并绑定两项 GUI 证据。 |
| M11 Docker | **已延期** | 当前 host 没有 Docker Engine/NVIDIA Container Toolkit；不 build/run/push。 |

因此，本迁移的完整 adaptation 仍未完成：M2.1 的字面标准、M8 GUI、M5.2
历史 snapshot 以及 Docker 都没有被重写为通过。刷新 M11 native freeze 只会绑定当前
source-clean non-GUI evidence，绝不改变这些 gate 的状态。
