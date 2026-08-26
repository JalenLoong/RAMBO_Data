# RAMBO Isaac Sim 5.1 / Isaac Lab 2.3.2 Blackwell 迁移实现与复现手册

> **历史预切换记录。** 本文中的旧根路径、旧 runtime 与旧 checkout 仅供审计或复现历史证据，不能作为当前 workspace 命令。当前 canonical 路径以根 `README.workspace.md`、`workspace.env` 与 `workspace.lock.yaml` 为准。

## 0. 文档定位

本文记录 RAMBO 从旧 Isaac Sim 4.5 栈迁移到 Isaac Sim 5.1 / Isaac Lab 2.3.2 的**实际实现过程、失败诊断、最终代码契约和验收结果**。目标是在另一台 NVIDIA Blackwell 宿主机上，从干净 shell 开始，尽快复现以下三个已有闭环：

1. quadruped 原 checkpoint 回放、QP/执行器运行、30 秒稳定性和 RGB；
2. biped 原 checkpoint 回放、独立双前肢拓扑、30 秒稳定性和 RGB；
3. quadruped 在同一 Isaac Sim 5.1 会话中完成“行走 + FL 前腿按实体弹簧按钮”的 loco-manip 遥操。

本文不是新的需求计划。验收边界仍以 [rambo-isaacsim5-adaptation-plan.md](rambo-isaacsim5-adaptation-plan.md) 为准；本文回答“当时具体怎样实现、为什么这样实现、在新机器上按什么顺序复现”。

记录基线：

| 项目 | 值 |
| --- | --- |
| 实现分支 | `new_IsaacSim_IsaacLab` |
| 双模式主迁移提交 | `dcf5e323a832bb785203c655bdc4b4d904588ef8` |
| 移除 vendored Isaac Lab | `62e0a51dbef35fc8cbdb5de75ecb333d349fe59a` |
| biped 相机修复 | `12a48cdb2a900cbe5b6140bae2d1d23e0ee767d4` |
| loco-manip 主实现 | `2f46ffb0380c150a9df82ef36e65873a610191f2` |
| 真实键盘链路修复 | `1034bbf7daffc53d0ddccb67a08bdd1c8cd04dc8` |
| 最终验证日期 | 2026-08-22 |
| 验证 GPU | NVIDIA GeForce RTX 5080，16 GB，compute capability 12.0 |
| 验证驱动 | 580.159.03 |

> 重要：当前 smoke gate 明确要求 compute capability `(12, 0)` 和 Torch 架构 `sm_120`，即 RTX 50 系 Blackwell。若新宿主机是数据中心 Blackwell 且报告其他 capability（例如 `sm_100`），不要直接删除检查；应把它视为一个新的目标栈验证工作。

## 1. 已完成范围与未完成范围

### 1.1 已完成

| Task ID | 模式 | observation/action | checkpoint | 状态 |
| --- | --- | ---: | --- | --- |
| `Isaac-RAMBO-Quadruped-Go2-v0` | quadruped，单前肢 EE 语义 | 405 / 18 | `model_2000.pt` | 3000 步 + 375 RGB 通过 |
| `Isaac-RAMBO-Biped-Go2-v0` | biped，双前肢 EE 语义 | 435 / 18 | `model_4000.pt` | 3000 步 + 375 RGB 通过 |
| `Isaac-RAMBO-Quadruped-Button-Go2-v0` | quadruped loco-manip | 405 / 18 | 同一 `model_2000.pt` | 行走、FL 按压、回弹、GUI 键盘通过 |

两种基础模式都是一等公民。biped 不是 quadruped 的参数开关：它保留独立的双前肢位置/力命令、独立 QP 接触覆盖、435 维观测和原 checkpoint。

Button task 不改变 policy 网络。base 速度、FL Cartesian target 和按钮状态都位于 task/teleop 层；policy 仍接收 405 维观测并输出 18 维动作。

### 1.2 尚未迁移

- WAM/LingBot-VA、Manipulator task；
- 15 维高层动作；
- 三相机数据 contract；
- trajectory recorder、遥操数据集生成和 WAM trajectory；
- Button/Manipulator 训练与 policy retraining；
- Isaac Sim 4.5 与 5.1 数值 parity；
- 旧的 `isaaclab.sh`、`scripts/environments/`、`scripts/reinforcement_learning/` 工作流。

这些内容保存在 legacy 快照中，但不是当前分支的可运行承诺。

## 2. 最终参考宿主机

### 2.1 通过验证的版本

| 组件 | 实际版本 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS，kernel 6.8.0-124-generic，glibc 2.35 |
| Python | 3.11.15 |
| NVIDIA driver | 580.159.03；最低目标为 production branch 580.65.06 |
| GPU | GeForce RTX 5080，16303 MiB，capability `(12, 0)` |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| Torch CUDA runtime | 12.8 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.2 |
| qpth | 0.0.18 |
| NumPy | 1.26.0（contract 是 `<2`） |
| CVXPY / OSQP / ECOS | 1.5.4 / 0.6.7.post3 / 2.0.14 |
| setuptools / flatdict | 78.1.0 / 4.0.1 |

Torch 在验证机上报告：

```text
cuda_available True
gpu NVIDIA GeForce RTX 5080
capability (12, 0)
arch_list ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
```

### 2.2 容量和图形要求

- venv 实际占用约 19 GB；建议至少预留 35 GB，以容纳 pip cache、Kit extension cache 和验证输出。
- 一次 quadruped 3000 步 RGB 输出约 31 MB；一次 biped 输出约 44 MB。
- GUI 验证需要可用的 X11/桌面会话和 Vulkan。
- headless camera 仍需要正常的 NVIDIA RTX/Vulkan 栈；“没有窗口”不等于不需要图形驱动。
- 当前按至少 16 GB 显存验证，验收固定为单环境。

## 3. Git 基线和旧工作保护

迁移前的 repository 不是干净上游：旧 vendored extension 自报版本为 `0.36.6`，最接近的 Isaac Lab upstream 是 `09590912792d...`，同时包含 RAMBO 私有改动。因此不能把它误认为 Isaac Lab 2.0.2，也不能用 upstream 版本号替代真实基线。

旧工作已冻结为：

```text
branch: codex/legacy-isaacsim-4.5-local
commit: 3c83e07325d1b11e03f80aecf8f35a353e5326af
tag:    legacy-isaacsim-4.5-local
```

tag 是 annotated tag；peeled commit 是 `3c83e073...`。其中两个历史提交分别为：

```text
dd746a8 docs: add Isaac Sim 5 adaptation plan
3c83e07 wip: preserve Isaac Sim 4.5 RAMBO experiments
```

新宿主机应直接使用 `new_IsaacSim_IsaacLab`，不要把 legacy 分支里的 vendored `source/isaaclab*` 合回来：

```bash
cd /workspace/rambo
git switch new_IsaacSim_IsaacLab
git log -5 --oneline
git status --short --branch
```

实现基线应至少包含以下提交序列：

| 顺序 | commit | 作用 | 规模 |
| ---: | --- | --- | --- |
| 1 | `dcf5e32` | 双模式外部 extension、runner、validator、测试和环境脚本 | 64 files，+5314/-317 |
| 2 | `62e0a51` | 删除 vendored Isaac Lab/Lab assets/tasks/rl/mimic | 427 files，主要为 -89988 |
| 3 | `12a48cd` | 修正 upright biped 的 RGB 相机位置和旋转 | 4 files |
| 4 | `2f46ffb` | Button task 和 quadruped loco-manip teleop | 9 files，+803/-19 |
| 5 | `1034bbf` | Isaac Sim 5.1 字符串键事件、Space 冲突、遥测和严格 done 检查 | 4 files，+84/-13 |

关键文件索引：

| 关注点 | 权威文件 |
| --- | --- |
| 一键环境安装 | `scripts/setup_isaacsim51.sh` |
| 依赖意图/完整 freeze | `requirements/isaacsim51.in`、`requirements/isaacsim51.lock` |
| CUDA/NVRTC/pycache 启动边界 | `scripts/rambo/run.sh`、`source/rambo/rambo/torch_runtime.py` |
| official Go2 与 camera preflight | `scripts/rambo/smoke.py` |
| checkpoint playback | `scripts/rambo/play.py` |
| 3000 步和 RGB 验证器 | `scripts/rambo/validate.py`、`source/rambo/rambo/validation/rollout.py` |
| checkpoint allow-list/恢复 | `source/rambo/rambo/validation/checkpoints.py` |
| task 注册 | `source/rambo/rambo/__init__.py`、两个 mode 的 `__init__.py` |
| quadruped/biped QP 环境 | `source/rambo/rambo/tasks/direct/rambo_{quadruped,biped}/qp_env.py` |
| Go2 名称映射 | `source/rambo/rambo/utils/articulation.py` |
| delayed actuator | `source/rambo/rambo/actuators/` |
| 精确前视 RGB 相机 | `source/rambo/rambo/tasks/common/camera.py` |
| CRL2 Gymnasium adapter | `source/rambo/rambo/rl/crl2_vec_env.py` |
| 物理按钮 | `source/rambo/rambo/tasks/direct/rambo_quadruped/button_env.py` |
| loco-manip 遥操/联合 smoke | `scripts/rambo/teleop_loco_manip.py` |
| 自动测试 | `tests/rambo/` |

checkpoint、datasets、运行日志、RGB 帧和 contact sheet 不进入 Git。

## 4. 新 Blackwell 宿主机最快复现路径

以下步骤是推荐执行顺序。每一级 gate 都通过后再进入下一级；这样可以区分驱动、官方 Isaac 栈、RAMBO extension、checkpoint 和相机问题。

### 4.1 准备代码和路径

```bash
export RAMBO_REPO=/workspace/rambo
export RAMBO_VENV=/workspace/venvs/rambo51

cd "$RAMBO_REPO"
git switch new_IsaacSim_IsaacLab
git status --short --branch
```

要求 Python 3.11 和 venv module 已由宿主机管理员安装。若 `python3.11` 不存在，先使用适合该发行版的系统包、deadsnakes 或 pyenv 提供 3.11；不要让 setup script 落回 Python 3.10/3.12。

确认驱动先于安装 Python 包：

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```

预期为可见的 NVIDIA GPU、驱动至少 580.65.06。当前严格目标应报告 compute capability 12.0。

### 4.2 准备 checkpoint

checkpoint 不在 Git。复制两个已验证本地文件到新宿主机任意只读或受控路径，然后设置：

```bash
export RAMBO_QUAD_CKPT=/workspace/rambo-checkpoints/quadruped/model_2000.pt
export RAMBO_BIPED_CKPT=/workspace/rambo-checkpoints/biped/model_4000.pt

sha256sum "$RAMBO_QUAD_CKPT" "$RAMBO_BIPED_CKPT"
```

必须严格得到：

```text
1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706  model_2000.pt
c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6  model_4000.pt
```

不要为了加载不同 hash 而关闭 allow-list。当前 loader 只有在 SHA256 完全匹配后，才会对这个可信本地历史 checkpoint 使用 `torch.load(..., map_location="cpu", weights_only=False)`。

### 4.3 创建独立 Python 3.11 环境

```bash
cd "$RAMBO_REPO"
RAMBO_VENV="$RAMBO_VENV" bash scripts/setup_isaacsim51.sh
source "$RAMBO_VENV/bin/activate"
```

安装脚本会按以下顺序执行：

1. 创建或复用 Python 3.11 venv；拒绝其他 Python minor version。
2. 设置 `PYTHONNOUSERSITE=1` 并清除 `PYTHONPATH`。
3. 预先固定 `setuptools<81`、`flatdict==4.0.1`、`wheel==0.45.1`。
4. 独立安装 `numpy<2`、Torch 2.7.0 cu128、torchvision 0.22.0 cu128。
5. 从 NVIDIA/PyTorch index 安装官方 `isaaclab[isaacsim,all]==2.3.2`。
6. 从已安装 wheel 的 `isaaclab/source/isaaclab_assets` 安装官方 bundled assets。
7. 固定 `packaging==23.0`、`osqp==0.6.7.post3`、CVXPY 1.5.4、ECOS 2.0.14、qpth 0.0.18。
8. editable 安装 `source/crl2` 和 `source/rambo`。
9. 执行 `pip check` 并打印 Python/Torch/GPU/assets 信息。

依赖意图在 [requirements/isaacsim51.in](requirements/isaacsim51.in)，已验证 freeze 在 [requirements/isaacsim51.lock](requirements/isaacsim51.lock)。setup script 以 `.in` 和显式安装事务为准；lock 用于审计和比对，不包含机器相关 editable 路径。

安装后立即检查：

```bash
"$RAMBO_VENV/bin/python" -m pip check
"$RAMBO_VENV/bin/python" -m pip freeze --all | \
  rg '^(isaaclab|isaacsim|torch|torchvision|qpth|cvxpy|osqp|numpy|setuptools|flatdict)=='
```

预期 `No broken requirements found.`。

### 4.4 首次 EULA 和纯官方栈 gate

先在交互 shell 中启动一次：

```bash
source "$RAMBO_VENV/bin/activate"
isaacsim
```

由操作者阅读并接受 NVIDIA EULA，确认 GUI 能正常显示，然后退出。自动化环境可以在已经合法接受 EULA 后使用 `OMNI_KIT_ACCEPT_EULA=YES`，但 repository 的 smoke script 不替操作者接受。

从 fresh shell 运行官方 Go2 gate。它不 import RAMBO，所以失败应先检查驱动/Isaac Lab/assets，而不是 QP task：

```bash
cd "$RAMBO_REPO"
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh \
  scripts/rambo/smoke.py --steps 100

RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh \
  scripts/rambo/smoke.py --headless --steps 100 --enable-camera \
  --camera-output /tmp/rambo-go2-smoke-rgb.npy
```

通过标记：

```text
CUDA_RUNTIME=...
GO2_SMOKE=...
GO2_SMOKE_SUCCESS
```

smoke 会检查：

- Python 3.11、Torch 2.7.0、CUDA 12.8；
- capability `(12, 0)` 和 `sm_120`；
- 实际提交并同步一个 CUDA tensor 运算；
- official `UNITREE_GO2_CFG`、ground、physics finite state；
- camera tensor 恰为 `(1, 480, 640, 3)`、`uint8` 且非全黑。

### 4.5 qpth 和静态回归 gate

始终通过 wrapper 执行，因为它会设置 PyTorch CUDA/NVRTC 动态库路径和独立 pycache：

```bash
cd "$RAMBO_REPO"
RAMBO_REQUIRE_CUDA_QPTH=1 RAMBO_VENV="$RAMBO_VENV" \
  scripts/rambo/run.sh -m pytest -q tests/rambo/test_qpth_contract.py

RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh -m pytest -q tests/rambo
```

当前结果：

```text
qpth CPU/CUDA: 2 passed in 1.64s
RAMBO suite:   26 passed
```

qpth 用同一个 2 变量严格凸 QP 在 CPU/CUDA 上分别检查 forward、backward、等式/不等式/stationarity residual，并重复两次要求 `1e-10` 确定性。最终没有修改 qpth 0.0.18 的源码；Blackwell 所需修复位于运行时动态库加载边界。

### 4.6 两种 task 的短验证

`validate.py` 要求 steps 是 8 的倍数，且拒绝覆盖非空 output directory。每次使用新目录：

```bash
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint "$RAMBO_QUAD_CKPT" \
  --steps 16 --enable_cameras \
  --output-dir /workspace/rambo-validation/quadruped-short-01

RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint "$RAMBO_BIPED_CKPT" \
  --steps 16 --enable_cameras \
  --output-dir /workspace/rambo-validation/biped-short-01
```

必须看到：

```text
SUMMARY_PATH=.../summary.json
VALIDATION_SUCCESS
```

并检查：

```bash
jq '{passed, task, checkpoint, validation_config, rollout, rgb}' \
  /workspace/rambo-validation/quadruped-short-01/summary.json
jq '{passed, task, checkpoint, validation_config, rollout, rgb}' \
  /workspace/rambo-validation/biped-short-01/summary.json
```

不要在标准 biped 验收中传 `--contact-phase-offset-s`。默认值 19.6 秒是 task/checkpoint 在目标运行时验证过的固定 gait phase，并且会写入 `summary.json`。

### 4.7 完整 30 秒双模式验收

```bash
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint "$RAMBO_QUAD_CKPT" \
  --steps 3000 --enable_cameras \
  --output-dir /workspace/rambo-validation/quadruped-3000-01

RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint "$RAMBO_BIPED_CKPT" \
  --steps 3000 --enable_cameras \
  --output-dir /workspace/rambo-validation/biped-3000-01
```

快速机器检查：

```bash
jq -e '.passed == true and .rollout.steps == 3000 and .rgb.frame_count == 375' \
  /workspace/rambo-validation/quadruped-3000-01/summary.json
jq -e '.passed == true and .rollout.steps == 3000 and .rgb.frame_count == 375' \
  /workspace/rambo-validation/biped-3000-01/summary.json

find /workspace/rambo-validation/quadruped-3000-01/rgb -type f | wc -l
find /workspace/rambo-validation/biped-3000-01/rgb -type f | wc -l
```

两个计数都应为 375。随后人工查看两个 `contact_sheet.png`，确认无黑帧、雪花、固定画面或机身内部遮挡。

### 4.8 GUI playback

```bash
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/play.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint "$RAMBO_QUAD_CKPT"

RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh scripts/rambo/play.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint "$RAMBO_BIPED_CKPT"
```

可用 `--steps N --deterministic` 做有限 GUI run；需要 task 前视 RGB 时再加 `--enable-rgb-camera`。

### 4.9 loco-manip headless gate 和 GUI 遥操

先运行同一环境内的可重复 gate：

```bash
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh \
  scripts/rambo/teleop_loco_manip.py --headless --smoke-loco-manip \
  --checkpoint "$RAMBO_QUAD_CKPT"
```

必须看到：

```text
[BUTTON] PRESSED: travel=...
[BUTTON] RELEASED
[SMOKE] forward displacement=... joint_delta=...
LOCO_MANIP_SMOKE_LOCO_MANIP_SUCCESS
```

再运行 GUI：

```bash
RAMBO_VENV="$RAMBO_VENV" scripts/rambo/run.sh \
  scripts/rambo/teleop_loco_manip.py \
  --checkpoint "$RAMBO_QUAD_CKPT" --telemetry-every 10
```

点击 viewport 后使用：

| 功能 | 键 |
| --- | --- |
| base 前后/左右 | 方向键或 numpad `8/2/4/6` |
| base yaw | `Z` / `X` |
| FL 前进/回撤 | `W` / `S` |
| FL 横向 | `A` / `D` |
| FL 上升/下降 | `R` / `F` |
| 停止并清除全部 command | `L` |
| 按钮回弹后清除 success latch | `C` |

不要使用 `Space`：Isaac Sim 5.1 将其保留给 timeline play/pause，即使 Python callback 收到按键，也可能先暂停仿真。

推荐人工顺序：

1. `Up` 向墙面行走，松开停止；
2. 等待 `[FL] ready`；
3. `R` 抬高 FL；
4. `W` 向前推进并压住按钮；
5. 看到 `[BUTTON] PRESSED` 和绿色状态球；
6. `S` 回撤，确认 `[BUTTON] RELEASED`、travel 回到 0；
7. `C` 清除绿色 latch，确认 `[BUTTON] CLEARED`。

## 5. 实际迁移实现过程

### 5.1 冻结 4.5 工作，避免边迁移边丢失实验

先把迁移文档和旧 4.5 实验分别提交到 `codex/legacy-isaacsim-4.5-local`，打 `legacy-isaacsim-4.5-local` tag，再回到 `new_IsaacSim_IsaacLab`。Button、Manipulator、teleop、trajectory recorder 和相机试验因此都有可恢复源，而 adaptation 分支只带当前里程碑核心。

验收结果：branch、annotated tag 和 peeled commit 均可解析；legacy Python 3.10 环境未被修改。

### 5.2 建立隔离的官方 pip 运行时

没有复用 Isaac Sim 4.5 Python 3.10/CUDA 12.4 环境。原因是旧 Torch 2.5.1 CUDA 12.4 wheel 不包含 RTX 5080 `sm_120`，qpth 一进入 CUDA 线性代数就无法成为有效验收基线。

最终依赖处理包含四个非显然点：

1. `setuptools<81`：避免 `pkg_resources` 被移除后旧 extension setup/import 失败；
2. `flatdict==4.0.1`：Isaac Lab pip 依赖图需要，但某些安装路径没有提前补齐；
3. official `isaaclab_assets`：它随 `isaaclab` wheel 带源码但不会自动成为已安装 distribution；setup script 从 wheel 内的准确路径安装它；
4. qpth/CVXPY：不能让最新 CVXPY 自动升级 Isaac Sim 固定的 `packaging==23.0` 和 `osqp==0.6.7.post3`，因此显式选 CVXPY 1.5.4/ECOS 2.0.14。

验收结果：`pip check` 无 broken requirements，RTX 5080 CUDA 运算和 official Go2 scene/camera 通过。

### 5.3 把 RAMBO 提取为独立 external extension

新增 `source/rambo`，结构为：

```text
source/rambo/
├── config/extension.toml
├── pyproject.toml
├── setup.py
└── rambo/
    ├── actuators/
    ├── rl/
    ├── tasks/common/
    ├── tasks/direct/rambo_quadruped/
    ├── tasks/direct/rambo_biped/
    ├── utils/
    └── validation/
```

RAMBO 包只在 `AppLauncher` 已启动、`omni.kit.app` 已存在时注册 task。普通 `import rambo` 只读取 metadata，不应该隐式启动 Kit 或弹 EULA。

RAMBO 自己提供 registry YAML resolver，因此不依赖官方示例集合 `isaaclab_tasks`；CRL2 wrapper 也不依赖 `isaaclab_rl`。

验收结果：三项 Gym task 正确注册；静态 import 不启动 Isaac Lab/Kit；删除 vendored packages 后两个 16 步 camera run 仍通过。

### 5.4 重新实现 Go2 delayed actuator

Isaac Lab 2.3.2 仍有 delayed PD actuator，但不再暴露旧 `DelayedDCMotor`。RAMBO-local `DelayedDCMotor` 做三件事：

1. 用三组 `DelayBuffer` 同步延迟 position、velocity、feed-forward effort；
2. 每个环境 reset 时采样 0–10 physics steps 延迟，或使用配置的固定 delay；
3. 把 torque-speed saturation 交给官方 2.3.2 `DCMotor`。

两个 mode 共享唯一的 Go2 actuator factory：

| joint | effort/saturation | velocity | Kp/Kd | friction | delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| hip + thigh | 21.33 Nm | 30.1 rad/s | 40 / 1 | 0 | 0–10 physics steps |
| calf | 40.887 Nm | 15.70 rad/s | 40 / 1 | 0 | 0–10 physics steps |

验收结果：两种 task 的 delay/saturation contract 参数化测试通过；3000 步中 `max_torque_limit_violation=0`。

### 5.5 用名称映射替代 PhysX 裸索引

旧代码中的 `1,2,4,5...` 等索引会在 USD/body 排序变化后静默映射错误。新实现按名字解析后固定逻辑顺序：

```text
body:  base,
       FL/FR/RL/RR hip,
       FL/FR/RL/RR thigh,
       FL/FR/RL/RR calf,
       FL/FR/RL/RR foot

joint: FL/FR/RL/RR hip_joint,
       FL/FR/RL/RR thigh_joint,
       FL/FR/RL/RR calf_joint
```

`Head_*` 不进入 QP body order，但 contact sensor 必须能找到它们，且仍参与碰撞终止。

Isaac Sim 5 的 floating-base PhysX Jacobian view 可能省略 root row。实现检测 body dimension；若少一行，显式插入全零 base Jacobian row，再按名称 gather 其他 body。

验收结果：测试覆盖 root row 存在/不存在两种布局、缺失 joint/body、contact sensor 返回顺序错误；都能正确处理或明确失败。

### 5.6 保留 mode-specific QP 拓扑和观测语义

共同不变量：

- physics `dt=0.002`，500 Hz；
- `decimation=5`，policy/control 100 Hz；
- 5 帧 observation history；
- 18 维动作 = 6 维 base action + 12 维 joint action；
- QP solver 仍为 qpth `QPFunction`，没有更换求解器或网络结构。

quadruped：

- 每帧 81 维，history 后 405 维；
- FL 是单前肢 end-effector；
- FL 的三组 joint/contact slots 在 QP feed-forward path 中强制按接触处理；
- external EE force 只施加到 `FL_foot`。

biped：

- 每帧 87 维，history 后 435 维；
- FL 和 FR 是两个独立 end-effector；
- FL/FR 六组 joint/contact slots 都进入 QP 接触覆盖；
- external force 分别施加到 `FL_foot` 和 `FR_foot`；
- Go2 base 初始绕 Y 轴 -90° 直立，原生 gravity target 为 `(-1,0,0)`。

验收结果：两个 task 分别严格匹配 405/18 和 435/18 checkpoint；没有把 quadruped observation/controller 隐式套给 biped。

### 5.7 对齐 Isaac Lab 2.3.2 DirectRLEnv step/reset

RAMBO 保留自定义控制顺序，但补齐 2.3.2 runtime 语义：

```text
action noise
→ action scale/clip
→ QP desired state + GRF
→ desired position/velocity/torque filter
→ external EE force
→ 每个 decimation：target → scene.write → PhysX → conditional render → scene.update
→ episode/common counters
→ contact generator + joint controller
→ done → reward → reset
→ reset RTX rerender
→ interval events
→ observation/history
→ observation noise
```

实现使用 2.3.2 `action_noise_model`、`observation_noise_model`、event manager、`_sim_step_counter`、render interval 和 `num_rerenders_on_reset`。validator 开启 termination snapshot，在 DirectRLEnv 自动 reset 覆盖状态之前保存真实失败原因。

原生安全阈值保持不变：

| mode | 最低 base height | 最大 orientation error | gravity target |
| --- | ---: | ---: | --- |
| quadruped | 0.1 m | 0.75 | `(0,0,-1)` |
| biped | 0.3 m | 0.8 | `(-1,0,0)` |

### 5.8 迁移 CRL2 Gymnasium wrapper

新 wrapper 使用：

- `single_action_space`；
- `single_observation_space`；
- `gymnasium.spaces.flatdim`；
- Dict observation 中只展开 `policy` member；
- 对外保留 CRL2 tensor tuple 接口。

一个关键修复是缓存 reset/step 返回的 observation。RAMBO 的 `_get_observations()` 会推进五帧 history；如果 wrapper contract check 或 PPO init 再调用它，policy 的第一步输入会被悄悄改变。现在 `get_observations()` 优先返回 `_last_observations`/`obs_buf`，不会重复推进 history。

CRL2 另有两个小改动：

- observation normalizer 的 `count` 是普通 Python attribute，保存和恢复时显式处理；
- W&B 改为真正的可选依赖，只有选择 wandb logger 才报缺包。

验收结果：wrapper dimensions、tensor tuple、cached observation 和 normalizer count 测试通过。

### 5.9 建立严格 checkpoint contract

loader 在反序列化之前检查：

- task 对应 SHA256；
- policy/value 每层 shape；
- observation normalizer `_mean/_var/_std` shape；
- iteration；
- `obs_normalizer_count=100003840`；
- 405/18 或 435/18 schema。

恢复时不调用旧 `PPO.load(path)` 二次反序列化，而是把已验证的内存 checkpoint 精确写入 policy、可选 value、normalizer 和 iteration，再可选逐 tensor byte-exact 比较。

验收结果：正确 checkpoint 两种 mode 都通过；错误 hash 在 deserialization 前被拒绝；Button task 复用 quadruped allow-listed checkpoint。

### 5.10 修复 Blackwell qpth 动态加载

Torch 2.7 cu128 wheel 延迟加载 `libtorch_cuda_linalg.so`；qpth 第一次 CUDA linalg 时，宿主机默认 loader path 找不到 wheel 内 sibling library。另一个 JIT 路径需要 `libnvrtc-builtins.so.12.8`。

修复分两层：

1. `scripts/rambo/run.sh` 把 `torch/lib` 和 `nvidia/cuda_nvrtc/lib` 放在 `LD_LIBRARY_PATH` 前部；
2. `rambo.torch_runtime.ensure_cuda_linalg_loaded()` 以 absolute path 和 `RTLD_GLOBAL` 预加载 `libtorch_cuda_linalg.so`。

最终 qpth 0.0.18 本身没有打补丁。CPU/CUDA forward、backward、KKT residual 和重复确定性均通过。

### 5.11 建立 deterministic validation harness

标准验收固定：

- `seed=42`；
- 单环境；
- episode 31 秒，运行恰好 3000 个 100 Hz step；
- `events=None`；
- 关闭 observation noise、随机初态、随机 episode progress；
- 关闭 sampled velocity/position/force commands；
- 关闭所有 controller/marker debug visualization；
- 深拷贝并延长 contact schedule，不污染 registry class config。

每步检查 finite：observation、action、reward、GRF、QP cost、desired position/velocity/torque、applied torque、root pose 和 projected gravity。还检查 native safety threshold 和 torque limit。

`summary.json` 永远记录 `passed`，失败时记录原始 exception；output directory 非空时拒绝覆盖，避免把两次验证产物混在一起。

### 5.12 精确 RGB cadence 和 fresh-frame recorder

目标相机是 RAMBO 添加在 Go2 `base` 上的合成 PinholeCamera，不是 Go2 USD 中已有的真实硬件相机。GUI viewport 截图也不是这路 sensor；`validate.py` 生成的 RGB/contact sheet 才是 task 前视相机。

相机 contract：640×480 RGB、更新周期 0.08 秒，即 12.5 Hz。3000 个 100 Hz policy step 应得到 375 帧。

实现中发现 Isaac Lab generic sensor clock 使用 float32 timestamp。在 2 ms physics dt 下，理论 40 ticks 的周期偶尔变成 41 ticks，早期 3000 步只得到 356 帧；240 步 eager run 也得到 29/30。

最终修复：RAMBO-only `Camera` subclass 用整数 physics tick 决定 deadline；0.08/0.002 精确等于 40 physics ticks，也等于 8 policy steps。配置仍公开真实 `update_period=0.08`。

recorder 还有两个顺序约束：

1. lazy sensor 必须先读取 `camera.data`，再读取 `camera.frame`；读取顺序反过来会永远看到旧 frame id；
2. 最后一个 policy step 恰落在 deadline 时，额外 render 一次只 flush render product，不推进 camera cadence，然后尝试消费最终 fresh frame。

自动 RGB gate 检查：

- 恰好 375 个连续 frame id；
- 尺寸持续为 640×480×3；
- min mean > 2、min std > 1；
- max adjacent MAD > 1；
- first/last MAD > 1；
- changing fraction > 0.1；
- 写出 `rgb/*.png`、timestamps、frame ids、六帧 contact sheet。

### 5.13 稳定 biped deterministic phase

关闭随机 episode progress 后，biped 在 gait phase 0 的诊断 run 于 policy step 66 触发原生姿态终止：

```text
large_orientation_error
base_height=0.365897
orientation_error=0.811436 > 0.8
```

这不是通过放宽 threshold 解决。诊断扫描确定 `contact_phase_offset_s=19.6` 能让原 checkpoint 在目标 runtime 稳定。offset 只进入 biped contact generator 的 gait clock，不写 `episode_length_buf`、不消耗 31 秒 episode budget，也不重新启用随机 progress。contact sequence 被复制并延长到至少 `19.6 + 31` 秒。

验收结果：80、240、3000 步逐级通过；标准 `summary.json` 明确记录 `contact_phase_offset_s: 19.6`。

### 5.14 修复 biped 相机“中间一团灰色”

biped 把 Go2 base 绕 Y 轴 -90°。最初只调整了相机 rotation，没有把 mount position 一同变换，导致相机位于 chassis 后/内部，contact sheet 中间出现大块灰色机身。

最终 parent-frame offset：

```text
position = (0.08, 0.0, -0.30)
rotation = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)
```

它在初始 world frame 中等价于相机位于 base 前方 +0.30 m、上方 +0.08 m，并朝 world +X。修复后 16 步 GUI/camera 和完整 3000 步 camera run 通过，contact sheet 不再看入 chassis。

### 5.15 处理 camera run 正常结束时的 Replicator hang

Isaac Sim 5.1 中，即使 `close(wait_for_replicator=False)`，live camera render product 仍可能同步进入 `rep.orchestrator.stop()` 并卡住。

runner 的处理原则：

- 失败路径保留正常 traceback 和 non-zero exit，不用 immediate exit 吞掉错误；
- 成功路径在 env、RGB、summary 都 flush 后调用 `simulation_app.close(skip_cleanup=True)`；
- camera smoke 只有同步读取，不持有 writer，也在成功收尾后 immediate close。

### 5.16 移除 vendored Isaac Lab

双模式、camera、qpth 和 checkpoint gate 通过后，提交 `62e0a51` 删除 `source/isaaclab*`、`source/isaaclab_assets`、`source/isaaclab_tasks`、`source/isaaclab_rl`、`source/isaaclab_mimic`。新分支只依赖官方 pip package 和 `source/rambo`。

删除后重新执行 quadruped/biped 16 步验证，均得到 `passed=true` 和 2/2 RGB 帧。

### 5.17 添加 quadruped 物理 Button task

Button 使用原生 USD/PhysX asset，不改变官方 pip package：

- 墙板：固定 Cuboid；
- 红色 cap：0.15 kg rigid Cuboid；
- prismatic joint：沿 +X，行程 0–20 mm；
- force drive：max effort 40 N、max velocity 0.5 m/s、stiffness 800、damping 25；
- press threshold：12 mm；
- 必须连续保持 5 个 control steps；
- release threshold：2 mm；
- success latch 用普通 USD 绿色 sphere 显示；
- base 进入 2 mm contact guard 后禁止继续向墙推进。

FL 在第 1 秒后进入 swing/manipulator mode，其他三腿继续 checkpoint-compatible gait。Button 配置固定单环境、无随机化、episode 300 秒。

### 5.18 添加 loco-manip runner 和双层验收

headless combined smoke 在一个 environment 中按固定阶段执行：

```text
step 110–269: base vx 前进
step 300–549: FL z 上升
step 550–809: FL x 前进并压按钮
step 810–909: FL x 回撤
```

它要求：

- base displacement > 0.05 m；
- joint delta > 0.05 rad；
- 实体 cap 超过 12 mm 并保持 5 steps；
- 回撤后 cap 小于 2 mm；
- 全程没有 done/reset；
- policy action 全部 finite。

随后通过真实 X11/Isaac Sim viewport event path 验证交互，而不只依赖 scripted branch。期间修复了五类问题：

1. Isaac Sim 5.1 的 `event.input` 可能是 enum，也可能直接是 string；统一解析 `.name` 或 `str(raw_key)`；
2. 官方 `Se2Keyboard` 的旧 handler 也假定 `.name`，因此 RAMBO handler 完整处理 base press/release，不能在 string path 上调用旧 superclass；
3. `Space` 是 Isaac Sim timeline hotkey，改为松开方向键停止对应命令，`L` 全部 reset；
4. interactive run 也严格检查 done，不能在 env 已自动 reset 后继续制造“看似可操作”的假象；
5. release telemetry 加 latch 去抖，一个按压周期只报告一次 `RELEASED`。

## 6. 最终验收结果

本节数值来自迁移过程中保留的原始 `summary.json`/teleop log，而不是事后估算。quadruped 的完整 3000 步结果产生于双模式核心定稿阶段；随后 vendored package 删除通过 16 步回归，Button 提交对 base task 只增加默认 no-op 的 task-asset hook，并通过完整 loco-manip gate。biped 的 3000 步结果已包含最终相机 mount 修复。新宿主机仍必须按第 4.7 节在当前 HEAD 上重新执行两种 3000 步验收，不能仅引用这里的历史数值。

### 6.1 quadruped 3000 步

最终 artifact 原始位置：`/tmp/rambo-quadruped-validation-3000-final/summary.json`。`/tmp` 不是长期存储；新机器应使用 `/workspace/rambo-validation/...`。

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| passed / steps | true / 3000 | true / 3000 |
| action shape | `[3000,18]` | `[3000,18]` |
| action NaN / Inf | 0 / 0 | 0 / 0 |
| action min / max | -3.690178 / 3.386194 | finite |
| min base height | 0.263394 m | ≥ 0.1 m |
| max orientation error | 0.157477 | ≤ 0.75 |
| max desired torque | 17.328260 Nm | within task limits |
| max applied torque | 31.690445 Nm | finite |
| max GRF | 119.304924 | finite |
| max QP cost | 744.286682 | finite |
| max torque limit violation | 0 | ≤ 1e-4 |
| RGB | 375 / 375，640×480 | 375 / 375 |
| RGB changing fraction | 1.0 | > 0.1 |
| first-last MAD | 6.571302 | > 1 |
| max adjacent MAD | 25.411445 | > 1 |

人工 contact sheet 检查：正常前视画面，无黑帧、固定帧或雪花。

### 6.2 biped 3000 步（相机位置修复后）

最终 artifact 原始位置：`/tmp/rambo-biped-validation-3000-camera-fixed/summary.json`。

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| passed / steps | true / 3000 | true / 3000 |
| action shape | `[3000,18]` | `[3000,18]` |
| action NaN / Inf | 0 / 0 | 0 / 0 |
| action min / max | -4.514565 / 3.988093 | finite |
| min base height | 0.428399 m | ≥ 0.3 m |
| max orientation error | 0.183116 | ≤ 0.8 |
| max desired torque | 34.298717 Nm | within task limits |
| max applied torque | 40.887001 Nm | finite |
| max GRF | 279.103363 | finite |
| max QP cost | 712.390930 | finite |
| max torque limit violation | 0 | ≤ 1e-4 |
| validation gait offset | 19.6 s | fixed、非随机 |
| RGB | 375 / 375，640×480 | 375 / 375 |
| RGB changing fraction | 1.0 | > 0.1 |
| first-last MAD | 24.930712 | > 1 |
| max adjacent MAD | 18.083529 | > 1 |

人工 contact sheet 检查：相机已在 chassis 前方，不再出现“中间一团灰色”的机身内部视图；无雪花或黑帧。

### 6.3 GUI 短验证

| 模式 | steps | RGB | 结果 |
| --- | ---: | ---: | --- |
| quadruped GUI | 16 | 2 / 2 | passed |
| biped GUI + 修正后 mount | 16 | 2 / 2 | passed |
| 删除 vendored packages 后 quadruped | 16 | 2 / 2 | passed |
| 删除 vendored packages 后 biped | 16 | 2 / 2 | passed |

### 6.4 loco-manip headless 和真实键盘

最终 combined smoke：

```text
[BUTTON] PRESSED: travel=16.8 mm
[BUTTON] RELEASED
[SMOKE] forward displacement=0.524m joint_delta=0.972rad
LOCO_MANIP_SMOKE_LOCO_MANIP_SUCCESS
```

真实 viewport keyboard session（同一会话，无 reset/termination/truncation）：

| 阶段 | 键事件 | 实测 |
| --- | --- | --- |
| 行走 | hold/release `Up` | `root_x=0.528` |
| FL 抬升 | hold/release `R` | FL z `0.315` |
| 按压 | hold/release `W` | cap travel `19.9 mm`，success=1 |
| 回撤 | hold/release `S` | cap travel `0.0 mm`，released=1 |
| 清除 | press `C` | `[BUTTON] CLEARED` |

GUI 画面显示 Go2、墙板、红色 cap 和绿色 success indicator，渲染正常。

### 6.5 测试覆盖

当前 `tests/rambo` 共 26 个通过项，覆盖：

- qpth CPU/CUDA forward/backward/KKT/determinism；
- 三 task registration 和 RAMBO-local registry；
- quadruped/biped checkpoint hash、shape、iteration、normalizer count；
- 错 hash 在反序列化前拒绝；
- CRL2 Gymnasium flatdim 和缓存 observation；
- Go2 body/joint/Jacobian/contact name order；
- delayed actuator 参数与 saturation contract；
- deterministic config 不修改源 contact schedule；
- biped 固定 gait phase 不推进 episode；
- 3000-step camera integer cadence；
- biped upright camera mount 变换；
- fresh frame、lazy data-before-frame、RGB recorder；
- Isaac Sim enum/string 键盘输入兼容；
- static import 不启动 Kit；
- runtime 不依赖已删除的 vendored `isaaclab_tasks`。

## 7. 失败记录与排查表

| 症状 | 实际原因 | 已采用修复 | 复核命令/证据 |
| --- | --- | --- | --- |
| RTX 5080 上旧 qpth/CUDA 不能执行 | Torch 2.5.1 cu124 无 `sm_120` | 独立 Python 3.11 + Torch 2.7 cu128 | `smoke.py` capability/arch gate |
| `pkg_resources` 缺失 | setuptools 81 移除历史入口 | 固定 `setuptools<81` | `pip check` + import |
| `flatdict` 缺失 | pip bundle 依赖未预装 | 先装 4.0.1 | setup script |
| `UNITREE_GO2_CFG` import 失败 | wheel 内 assets 源码未安装为 distribution | 从 official wheel 内路径安装 `isaaclab_assets` | official Go2 smoke |
| pip 安装 qpth 后 Isaac Sim 依赖 broken | 最新 CVXPY/OSQP/packaging 覆盖官方 pins | CVXPY 1.5.4、OSQP 0.6.7.post3、packaging 23.0 | `pip check` |
| CUDA qpth 找不到 linalg/NVRTC so | wheel sibling library 不在 late `dlopen` path | `run.sh` LD path + absolute preload | `RAMBO_REQUIRE_CUDA_QPTH=1 ...` |
| `bad marshal data` | wheel 中 `.pyc` 来自不同 Python 3.11 micro | `PYTHONPYCACHEPREFIX=/tmp/rambo51-pycache` | 通过 wrapper 启动 |
| 普通 `import rambo` 启动 Kit/EULA | 顶层 import 触发 Isaac Lab | `kit_runtime_ready()` guard | static import test |
| 第一 policy action 与 legacy history 不符 | wrapper 重复调用 mutating `_get_observations()` | 缓存 reset/step observation | CRL2 cache test |
| link/Jacobian 错位 | 依赖旧数字索引和 root row 假设 | 名称解析 + root zero row | articulation tests |
| quadruped 3000 步只录 356/375 | float32 sensor timestamp cadence 漂移 | integer physics-tick camera | final 375/375 |
| 240 步录 29/30 | eager/final render deadline 丢帧 | lazy data-first + terminal flush | camera tests |
| biped step 66 摔倒 | deterministic gait 从不稳定 phase 0 开始 | 固定 contact-only offset 19.6 s | 3000 passed，不放宽 safety |
| biped contact sheet 中间灰块 | 相机 position 没随 upright base 变换，镜头在 chassis 内 | position+rotation 同时变换 | camera-fixed contact sheet |
| camera run 成功后进程不退出 | Replicator synchronous stop hang | 成功 flush 后 `skip_cleanup=True` | CLI 正常 0 退出 |
| 合成/真实 GUI key 报 `.name` error | Isaac Sim 5.1 event input 可能是 string | enum/string normalization + 自有 handler | teleop test + live session |
| `Space` 后仿真停住 | Isaac Sim timeline 全局 hotkey | 不绑定 Space，使用 key release/`L` | GUI live session |
| 按钮回弹重复打印 RELEASED | threshold 附近物理抖动 | 每个 success 周期 release latch 去抖 | final smoke 1 PRESS/1 RELEASE |
| validator 拒绝 output directory | 有意避免覆盖既有证据 | 换一个新的空目录 | error message 即说明 |

## 8. 正常警告与真正失败的区分

以下信息在验证机上出现过，但本身不构成失败：

- `Could not initialize nvperf_grfx_host library`：PerfSDK disabled，不影响功能；
- `omni.isaac.dynamic_control is deprecated`：官方依赖发出的兼容警告；
- IOMMU enabled：信息提示；
- CPU performance profile 为 powersave：会影响速度，建议生产验证改 performance，但不改变 pass/fail；
- `quat_rotate` / `quat_rotate_inverse` 将弃用：当前 2.3.2 仍可用，属于后续技术债；
- `set_external_force_and_torque` 将弃用：当前 loco-manip 仍可用；
- `DirectRLEnvCfg.num_actions/num_observations` deprecated：同时已经提供新 `action_space/observation_space`，当前不是失败；
- render interval 小于 decimation：camera/GUI 路径可能每个 control step 多次 render，是已知配置提示。
- 某些 `summary.json` 中 `isaaclab_version`/`isaacsim_version` 为 `unknown`：这两个 distribution 在该 import path 没有公开 `__version__`，以 `python -m pip show` 或 `importlib.metadata.version(...)` 核对，不影响数值验收。

以下情况必须判失败：

- process 非零退出或没有 `VALIDATION_SUCCESS`；
- `summary.json` 不存在或 `.passed != true`；
- 任一 early terminated/truncated；
- 任一 NaN/Inf；
- torque violation > `1e-4`；
- base height/orientation 超过 mode 原生安全阈值；
- RGB 不是 375/375、不是 640×480、黑帧/固定帧；
- GUI 出现雪花、chassis 内部灰块或按钮没有真实位移；
- loco-manip 只跑了 scripted smoke，却没有验证实际 viewport keyboard path。

## 9. 新宿主机验收清单

### 9.1 环境

- [ ] `new_IsaacSim_IsaacLab` 含 `dcf5e32` 至 `1034bbf` 的实现；
- [ ] 没有把 legacy `source/isaaclab*` 加回新分支；
- [ ] Python 3.11、Torch 2.7.0+cu128、CUDA runtime 12.8；
- [ ] GPU capability `(12,0)`，Torch arch list 包含 `sm_120`；
- [ ] driver ≥ 580.65.06；
- [ ] `pip check` 通过；
- [ ] operator 已完成 EULA；
- [ ] official Go2 GUI、headless physics 和 camera smoke 通过。

### 9.2 数值和 checkpoint

- [ ] 两个 checkpoint SHA256 完全匹配；
- [ ] qpth CPU/CUDA 2 tests 通过；
- [ ] RAMBO 26 tests 通过；
- [ ] quadruped 405/18，biped 435/18；
- [ ] normalizer count 恢复为 100003840；
- [ ] 两种 mode 都无 early done、NaN/Inf 或 torque violation；
- [ ] biped summary 记录 contact phase 19.6，episode 仍为完整 31 秒。

### 9.3 RGB 和 GUI

- [ ] 每种 mode 375 个连续 fresh frame；
- [ ] 每帧 640×480；
- [ ] contact sheet 无黑帧/固定帧/雪花；
- [ ] biped 视野在 chassis 前方；
- [ ] 理解 sensor RGB 是 RAMBO 合成 base camera，不是 Go2 USD 自带硬件相机；
- [ ] GUI viewport 对两种 mode 都正常。

### 9.4 loco-manip

- [ ] headless combined smoke 输出 success marker；
- [ ] 同一 GUI session 中 base 产生可见行走；
- [ ] FL 进入 manipulator-ready 后可用 W/S/A/D/R/F 连续控制；
- [ ] cap 位移 ≥12 mm 且保持 5 steps；
- [ ] success marker 变绿；
- [ ] FL 回撤后 cap ≤2 mm；
- [ ] `C` 只在回弹后清除 latch；
- [ ] 全程无 reset/termination/truncation；
- [ ] 没有用 `Space` 作为停止键。

## 10. 产物保存建议

不要把新一轮证据只留在 `/tmp`。推荐每台宿主机使用独立目录：

```text
/workspace/rambo-validation/<hostname>/<date>/
├── host.txt
├── pip-freeze.txt
├── qpth-test.txt
├── tests-rambo.txt
├── quadruped/
│   ├── summary.json
│   ├── contact_sheet.png
│   ├── rgb_timestamps.npy
│   ├── rgb_frame_ids.npy
│   └── rgb/
├── biped/
│   └── ...
└── loco-manip/
    ├── smoke.log
    ├── live-keyboard.log
    └── gui-success.png
```

建议保存以下 host 信息：

```bash
uname -a
cat /etc/os-release
nvidia-smi
"$RAMBO_VENV/bin/python" -m pip freeze --all
git -C "$RAMBO_REPO" rev-parse HEAD
git -C "$RAMBO_REPO" status --short --branch
```

只有在新机器完整通过后，才考虑更新 `requirements/isaacsim51.lock`。更新时保留 NVIDIA/PyTorch extra-index header，并排除本机 editable absolute paths。

## 11. 最短故障定位顺序

新宿主机失败时按这个顺序，不要先改 task 参数：

1. `nvidia-smi` 和 capability；
2. venv Python/Torch/CUDA/arch；
3. `pip check` 和 official `isaaclab_assets` import；
4. official Go2 smoke，无 RAMBO；
5. qpth CPU，再 qpth CUDA；
6. RAMBO static/unit tests；
7. quadruped 16 steps；
8. biped 16 steps，确认默认 19.6 phase；
9. 各自 3000 steps，无 camera 参数改写；
10. contact sheet 人工检查；
11. loco-manip headless combined smoke；
12. 最后才验证实际 GUI keyboard。

不要通过以下方式“修复”失败：

- 降低 height threshold 或提高 orientation threshold；
- 关闭 body/head/limb termination；
- 重新启用随机 episode progress 来碰运气；
- 忽略 checkpoint hash；
- 改网络结构或替换 qpth solver；
- 把 legacy vendored Isaac Lab 覆盖到 venv；
- 接受少于 375 帧；
- 把灰色 chassis 内部画面当成正常 biped camera。

按本文 gate 全部通过后，新 Blackwell 宿主机才算复现了当前 RAMBO Isaac Sim 5.1 adaptation 的已有进展。
