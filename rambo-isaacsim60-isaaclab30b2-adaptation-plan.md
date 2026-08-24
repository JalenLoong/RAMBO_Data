# RAMBO → Isaac Sim 6.0.1 / Isaac Lab 3.0 Beta 2 Patch 1 迁移计划

> 目标：在当前 Vast.ai RTX 4090 + R595 实例上，将已经在 Isaac Sim 5.1 / Isaac Lab 2.3.2 上验证通过的 RAMBO 迁移到新版 Isaac 栈，同时保持原 RAMBO policy / checkpoint / action / observation / QP 语义不变，为后续 LingBot-VA → RAMBO adaptation 数据合成建立稳定运行基线。

## 0. 当前目标主机

| 项目 | 当前配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| VRAM | 24 GB |
| Compute Capability | 8.9 (`sm_89`) |
| NVIDIA Driver | 595.71.05 |
| `nvidia-smi` Max CUDA | 13.2 |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.0-124-generic |
| RAM | 61 GiB |
| Swap | 8 GiB |
| Workspace disk | 100 GB total / 56 GB available at migration start |
| Vulkan | NVIDIA RTX 4090 correctly enumerated |

这台机器作为本次新版迁移的目标 host。后续不修改 host NVIDIA driver。

## 1. 目标软件栈

本次迁移接受 Isaac Lab Beta，但只使用固定、可复现的 tagged release，禁止 nightly / `develop` / floating branch。

```text
OS: Ubuntu 24.04.4 LTS
Driver: 595.71.05
GPU: RTX 4090, CC 8.9 / sm_89

Python: 3.12
Isaac Sim: 6.0.1.0
Isaac Lab: v3.0.0-beta2.patch1

PyTorch: 2.10.0 + cu128
torchvision: 0.25.0 + cu128

Physics backend: PhysX only
Renderer: Isaac RTX
Headless: --viz none
GUI: --viz kit

QP solver: qpth 0.0.18
```

注意：

- `nvidia-smi` 显示 CUDA 13.2 只代表 host driver 支持的最高 CUDA API family。
- RAMBO runtime 不要求安装 CUDA 13.2。
- Torch / CUDA runtime 跟随 Isaac Lab 3.0 Beta 2 官方 x86_64 contract。
- 不迁移到 Newton。
- 不在迁移过程中更换 QP solver。

## 2. 迁移核心原则

本次目标不是重新设计 RAMBO，而是：

> 让已经验证过的 RAMBO checkpoint 在新的 Isaac Sim / Isaac Lab 栈中继续保持原有控制语义与任务行为。

迁移期间禁止改变：

```text
18D policy action
405D quadruped observation
435D biped observation

base action indices
joint action indices
action scaling
joint/body ordering
observation normalization
checkpoint semantics
QP objective / constraints
teleop key semantics
button task semantics
trajectory serialization semantics
```

## 3. Isaac Sim 5.1 已验证 baseline

旧环境：

```text
Isaac Sim 5.1.0
Isaac Lab 2.3.2
Python 3.11
Torch 2.7.0 + cu128
Driver 580.159.03
RTX 5080
```

| Task | Observation / Action | Checkpoint | 旧版验收 |
|---|---:|---|---|
| Quadruped | 405 / 18 | `model_2000.pt` | 3000 steps + 375 RGB |
| Biped | 435 / 18 | `model_4000.pt` | 3000 steps + 375 RGB |
| Quadruped Button loco-manip | 405 / 18 | `model_2000.pt` | walking + FL button press + rebound + GUI keyboard |

Checkpoint SHA256：

```text
quadruped:
1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706

biped:
c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6
```

旧 `new_IsaacSim_IsaacLab` 分支作为 frozen fallback，迁移分支不得覆盖它。

# 4. Milestone 总览

| Milestone | 目标 | 未通过是否继续 |
|---|---|---|
| M0 | 清理空间、冻结旧 baseline | 否 |
| M1 | 安装 exact Isaac Sim 6.0.1 / Isaac Lab 3.0 Beta 2 Patch 1 | 否 |
| M2 | 官方 CUDA / Cartpole / Go2 / RGB / GUI 验证 | 否 |
| M3 | 建立 RAMBO API migration inventory | 否 |
| M4 | 基础 Isaac Lab 3.0 API 迁移 | 否 |
| M5 | qpth / QP stack 迁移 | 否 |
| M6 | Quadruped 单环境 + checkpoint | 否 |
| M7 | Quadruped RGB + 3000-step | 否 |
| M8 | Button loco-manip + GUI keyboard | 否 |
| M9 | Biped | 否 |
| M10 | LingBot adaptation 数据合成 contract | 否 |
| M11 | requirements lock / runbook / Docker image | 最后执行 |

每个 milestone 都必须：

```text
修改 → 测试 → 保存 artifact → commit → 再进入下一 milestone
```

# M0. 磁盘清理与旧工作保护

## M0.1 检查磁盘占用

当前约 56 GB 可用，新版 Isaac Sim + Isaac Lab + Torch + Kit cache 可能不足。

```bash
du -xhd1 /workspace | sort -h
du -xhd1 /workspace/venvs 2>/dev/null | sort -h
du -xhd1 /root/.cache 2>/dev/null | sort -h
du -xhd1 /root/.nv 2>/dev/null | sort -h
df -h /workspace
```

建议安装前至少保证约 70–75 GB free。

## M0.2 可以清理的旧内容

在确认旧代码、checkpoint、validation artifacts 已保存后，可删除新实例上的旧运行时与纯缓存：

```bash
rm -rf /workspace/venvs/rambo51
rm -rf /root/.cache/pip
rm -rf /root/.cache/uv
rm -rf /root/.nv/ComputeCache
```

不要删除：

```text
/workspace/rambo
/workspace/rambo-checkpoints
旧 16-step / 3000-step validation
RGB reference
trajectory / dataset
runbook
migration logs
```

## M0.3 推荐目录布局

```text
/workspace/
├── rambo/                          # Isaac 5.1 frozen baseline
├── rambo60/                        # 新迁移 worktree
├── IsaacLab-3.0.0-beta2.patch1/    # 官方 exact tag
├── venvs/
│   └── rambo60/
├── rambo-checkpoints/
├── migration-reference/
├── migration-output/
└── migration_logs/
```

# M1. Git baseline 与新版环境安装

## M1.1 冻结旧版

```bash
cd /workspace/rambo
git fetch origin
git switch new_IsaacSim_IsaacLab
git status --short
git log -5 --oneline
```

确认 clean 后：

```bash
git tag -a   isaac51-golden-10154c6   10154c6d2f20d33f51e5e126801c507e6fe206db   -m "Known-good Isaac Sim 5.1 / Isaac Lab 2.3.2 baseline"

git worktree add   -b adapt/isaacsim60-isaaclab30b2   /workspace/rambo60   10154c6d2f20d33f51e5e126801c507e6fe206db
```

后续 Codex 只允许修改：

```text
/workspace/rambo60
```

## M1.2 退出旧 conda 环境

如果 shell 显示 `(main)`：

```bash
conda deactivate || true
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

不要将新版 Isaac 栈安装进原 conda `main`。

## M1.3 安装基础工具

```bash
apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y   build-essential   cmake   git   git-lfs   python3.12   python3.12-dev   curl
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version
python3.12 --version
ldd --version | head -1
```

## M1.4 Clone exact Isaac Lab tag

```bash
git clone   --depth 1   --branch v3.0.0-beta2.patch1   https://github.com/isaac-sim/IsaacLab.git   /workspace/IsaacLab-3.0.0-beta2.patch1

cd /workspace/IsaacLab-3.0.0-beta2.patch1

git describe --tags --exact-match
git rev-parse HEAD
git status --short
```

必须确认：

```text
v3.0.0-beta2.patch1
```

禁止使用 `develop`、`main`、nightly、floating release branch 或未固定版本的安装。

## M1.5 创建全新 Python 3.12 venv

```bash
export RAMBO60_VENV=/workspace/venvs/rambo60

uv venv   --python 3.12   --seed   "$RAMBO60_VENV"

source "$RAMBO60_VENV/bin/activate"

python --version
which python
```

## M1.6 安装 Isaac Sim / Torch / Isaac Lab

```bash
uv pip install   "isaacsim[all,extscache]==6.0.1.0"   --extra-index-url https://pypi.nvidia.com   --index-strategy unsafe-best-match   --prerelease=allow

uv pip install   "torch==2.10.0"   "torchvision==0.25.0"   --index-url https://download.pytorch.org/whl/cu128
```

然后：

```bash
cd /workspace/IsaacLab-3.0.0-beta2.patch1

./isaaclab.sh -i   "assets,physx,tasks,visualizers[kit]"
```

初期不安装 Newton、Mimic、Teleop 和无关 RL framework，也暂时不安装 qpth、CRL2 或 RAMBO editable package。

## M1.7 版本 gate

```bash
python - <<'PY'
import importlib.metadata as m
import sys
import torch

print("python", sys.version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torchvision", m.version("torchvision"))
print("isaacsim", m.version("isaacsim"))
print("isaaclab", m.version("isaaclab"))
print("numpy", m.version("numpy"))
print("cuda_available", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("arch_list", torch.cuda.get_arch_list())
PY
```

目标：

```text
Python 3.12
Torch 2.10.0
Torch CUDA 12.8
Isaac Sim 6.0.1.0
Isaac Lab v3.0.0-beta2.patch1
RTX 4090
Capability (8, 9)
sm_89 available
```

# M2. 官方 Isaac stack 验证

这个阶段禁止 import RAMBO。

## M2.1 CUDA gate

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 9)
assert "sm_89" in torch.cuda.get_arch_list()

x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = x.square().sum().item()
torch.cuda.synchronize()

print("CUDA_OK", y)
PY
```

## M2.2 官方 Cartpole headless

先找实际 task name：

```bash
cd /workspace/IsaacLab-3.0.0-beta2.patch1

./isaaclab.sh -p scripts/environments/list_envs.py   | grep -i cartpole
```

然后：

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py   --task <实际发现的 Cartpole Direct task>   --num_envs 16   --viz none
```

## M2.3 Kit GUI

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py   --task <同一 Cartpole task>   --num_envs 1   --viz kit
```

在 Selkies 中确认：

```text
GUI window appears
viewport normal
play / stop normal
no Vulkan swapchain error
nvidia-smi sees Kit GPU memory
active renderer GPU is RTX 4090
not llvmpipe
```

## M2.4 官方 Go2 + RGB smoke

在 `/workspace/rambo60` 新建：

```text
scripts/rambo60/official_go2_smoke.py
```

要求：

```text
不 import RAMBO
只使用官方 UNITREE_GO2_CFG
1 env
PhysX
ground plane
1000 physics steps
640x480 RGB camera
all state tensors finite
save one RGB frame
clean shutdown
```

运行至少两次：

```text
Run 1: cache / shader warm-up
Run 2: stable restart verification
```

M2 PASS：

```text
CUDA PASS
Cartpole headless PASS
Cartpole GUI PASS
Go2 physics PASS
Go2 RGB PASS
second launch PASS
```

未通过前禁止修改 RAMBO QP environment。

# M3. RAMBO migration inventory

重点审计：

```text
Quaternion WXYZ → XYZW
ProxyArray
write_*_to_sim API
AppLauncher / --viz
PhysX config
direct isaacsim / omni imports
camera / renderer
contact sensor
```

## M3.1 Quaternion 扫描

```bash
cd /workspace/IsaacLab-3.0.0-beta2.patch1

python scripts/tools/find_quaternions.py   --path /workspace/rambo60/source/rambo   --all-quats   > /workspace/migration_logs/quaternion-scan.txt
```

只生成报告，不自动批量修改。

## M3.2 全局 API inventory

```bash
cd /workspace/rambo60

rg -n   "rot\s*=|quat|convert_quat|root_quat|body_quat|write_.*_to_sim|\.data\.|root_physx_view|--headless|AppLauncher|isaacsim\.|omni\.|ContactSensor|CameraCfg|SimulationCfg|PhysxCfg"   source scripts tests apps   > /workspace/migration_logs/api-inventory.txt
```

生成：

```text
docs/isaacsim60-isaaclab30-migration-inventory.md
```

每项包含：

```text
file
line
old API
new API
semantic risk
required test
status
```

优先级：

```text
P0 quaternion convention
P0 QP tensor semantics
P0 write-to-sim API
P0 joint/body ordering

P1 ContactSensor
P1 camera pose
P1 renderer
P1 launcher
P1 direct omni / isaacsim imports

P2 randomization / event APIs
P2 deprecated aliases
```

# M4. 基础 Isaac Lab 3.0 API 迁移

该阶段只做框架兼容，不跑 checkpoint。

## M4.1 Quaternion

Simulator-facing quaternion 使用 XYZW。

```python
# old WXYZ
(1.0, 0.0, 0.0, 0.0)

# new XYZW
(0.0, 0.0, 0.0, 1.0)
```

Checkpoint / observation-facing quaternion 不得无条件一起改变。

必要时增加：

```python
sim_xyzw_to_legacy_wxyz(...)
legacy_wxyz_to_sim_xyzw(...)
```

运行时：

```bash
export WARN_ON_TORCH_QUATF_ACCESS=1
```

## M4.2 ProxyArray

所有进入 Torch / QP / policy / normalizer 的数据都必须显式确认 `.torch` semantics，例如：

```python
robot.data.joint_pos.torch
robot.data.joint_vel.torch
robot.data.root_pos_w.torch
sensor.data.net_forces_w.torch
```

## M4.3 Write API

逐项迁移并确认 indexing / shape semantics：

```text
root pose
root velocity
joint position
joint velocity
body pose
```

例如旧：

```python
robot.write_root_pose_to_sim(...)
```

迁移到 Lab 3.0 对应的 indexed/masked write API。

## M4.4 PhysX

显式使用 PhysX config，保持：

```text
physics dt = 0.002 s
control decimation = 5
原 friction
原 restitution
原 physics semantics
```

不迁移 Newton。

## M4.5 新 runtime files

新增：

```text
scripts/setup_isaacsim60.sh
scripts/rambo/run60.sh
scripts/rambo/smoke60.py
requirements/isaacsim60.in
requirements/isaacsim60.lock
```

旧 5.1 文件保留。

# M5. qpth / QP stack

```bash
source /workspace/venvs/rambo60/bin/activate
uv pip install --no-deps "qpth==0.0.18"
```

禁止 resolver 自动降级 Torch / NumPy / Isaac dependencies 或安装旧 CUDA runtime。

## M5.1 Toy CUDA QP

验证：

```text
forward finite
backward finite
batch solve
equality residual
inequality residual
CUDA synchronize
```

## M5.2 RAMBO QP snapshot

从旧 baseline 取：

```text
mass matrix
Jacobian
contact state
desired force
constraint matrix
```

比较：

```text
QP solution
joint torque
contact force
constraint residual
```

如果 qpth 仅有 Python 3.12 / Torch 2.10 build incompatibility，优先做最小 compatibility patch，不更换 solver。

# M6. Quadruped 单环境迁移

初始配置：

```text
num_envs = 1
camera disabled
domain randomization disabled
external push disabled
fixed seed
PhysX
--viz none
```

## M6.1 注册与构建

确认 `Isaac-RAMBO-Quadruped-Go2-v0`：

```text
register
instantiate
resolve official Go2
resolve all joints
resolve all bodies
```

## M6.2 Ordering contract

严格验证：

```text
12 joint order
foot body order
calf body order
thigh body order
sensor body ids
Jacobian order
```

## M6.3 Zero-action

运行 100 steps 和 1000 steps，要求：

```text
root state finite
joint state finite
contact finite
no unintended reset
no physics explosion
```

## M6.4 Observation / action

```text
Quadruped observation = 405
Policy action = 18
Action scaling unchanged
```

## M6.5 Checkpoint

```text
model_2000.pt
SHA256:
1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706
```

要求：

```text
strict state_dict load
first observation finite
first action finite
16-step rollout stable
```

优先比较：

```text
initial observation
first policy action
first QP output
```

长期 physics trajectory 不要求逐元素等于 Sim 5.1。

## M6.6 Long rollout

```text
3000 steps
camera disabled
```

要求：

```text
no NaN
no unintended reset
robot stable
base height bounded
joint velocity bounded
no sustained memory growth
```

# M7. RGB camera

开启：

```text
1 env
640x480
Isaac RTX
```

重新确认：

```text
camera pose
XYZW orientation
intrinsics
dtype
shape
channel order
value range
frame timing
first frame after reset
```

旧版 `num_rerenders_on_reset = 1` 不直接照搬，针对新版 renderer 重新测。

保持：

```text
3000 action steps
375 RGB frames
```

要求：

```text
non-black
non-zero
robot visible
scene visible
no dropped frame
timestamp alignment valid
```

# M8. Button loco-manip 与 GUI keyboard

先程序化验证：

```text
base velocity
FL Cartesian target
button contact
button depression
button rebound
```

证明 task physics 正常后再接 keyboard。

GUI：

```text
--viz kit
```

Selkies 验证：

```text
viewport
keyboard focus
key down/up
base command
FL target
Space conflict
button press
button rebound
telemetry
```

Selkies 不稳定时再考虑 Isaac Sim 官方 WebRTC。

# M9. Biped

只有 Quadruped 的 16-step、3000-step、RGB、Button、GUI 全通过后才迁 Biped。

要求：

```text
observation = 435
action = 18
model_4000.pt strict load
dual front-leg command independence
QP contact override independence
16-step PASS
3000-step PASS
375 RGB PASS
```

Checkpoint SHA：

```text
c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6
```

# M10. LingBot-VA adaptation 数据合成 contract

Simulator / RAMBO migration 完成后才恢复真正的数据合成。

每条 episode 至少记录：

```text
prompt
task id
seed

RAMBO commit
Isaac Lab tag
Isaac Sim version
driver version
GPU model
checkpoint SHA256

physics dt
control decimation
camera rate

observation schema version
action schema version

RGB timestamp
action timestamp
state timestamp

terminal state
task success
```

30 秒 trajectory gate：

```text
no missing RGB
no missing action
timestamps monotonic
frame / action / state aligned
terminal state recorded
task success recorded
seed reproducible
```

迁移阶段先保持单 RGB camera 和现有 RAMBO action contract，暂不同时加入 3-camera dataset、15D LingBot action 或新的 WAM recorder architecture。

# M11. 最终冻结、lock 与 Docker

只有所有 runtime gates 通过后才固化环境。

最终产物：

```text
requirements/isaacsim60.in
requirements/isaacsim60.lock

host-manifest.json
checkpoint-manifest.json
migration-report.md

IsaacLab exact tag + commit
RAMBO exact commit
```

Docker 目标：

```text
Ubuntu 24.04
R595-compatible userspace
Python 3.12
Isaac Sim 6.0.1.0
Isaac Lab v3.0.0-beta2.patch1
Torch 2.10.0 cu128
PhysX
RTX
```

Beta 风险控制：

```text
只使用 exact tag v3.0.0-beta2.patch1
不 follow release branch
不 follow develop
不使用 nightly
不自动 pip upgrade
完整 requirements lock
每个 milestone 保存 validation artifacts
未来 GA 单独开 migration branch
```

# 5. Codex 执行纪律

所有 Codex hand-off 都必须包含：

```text
不得使用 develop/main/nightly
不得更换 host NVIDIA driver
不得复用 rambo51 venv
不得复用旧 Kit/shader/torch-extension cache
不得迁移到 Newton
不得修改 policy/checkpoint contract
不得修改 observation/action dimension
不得替换 qpth solver
不得在 gate failure 后继续下一 milestone
不得把 import success 当作迁移成功
```

每个 milestone 结束必须报告：

```text
Files changed
Exact versions
Commands run
Tests
GPU validation
Artifacts
Remaining risks
Git commit SHA
```

# 6. 推荐执行顺序

```text
M0  清理旧 venv/cache，保证足够磁盘
 ↓
M1  建立 rambo60 worktree，安装 exact target stack
 ↓
M2  官方 CUDA / Cartpole / Go2 / RGB / Kit GUI
 ↓
M3  输出完整 migration inventory
 ↓
M4  基础 Lab 3.0 API migration
 ↓
M5  qpth / QP
 ↓
M6  Quadruped
 ↓
M7  RGB
 ↓
M8  Button / keyboard
 ↓
M9  Biped
 ↓
M10 LingBot adaptation trajectory
 ↓
M11 Lock + Docker
```

# 7. 最终完成标准

```text
RTX 4090 + R595 runtime stable

Isaac Sim 6.0.1 stable
Isaac Lab v3.0.0-beta2.patch1 stable

Official Go2 PASS
RTX RGB PASS

qpth CUDA PASS
RAMBO QP PASS

Quadruped 405/18 contract preserved
Quadruped checkpoint strict load
Quadruped 3000-step PASS
Quadruped RGB PASS

Button loco-manip PASS
GUI keyboard PASS

Biped 435/18 contract preserved
Biped checkpoint strict load
Biped 3000-step PASS
Biped RGB PASS

30s dataset synthesis PASS

versions locked
migration report complete
Dockerized only after all runtime validation
```
