# RAMBO → Isaac Sim 6.0.1 / Isaac Lab 3.0 Beta 2 Patch 1：执行计划

本文件是长时间自主执行的恢复点。权威行为契约仍以
`rambo-isaacsim60-isaaclab30b2-adaptation-plan.md` 为准；本文件补充已核实的
精确依赖事务、PhysX 门禁、命令顺序、产物和恢复规则。

## 0. 不可变约束

- 目标仅为 Isaac Sim `6.0.1.0` 与 Isaac Lab
  `v3.0.0-beta2.patch1`（commit
  `ffff603eafc6b74264a5261cc0183d6a65390d78`）。禁止 `main`、`develop`、nightly
  或任何其他 Isaac Sim/Lab 版本。
- RAMBO 运行时只能使用 PhysX。每个 RAMBO 配置、官方 smoke、验证、数据记录和
  生产启动都必须显式设置 `physics=PhysxCfg()`，创建 simulation 后必须验证
  `isaaclab_physx.physics.PhysxManager`。任何实际的 Newton/OvPhysX manager 是失败。
- Isaac Sim 的依赖图带来的 Newton 包允许存在；精确 tag 任务 import 所需的裸
  `isaaclab_newton` 包允许存在。禁止安装 `isaaclab_newton[all]`、`newton[sim]`、
  `isaaclab_physx[newton]` 或任何 Newton optional extra。
- 不改变 405/435 observation、18D action、action scale `5.0/0.15`、joint/body/
  Jacobian ordering、QP 目标与约束、checkpoint/normalizer 语义、Button/teleop 语义。
- 验证 checkpoint SHA256：quadruped
  `1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706`；biped
  `c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6`。
- `physics_dt=0.002`、decimation=5、control=100 Hz；3000 control steps 必须对应
  15000 physics ticks 与 375 个 640×480 RGB frames。
- 用户已在本线程明确接受 EULA。原生 pip 运行仅对需要它的进程以
  `OMNI_KIT_ACCEPT_EULA=Y` 传递，不持久化到 shell、仓库、镜像或
  `EULA_ACCEPTED` 文件。`PRIVACY_CONSENT` 未获授权，不设置。

## 1. 路径、日志和提交纪律

- 旧参考 checkout：`/workspace/rambo`，禁止改写。
- 迁移 checkout：`/workspace/rambo60`，branch `adapt/isaacsim60-isaaclab30b2`。
- 旧基线 tag：`isaac51-golden-10154c6` →
  `10154c6d2f20d33f51e5e126801c507e6fe206db`。
- Isaac Lab checkout：`/workspace/IsaacLab-3.0.0-beta2.patch1`。
- target venv：`/workspace/venvs/rambo60`。
- 日志和输出：`/workspace/migration-output/isaac60/M<n>/<UTC>-<slug>/`。预设的
  `/workspace/migration_logs` 对当前执行用户不可写，因此不再依赖它。每次运行新建目录，
  禁止覆盖；保存 command manifest、exit code、backend evidence、GPU evidence、summary
  及 SHA256 manifest。

每个成功 milestone：更新 `MIGRATION_STATUS.md` → 测试 → commit → 尝试普通 push。
禁止 force-push、reset/rebase 旧分支或丢弃未解释改动。push 无权限时记录未推送 commit，
本地技术工作继续。

## 2. M0：空间和主机预检

1. 记录 `df`、GPU/driver、Vulkan、RAM/swap、Python/uv/git、Docker/Toolkit 状态到
   `host-manifest.json`。
2. 删除前必须验证 `realpath`、`lstat`、非 mountpoint 与目录大小；仅删除已授权且损坏的
   `/workspace/venvs/rambo51`。可逐项删除可重建 pip/uv/ComputeCache；绝不删除
   `isaac45-test`、`torch-cu128`、policy、WAM、dataset、旧源码或参考输出。
3. 删除后可用空间硬下限为 69 GiB；70 GiB 是目标值。当前已核实的授权删除预计释放到约
   69.8 GiB，因此不能为了不足 0.2 GiB 删除未授权数据。安装 `isaacsim` 前后记录可用空间，
   且任一阶段低于 20 GiB 时先只清理已核实 cache；仍不足才停止并请求用户。

## 3. M1：精确依赖事务

创建 Python 3.12 venv 后执行：

```bash
uv pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match --prerelease=allow
uv pip uninstall torch torchvision torchaudio
uv pip install "torch==2.10.0" "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu128
```

从精确 tag editable 安装且不加 Newton extras：`isaaclab`、`isaaclab_ppisp`、
`isaaclab_contrib`、`isaaclab_assets`、裸 `isaaclab_newton`、裸 `isaaclab_ovphysx`、
`isaaclab_physx`、`isaaclab_tasks`、`isaaclab_visualizers[kit]`。不安装
`isaaclab_rl`、Mimic、Teleop、experimental。

该 tag 的 `isaaclab_tasks` 在运行时直接 import Hydra、但最小 package metadata 未声明它；
因此补装普通依赖 `hydra-core==1.3.2`（连带 `omegaconf==2.3.1`）。这不是 Newton extra。

安装 `qpth==0.0.18 --no-deps`，以及 CRL2/RAMBO `--no-deps` 前先完成其 Python 3.12/
NumPy 2 metadata 改造。`pip check` 仅允许 Isaac Sim metadata 与官方 Torch override、
缺失 torchaudio、Isaac Lab base 的 `coverage==7.6.1` 与 Isaac Sim kernel `7.4.4` 差异，
以及 qpth 的 NumPy/CVXPY 已知声明差异；其余冲突失败并诊断。

## 4. M2：先证官方栈，再证 RAMBO

M2 的脚本不得 import RAMBO。依次运行 CUDA tensor、有限 Cartpole、
`Isaac-Velocity-Flat-Unitree-Go2-v0`、RTX RGB、`--viz kit` GUI 和 clean restart。
每次都在构造环境前设 `PhysxCfg()`、明确设
`SimulationCfg.use_newton_actuators=False`，构造后及结束前验证 config FQCN、活跃
manager FQCN（必须为 `PhysxManager`）与该开关，并写入 summary。GPU evidence 必须证明 Kit
使用 Vulkan GPU0 RTX 4090，而非 llvmpipe。

固定 Isaac Sim 6.0.1 的默认 `fast_shutdown=True` 必须保留：其 `False` 全扩展 teardown 路径
在本 host 上会在工作成功后崩溃。脚本须在调用 `simulation_app.close(exit_code=...)` 前写入并
flush summary；外层 exit code 与随后独立的 restart 共同证明关闭成功。RAMBO 代码不得直接
调用 `os._exit` 或 `skip_cleanup`。首次 RTX shader-cache 冷启动可超过 60 秒；缓存后的独立
restart 必须在 60 秒内通过。

## 5. M3–M5：API 与 QP

M3 生成完整 API inventory，覆盖所有 ProxyArray、root PhysX view、writer、ContactSensor、
四元数、camera、lifecycle、omni import、actuator、CRL2、Button、teleop 与 validator；每项
必须有目标 API 和测试。

M4 规则：

- 数值 ProxyArray 一律 `.torch`，names 仍是 Python list。
- 用公开 `body_com_jacobian_w`、mass/inertia/COM 数据替代 root PhysX view；测试含/不含
  floating-root row 的 Jacobian。
- 拆分 indexed root pose/velocity 与 joint position/velocity writers；Button 同样拆分。
- ContactSensor 使用 `find_sensors` 和 `.torch` output。
- simulator-facing quaternion 改 XYZW；Biped 初始姿态
  `(0,-sqrt(.5),0,sqrt(.5))`、camera `(0,sqrt(.5),0,sqrt(.5))`。
- render lifecycle 用 `has_gui` property、`is_rendering`/`render_enabled` 和缓存 RTX-sensor
  标志；固定跑五个 PhysX step/control step。
- camera cadence 采用 Warp int64 physics ticks 和 reset mask；禁止 float drift。

M5：先验证 qpth CPU/CUDA forward/backward、CPU gradcheck、analytic KKT、CUDA residual 和
determinism；再验证 RAMBO QP shape、ordering、friction/torque constraints。若 qpth 必须补丁，
只 vendor 精确 0.0.18 并保留算法；更换 solver 或 QP 语义是硬阻塞。

## 6. M6–M10：运行验收

- M6 Quadruped：hash + strict load、405/18、1-env 16-step、16-env smoke、1-env 3000-step
  camera-off。无 NaN/done/reset/torque violation；height ≥0.1，orientation error ≤0.75。
- M7 RGB：1-env 3000-step，375 连续 frames，640×480，timestamps `0.08…30.00`，每 40
  physics ticks 一帧；非黑、变化、camera projection 与 contact sheet 均证明 scene/robot 可见。
- M8 Button：暴露 `set_loco_manip_commands(base_velocity, fl_position, fl_force, env_ids=None)`；
  使用固定 seed 42 步序（110–269 前进、300–549 FL z、550–809 FL x、810–909 retract）。
  要求 12 mm 连续 5 steps、回弹 ≤2 mm、步行与关节变化均 ≥0.05。GUI 用 xdotool/wmctrl
  自动化；仅在无法自动化时把真正人工键盘检查保留到最后。
- M9 Biped：435/18、strict load、FL/FR 与 contact override 独立性、phase offset 19.6s、
  16-step/16-env/3000-step/RGB；height ≥0.3，orientation error ≤0.8。
- M10：新 recorder 生成 Button task 的 30s episode，3000 observation/action/post-state、375
  RGB 和 terminal record；保存 prompt、seed、versions、GPU/driver、checkpoint hashes、PhysX
  evidence、schema version、timestamps 和 checksums。button success/rebound、无缺样本、时间
  单调且 action/state/RGB 对齐是硬 gate。

所有 3000-step run 在 step 1000 后监控 GPU allocated memory：斜率 ≤1 MiB/100 steps 且首末
500-step median 差 ≤128 MiB；RSS 对应 ≤4 MiB/100 steps 和 ≤512 MiB。每关节速度不能超过
解析 actuator simulation limit 加 1% 容差。

## 7. M11、分类和恢复

冻结 `requirements/isaacsim60.in`、lock/freeze、setup/run scripts、host/checkpoint/package
manifests、migration report 和 final matrix。run wrapper 仅允许 `--viz none|kit` 并强制 PhysX
assert。重新跑最小官方 smoke、Quadruped、Button、Biped、RGB、dataset validator 和双 clean
restart 后提交。

Docker 只在 native 全通过后处理。Docker/Toolkit 缺失不否定 native 成功；交付 pinned Dockerfile
与命令并记录 deferred。容器运行也只能 PhysX，且仅传入已授权的 `OMNI_KIT_ACCEPT_EULA=Y`。

诊断层级固定：官方 smoke 失败=Isaac stack；manager/physics 失败=PhysX；headless 通过而
renderer 失败=RTX；API 单测/实例化失败=Lab migration；独立 QP 失败=qpth；其余 strict-load/
rollout 失败=RAMBO policy。普通失败自行修复并重试。只有 EULA、credential、额外破坏性删除、
checkpoint hash、宿主不兼容、pinned release 不可用、受保护语义改变和无法自动化的最终人工 GUI
检查才可请求用户。
