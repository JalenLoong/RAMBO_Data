# RAMBO Isaac Sim 6 / Isaac Lab 3 API 盘点（M3）

这是迁移计划指定位置的发布清单；根目录的
[`MIGRATION_API_INVENTORY.md`](../MIGRATION_API_INVENTORY.md) 保留面向运行手册的
摘要。本文额外保留 M3.2 所要求的逐项账本，因此今后以本文为审计主记录。

本清单基于迁移基线 `10154c6d2f20d33f51e5e126801c507e6fe206db` 与目标
Isaac Lab `v3.0.0-beta2.patch1` (`ffff603eafc6b74264a5261cc0183d6a65390d78`) 的
只读比对。最终 M3 clean-source 扫描绑定 RAMBO commit
`e449b1dd9f06ee3a4fc29338e5d917d12c8cce05`，证据目录为
`/workspace/migration-output/isaac60/M3/20260824T142802Z-api-inventory-clean/`；
该目录的 `checksums.sha256` 已复核。扫描找到的两个字面 quaternion 候选均为
已审计的 XYZW identity `[0,0,0,1]`，没有 `[1,0,0,0]` WXYZ identity 候选。
盘点期间没有启动 RAMBO 仿真，也没有选择或执行 Newton 后端。

除非本表明确列为 sealed runtime evidence，“runtime 通过”仅指此前保留的
技术 workload evidence；最终 runtime 状态以根目录的 `MIGRATION_STATUS.md` 为准。

## 不可协商的运行时合同

每个 RAMBO 配置和入口都必须在创建环境前执行下列等价操作：

```python
from isaaclab_physx.physics import PhysxCfg

cfg.sim.physics = PhysxCfg()
cfg.sim.use_newton_actuators = False
```

环境创建后、验证结束前和生产 rollout 的摘要中必须同时证明：

```text
isaaclab_physx.physics.physx_manager_cfg.PhysxCfg
isaaclab_physx.physics.physx_manager.PhysxManager
use_newton_actuators == false
```

`--viz` 只允许 `none` 或 `kit`；RAMBO 入口拒绝 `newton`、自定义 experience
和未审计 Kit 参数。官方依赖图按需加载的 `isaaclab_newton` 扩展或 Python 包不构成
失败；任何非 `PhysxManager` 的实际 RAMBO 运行构成失败。

## API 映射

| 范围 | 旧用法 | 目标实现 | 验收 |
|---|---|---|---|
| 四足/双足 cfg：`qp_env.py:373`、`:398` | `SimulationCfg` 依赖默认后端 | `physics=PhysxCfg()` 和 `use_newton_actuators=False` | 配置静态测试及 1-env PhysX runtime FQN |
| 所有 CLI / registry | 解析后没有固定后端 | 共享 `configure_physx()`；`gym.make` 后 `assert_physx_runtime()` | mock fail-closed 及实际入口摘要 |
| `utils/articulation.py:98-123` | `root_physx_view.get_jacobians/get_masses/get_inertias/get_coms` | `body_com_jacobian_w.torch`、`body_mass.torch`、`body_inertia.torch`、`body_com_pos_b.torch`/`body_com_quat_b.torch` | 含/不含 root Jacobian 行、Go2 名称排序 |
| `utils/articulation.py:127-141` | `body_state_w`、`find_bodies` | 必要时 `body_link_pose_w.torch`；ContactSensor 用 `find_sensors` | 传感器名称顺序与足端索引 |
| 两个 QP env 及 modules | 资产/传感器数值直接作 Torch 运算 | 每一个数值 `ProxyArray` 显式 `.torch`；`body_names`、`joint_names` 保持 Python list | 静态扫描、ProxyArray mock、1-env step |
| 两个 QP env reset | 旧 `write_root_*_to_sim`、`write_joint_state_to_sim` | `write_root_pose_to_sim_index`、`write_root_velocity_to_sim_index`、`write_joint_state_to_sim_index` | reset 后姿态、速度、关节值有限且生效 |
| `button_env.py` | `RigidObject.write_root_state_to_sim` | pose/velocity 两个 indexed writer；状态读 `.torch` | Button reset / press / release |
| `teleop_loco_manip.py` | 私有 command buffer 与旧 writer | 公共 `set_loco_manip_commands(...)` 和 indexed writer | 固定 seed 42 脚本序列 |
| 双足外力：`rambo_biped/qp_env.py:847-861` | 连续写 FL、FR 外力，后写可能覆盖前写 | 一次设置两个 body 的 wrench | 两只脚的 force 都非零 |
| 两个 QP env render | `has_gui()` / `has_rtx_sensors()` | `sim.has_gui` 属性和 `sim.is_rendering` | headless 与 RGB 渲染路径 |
| 两个 QP env reset | 私有 `_ALL_INDICES` | 显式 `torch.arange(num_envs, ...)` | DelayedDCMotor / reset index 类型 |
| ContactSensor | `net_forces_w_history.clone()` | `net_forces_w_history.torch.clone()`，并判空、验形状 | `N×history×sensor×3`、有限值 |
| RGB / rollout | RGB output 与 frame 当作 Tensor | `camera.data.output["rgb"].torch` 与 `camera.frame.torch` | 640×480、非黑、逐帧连续性 |
| `camera.py` 自定义时钟 | 修改 `_ALL_INDICES`、`_timestamp`、`_is_outdated` | 使用官方 Camera 的公开数据/帧号，并用 Warp int64 mask/clock 实现严格节拍 | 15000 physics ticks → 375 frames |
| 生命周期脚本 | `skip_cleanup=True`、`os._exit` | `env.close()` 后 `simulation_app.close(exit_code=...)` | 正常/异常各关闭一次、clean restart |
| debug 订阅 | 顶层 `omni.kit.app` 和三参 `set_debug_vis` | Kit 启动后延迟订阅；标准单参 API 另设 RAMBO 专用细分函数 | 静态 import 不启动 Kit、GUI smoke |

## M3.2 逐项迁移账本

下表满足计划要求的 `file`、`line`、`old API`、`new API`、`semantic risk`、
`required test`、`status` 字段。行号指向当前迁移分支；“旧 API”描述冻结的
Isaac Sim 5.1 / Isaac Lab 2.3.2 基线，而不是仍在目标树中执行的实现。

| 优先级 | file | line | 旧 API | 新 API | semantic risk | required test | status |
|---|---|---:|---|---|---|---|---|
| P0 | `source/rambo/rambo/tasks/direct/rambo_quadruped/qp_env.py` | 375–408 | `SimulationCfg` 默认物理、WXYZ `(1,0,0,0)` | `physics=PhysxCfg()`、`use_newton_actuators=False`、XYZW identity | 错误后端或把仿真姿态和 checkpoint 语义同时重排 | `test_physx_contract.py`；M4 四足 space contract | 已实现；M4/M6/M7 PhysX runtime 通过 |
| P0 | `source/rambo/rambo/tasks/direct/rambo_biped/qp_env.py` | 400–433 | 默认物理、WXYZ upright rotation | 显式 PhysX / 禁用 Newton actuators、XYZW upright rotation | biped 站立方向、关节/观测合同漂移 | `test_physx_contract.py`；M4 双足、M9 3000-step | 已实现；M4/M9 PhysX runtime 通过 |
| P0 | `source/rambo/rambo/utils/math.py` | 59–74 | simulator 与 QP 共用隐式 WXYZ | 仅边界 `xyzw_to_wxyz` / `wxyz_to_xyzw` | 405/435D policy、QP 与 checkpoint 被全局重排 | `test_quaternion_bridge.py`；M4 space contract | 已实现；contract 通过 |
| P0 | `source/rambo/rambo/utils/articulation.py` | 98–149 | `root_physx_view.get_*`、`body_state_w`、`find_bodies` | public `*.torch` ProxyArray、显式 link state、`find_sensors` | root Jacobian 行、Go2 body/foot 名称排序和 Tensor 语义 | `test_go2_articulation.py`；M4 四足 state finite | 已实现；静态与 runtime 通过 |
| P0 | `source/rambo/rambo/tasks/direct/rambo_quadruped/qp_env.py` | 1010–1028 | 旧 `write_root_*_to_sim` / 单一 root-state writer | `write_root_pose_to_sim_index`、velocity/joint indexed writers | reset 写入未生效或 env-id dtype 不兼容 | M4 四足 reset finite；`test_teleop_loco_manip.py` 静态 writer 断言 | 已实现；M4/M6 runtime 通过 |
| P0 | `source/rambo/rambo/tasks/direct/rambo_biped/qp_env.py` | 1067–1085 | 旧 root/joint writer | 相同 indexed writer contract | 双足 reset 与 435/18 空间合同漂移 | M4 双足、M9 3000-step | 已实现；M4/M9 runtime 通过 |
| P0 | `source/rambo/rambo/tasks/direct/rambo_quadruped/button_env.py` | 182–222 | `RigidObject.write_root_state_to_sim`、隐式 state array | button cap pose/velocity indexed writers 和 `.torch` state | 弹簧 cap 重置、按压/回弹物理错误 | M8 button smoke；M10 `--require-acceptance` | 已实现；M8/M10 runtime 通过 |
| P0 | `source/rambo/rambo/tasks/direct/rambo_biped/qp_env.py` | 846–884 | 分别写 FL/FR wrench，后写覆盖前写 | 合并两个 body-id 的单次 `set_forces_and_torques_index` | 前脚命令不独立、QP 接触语义失真 | `test_physx_biped_policy_smoke.py`；M9 policy rollout | 已实现；M9 runtime 通过 |
| P1 | `source/rambo/rambo/tasks/common/camera.py` | 248–294 | 默认 renderer、私有 camera timestamp mutation | `CameraCfg` + `IsaacRtxRendererCfg`、640×480、公开更新路径 | RGB 非黑、相机朝向、375-frame cadence | `test_validation_helpers.py`；M7/M9 RGB | 已实现；M7/M9 runtime 通过 |
| P1 | `source/rambo/rambo/tasks/direct/rambo_{quadruped,biped}/qp_env.py` | 430–436 / 455–466 | ContactSensor 默认数据、相机 offset WXYZ | 显式 `ContactSensorCfg`、相机 XYZW offset | 接触历史 shape、biped camera parent-frame pose | `test_go2_articulation.py`；M7/M9 RGB | 已实现；runtime 通过 |
| P1 | `source/rambo/rambo/utils/physx.py` | 28–97 | launcher/registry 可继承默认 manager | 创建 env 前强制 PhysX；创建后 exact FQN fail-closed | 仅配置看似正确而实际运行非 PhysX | `test_physx_contract.py`；所有 accepted runtime summary | 已实现；全部 RAMBO runtime 通过 |
| P1 | `scripts/rambo/{smoke,validate,play,teleop_loco_manip}.py` | 各 parser / `gym.make` 后 | `--headless` / 自定义 Kit 参数可改变路径 | 只接受显式 `--viz none|kit`，拒绝 experience/kit args，前后 PhysX 断言 | GUI/launcher 偷换 backend 或跳过关闭 | `test_static_imports.py`、`test_physx_contract.py`；M6–M10 | 已实现；headless runtime 通过；人工 GUI 待验收 |
| P1 | `scripts/reinforcement_learning/crl2/{train,play}.py` | 14–84 / 14–68 | Kit 前无启动参数 fail-closed guard，直接使用旧 torch helper | 先拒绝未审计 visualizer，再延迟 AppLauncher/Isaac Sim import | import 副作用、训练/回放入口绕过 PhysX guard | `test_static_imports.py`；入口静态审计 | 已实现；静态通过 |
| P1 | `source/rambo/rambo/tasks/common/camera.py` | 208–239 | 修改 `_ALL_INDICES`、`_timestamp`、`_is_outdated` | official Camera public update + Warp integer cadence | frame 丢失、首帧和 40 physics-tick 对齐错误 | M7 RGB 3000-step / 375 frames | 已实现；M7 runtime 通过 |
| P2 | `source/rambo/rambo/actuators/delayed_dc_motor.py` | reset paths | 私有 `_ALL_INDICES` | 目标-device `torch.arange` 的显式 indices | delayed actuator reset 目标错位 | `test_go2_articulation.py`；M4 reset | 已实现；M4 runtime 通过 |
| P2 | `source/rambo/rambo/utils/markers.py` 及 debug callers | 顶层 `omni.kit.app`、旧三参 debug API | Kit 后延迟订阅，RAMBO 专用 adapter | metadata import 意外启动 Kit、GUI debug 崩溃 | `test_static_imports.py`；GUI smoke | 已实现；静态通过；真实 GUI 人工验收待完成 |

## 四元数边界

目标 Isaac Lab 3 的 simulator-facing 四元数和 `isaaclab.utils.math` 均为
**XYZW**。因此：

- 四足初始姿态从 `(1, 0, 0, 0)` 迁为 `(0, 0, 0, 1)`；
- 双足初始姿态迁为 `(0, -sqrt(0.5), 0, sqrt(0.5))`；
- biped camera offset 迁为 `(0, sqrt(0.5), 0, sqrt(0.5))`，一般相机 identity
  迁为 `(0, 0, 0, 1)`；
- USD `Gf.Quatf` 仍是标量优先，不在该处转换；
- RAMBO 的 QP/checkpoint 数学中保留原有 WXYZ 语义。只在该内部边界使用显式
  `xyzw_to_wxyz` / `wxyz_to_xyzw` 适配；不得对 policy、checkpoint 或 QP 向量
  做全局重排。

## 打包、版本与入口

| 位置 | 当前问题 | 迁移动作 |
|---|---|---|
| `source/rambo/setup.py` | Python `<3.12`、`numpy<2` | 改为目标 Python 3.12 / NumPy 2.3.1 兼容 metadata |
| `source/crl2/setup.py` | 宽依赖会覆盖固定 Torch/NumPy | target venv 以 `--no-deps` editable 安装 |
| `source/rambo/config/extension.toml` | 未显式声明 PhysX | 增加 `isaaclab_physx` 依赖 |
| `scripts/rambo/run.sh` | 默认旧 rambo51/Python 3.11 | 默认 rambo60/Python 3.12，保留 CUDA 12.8 运行库检查 |
| `scripts/rambo/{smoke,validate,play,teleop_loco_manip}.py` | API、物理选择及退出路径均旧 | 统一 PhysX contract、限制 visualizer、迁移 ProxyArray/writer/lifecycle |
| `requirements/isaacsim51.*` / `scripts/setup_isaacsim51.sh` | 旧栈资料 | 不机械升级；M11 新建 Isaac Sim 6 pinned manifest/installer |

## M4–M10 验收顺序

1. 共享 PhysX 合同与 CLI 限制；所有 RAMBO cfg/入口显式 PhysX。
2. XYZW/ProxyArray、public articulation/writer/ContactSensor 和双足双外力迁移。
3. 1-env 四足、双足、Button 各 10+ step 的 PhysX-only 有限数回归。
4. qpth 0.0.18 CPU/CUDA forward/backward、gradcheck、KKT、residual、determinism。
5. 四足/双足 checkpoint strict load、观测/动作契约和 3000-step 运行。
6. RGB 30 秒、375 帧、每 40 physics ticks、非黑和 clean restart。
7. Button press/hold/release 与 recorder/dataset schema 完整性。

## 已知不属于失败的事项

- Isaac Sim 6.0.1 精确依赖图中的 `newton` / `isaaclab_newton` 包；
- 固定 Kit GUI experience 的 `isaaclab_newton` extension 日志行，前提是每次 RAMBO
  runtime summary 仍证明实际 manager 为 `PhysxManager`。
