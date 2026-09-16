---
id: CURRENT-DATASET-V2-1
type: current
status: accepted
source_map:
  - source/rambo/rambo/dataset_v2/spec/schemas.json
  - source/rambo/rambo/dataset_v2/spec/profile.json
  - source/rambo/rambo/dataset_v2/validation.py
  - source/rambo/rambo/dataset_v2/media.py
  - source/rambo/rambo/dataset_v2/finalization.py
---
# adaptation_v2_dataset_contract.md

# WAM-Policy Adaptation v2 Dataset Contract

> Work ID: `DATA-002` in the independent adaptation-v2 namespace.

> Dataset schema: `wam-quadruped-v2.1.0`; LeRobot codebase: `v2.1`; library: `lerobot==0.3.3`.
> DATA-002原验收范围：contract/schema、最小CPU读取/校验/映射适配与人工测试。
> DATA-004已验证一条真实50Hz/25Hz替代technical pilot及完整模型缓存链路；本页末尾记录当前结果。大规模生产及训练仍未运行。
> 原PNG数据契约2.0.0只用于历史审计；同步runtime协议不因落盘格式变化自动升级。

## 0. Purpose

本文档定义 **WAM-Policy Adaptation v2** 的数据采集、组织、打包、存储与模型侧派生缓存规范，用于在 `RAMBO_Data` 与 `WAM-Policy` 两个仓库之间建立稳定、可复现、可审计的数据边界。

该 contract 覆盖以下模态与元数据：

- RGB / multi-camera video
- 9D high-level action
- Language / text
- Robot proprioceptive state
- RAMBO controller telemetry
- Physics / contact telemetry
- Task / object state
- Episode metadata
- Asset provenance
- Camera calibration
- Multi-rate timestamp synchronization
- Canonical LeRobot packaging
- Train / validation / test splitting
- WAM-Policy model-specific latent / text / normalization cache

本版本的核心原则是：

```text
Raw Acquisition
      ↓
Canonical Robot Dataset
      ↓
LingBot Model Cache
```

三层必须严格分离，不允许将 simulator telemetry、canonical robot data 和 model-specific cache 混为一套数据。

---

# 1. Scope and Non-goals

## 1.1 Scope

本 contract 适用于当前 Adaptation v2 主线：

- Embodiment: quadruped only
- Manipulation leg: front-left (FL) only
- High-level action: 9D RAMBO-compatible command
- Policy cameras:
  - ego
  - task-centric
- Additional monitor camera:
  - observer
- Policy action frequency: 50 Hz
- Model RGB frequency: 12.5 Hz
- RAMBO control frequency: 100 Hz
- Isaac physics frequency: 500 Hz

## 1.2 Non-goals

本 contract 当前不解决：

- biped mode
- dual manipulation legs
- explicit proprioception as LingBot input
- non-zero desired EEF force learning
- learned bridge
- new WBC / manipulation-centric controller
- OPD / AnyFlow training data
- real-robot data collection
- tactile/audio modalities
- large-scale domain randomization

这些内容如需加入，必须通过新的 schema / dataset version 进行，不允许静默扩展当前 contract。

---

# 2. Repository Ownership

`RAMBO_Data` 对“机器人和仿真中实际发生了什么”负责；`WAM-Policy` 对“模型如何消费这些数据”负责。

| 内容 | RAMBO_Data | WAM-Policy |
|---|:---:|:---:|
| Isaac environment / assets | ✓ | |
| RAMBO controller | ✓ | |
| Task implementation | ✓ | |
| Teleoperation / expert trajectory | ✓ | |
| RGB / state / action recording | ✓ | |
| Raw multi-rate archive | ✓ | |
| Success annotation | ✓ | |
| Asset provenance | ✓ | |
| Canonical LeRobot dataset | ✓ | |
| Dataset validation / release | ✓ | |
| 9D action interface definition | shared | shared |
| LingBot dataset reader | | ✓ |
| 50 → 12.5 Hz video sampling | | ✓ |
| Multi-view latent composition | | ✓ |
| Wan VAE latent extraction | | ✓ |
| T5 / text embedding | | ✓ |
| Action normalization | | ✓ |
| SFT / inference / evaluation | | ✓ |

### Hard boundary

`WAM-Policy` 不应再包含：

```text
Isaac task implementation
teleoperation code
raw dataset recorder
asset importer
dataset release publisher
RAMBO controller implementation
```

`RAMBO_Data` 不应包含：

```text
Wan VAE latent cache
T5 embeddings
LingBot training windows
model normalization cache
SFT checkpoints
```

---

# 3. Three-layer Data Architecture

## 3.1 Layer A — Raw Acquisition Archive

Owner: `RAMBO_Data`

特性：

- 保留 native multi-rate streams
- 尽量保留原始物理量
- 不做模型 normalization
- 不强行将所有 modality 对齐到单一 FPS
- 以 authoritative simulation timestamp 对齐
- 用于 replay / debug / audit / future reprocessing

## 3.2 Layer B — Canonical Robot Dataset

Owner: `RAMBO_Data`

当前 training-facing format 优先采用 **LingBot-VA 当前官方 pipeline 容易直接消费的 LeRobot v2.1-style / `lerobot==0.3.3` compatible organization**。

特性：

- 50 Hz canonical timeline
- two policy RGB streams
- executed 9D action
- requested 9D action
- 50 Hz robot state snapshot
- task/object state
- raw UTF-8 language
- episode metadata
- LingBot-compatible `action_config`

Canonical dataset 是正式、可版本化、不可静默覆盖的数据产品。

## 3.3 Layer C — LingBot Model Cache

Owner: `WAM-Policy`

包括：

```text
Wan causal VAE latents
T5/text embeddings
action normalization statistics
sampled frame IDs
training windows / segment manifests
model-specific metadata
```

必须保证：

> 删除 Layer C 后，能够仅依赖 Layer B 中的数据，以及固定的模型/预处理/数据划分版本，重新生成全部 model-specific cache。Terminal RGB/state随canonical保留；不得隐式读取Raw才能补齐cache。

---

# 4. Global Time and Canonical Row Contract

所有stream以单一monotonic simulation clock的`simulation_time_ns:int64`为权威时间；reset完成后为0。
wall-clock仅用于性能和超时，不用于物理对齐。Physics step/controller tick/policy tick用于一致性检查。
同一stream时间严格递增；不同stream在同一采样时刻可以具有相同timestamp。

| Stream | Rate | Period |
|---|---:|---:|
| Physics | 500 Hz | 2 ms |
| RAMBO controller | 100 Hz | 10 ms |
| High-level action | 50 Hz | 20 ms |
| Ego/task raw RGB | 50 Hz | 20 ms |
| Ego/task canonical RGB | 50 Hz | 20 ms |
| Observer | 25 Hz | 40 ms |
| Model RGB | 12.5 Hz | 80 ms |

## 4.1 Row k 是动作前观测与后续动作区间

```text
t_k: sample RGB_k / state_k
  -> obtain requested_action_k
  -> currently enabled execution transform chain
  -> submitted high-level executed_action_k
  -> hold over [t_k, t_k+20ms), two RAMBO control steps
  -> confirm completed interval
t_{k+1}: sample next RGB/state
```

`observation_k/state_k`是动作执行前状态。`action.requested_k`基于该观测产生。
`action.executed_k`是最终提交给RAMBO并保持的50Hz高层命令；不是100Hz低层输出或500Hz物理结果。
数值在提交前确定，完整区间的执行确认在20ms后产生。只有确认完成的区间进入有效训练action/history。
部分执行保存实际结束时间和trace，不伪装为完整20ms标签。

## 4.2 N/N+1 和 terminal

Raw必须保留N个完整actions和N+1个boundary observations/states。
Canonical主体保留N行、两路各N帧主视频，paired range为[0,N)。不生成terminal action。
Terminal metadata记录第N个边界的simulation timestamp、physics step和state；canonical另附两路terminal PNG快照。
快照由已验证Raw MP4的第N帧解码导出，保存source video hash/source frame index/timestamp；无额外resize或增强。
这两张快照不是新的optional lossless archival mode，也不意味着压缩前exact pixels被保留。

Canonical的普通LeRobot读取只遍历N行；WAM可通过terminal metadata读取额外边界图像。
最后一个action是否进入某个训练窗口由合法future visual target和VAE grouping决定，不能据此删改Raw/Canonical。
DATA-004替代pilot已实测50Hz采集及terminal-before-reset；不得通过复制12.5Hz帧伪装50Hz采集。

---

# 5. Camera Contract

## 5.1 Camera Keys

Policy cameras:

```text
observation.images.ego
observation.images.task_centric
```

Monitor-only camera:

```text
monitor.images.observer
```

## 5.2 Roles

| Camera | Raw Rate | Canonical LeRobot | LingBot Input | Inference Input |
|---|---:|:---:|:---:|:---:|
| `ego` | 50 Hz | ✓ | 12.5 Hz | ✓ |
| `task_centric` | 50 Hz | ✓ | 12.5 Hz | ✓ |
| `observer` | 25 Hz | ✗ | ✗ | ✗ |

### Ego

负责：

- locomotion direction
- obstacle / target relation
- global scene context

### Task-centric

负责：

- FL manipulation leg
- target object
- contact region
- local interaction geometry

### Observer

固定 elevated oblique third-person camera，用于：

- human review
- debugging
- failure analysis
- paper/demo video

Observer camera:

- MUST NOT be enumerated as policy observation
- MUST NOT enter LingBot training
- MUST NOT enter LingBot inference
- MUST NOT be used as task success ground truth

---

# 6. RGB Storage and Encoding Contract

连续视频使用明确的有损MP4编码；不得将其描述为“无损Raw RGB”。这里Raw表示未经模型预处理的采集视频。

## 6.1 Policy video

| Parameter | Fixed value |
|---|---|
| Container | MP4 |
| Codec/encoder | H.264/AVC, libx264 |
| Rate | 50 Hz CFR |
| Pixel format | yuv420p |
| Primaries / transfer / colorspace | BT.709 |
| Range | limited/tv; explicit RGB full-range to BT.709 limited YUV conversion |
| Quality | high-quality lossy, CRF 18 |
| Preset | medium |
| GOP / minimum keyframe interval | 50 / 50 |
| Scene-cut keyframe insertion | disabled |

不增加未批准的resize、颜色增强、filter或clamp。编码所需颜色表示转换及参数必须进入manifest。
Observer使用MP4/H.264、25Hz CFR、yuv420p；CRF默认23，允许显式20–23并记录，不额外冻结observer GOP/preset。
如未来需要exact-pixel原始归档，另增版本化optional模式；当前默认保持有损。

## 6.2 Camera index and PTS

`camera_index.parquet`字段：camera_key:string、frame_index:int32、simulation_time_ns:int64、physics_step:int64、valid:bool。
Raw policy视频/index包含N+1个边界；canonical主视频/index包含N帧，terminal单独引用。
权威映射是`frame_index -> simulation_time_ns`；MP4 PTS仅用于播放/seek与一致性检查。
Canonical视频以零起点CFR PTS供LeRobot按episode-relative timestamp查询，不能反过来以PTS替代仿真时钟。

校验decoded frame count、显示帧顺序、CFR、ffprobe/decoder完整读取、camera_index与视频数量及仿真schedule。
编码receipt记录实际command、encoder/version、CRF、preset、GOP和颜色配置；文件hash绑定载荷。
CRF/preset来源于实际编码命令及receipt，不声称仅凭ffprobe能反推出所有编码参数。

## 6.3 Completion and PNG policy

先写`*.partial.mp4`；正常close、完整validation成功后在同一文件系统atomic rename为最终`.mp4`。
不得覆盖已有最终文件。失败保留partial、episode标记invalid，不修复后悄悄进入正式dataset。
当前提供single-writer辅助逻辑，未接生产recorder。

PNG仅用于asset/calibration/debug/contact/success/terminal快照，禁止将连续训练视频主要保存为PNG序列。
LeRobot测试工具生成的临时PNG会在编码后清理，它们不是正式Raw或Canonical产品。

---

# 7. Native 9D and Explicit Name Mapping

Action为float32[9]、50Hz，保留全部force通道，不改成6D。

| Index | RAMBO source field | Canonical name | Model name/index |
|---:|---|---|---|
|0|_velocity_commands[0]|base_vx|base_vx / 0|
|1|_velocity_commands[1]|base_vy|base_vy / 1|
|2|_velocity_commands[2]|base_yaw_rate|base_yaw_rate / 2|
|3|_ee_pos_commands[0]|fl_ee_x|fl_ee_x / 3|
|4|_ee_pos_commands[1]|fl_ee_y|fl_ee_y / 4|
|5|_ee_pos_commands[2]|fl_ee_z|fl_ee_z / 5|
|6|_ee_force_commands[0]|fl_force_x|fl_ee_fx / 6|
|7|_ee_force_commands[1]|fl_force_y|fl_ee_fy / 7|
|8|_ee_force_commands[2]|fl_force_z|fl_ee_fz / 8|

单位依次为m/s、m/s、rad/s、m、m、m、N、N、N。
保留`rambo-native9-v2`语义身份，名称映射不改变物理量，也不是30D桥接。

经现有RAMBO代码审计，速度使用`projected_com_controller`，FL位置/力使用`projected_com`。
该frame沿用重力投影旋转矩阵，位置原点为base world x/y投影到world z=0、当前com_offset=0；不是在线估计的整机质心。
FL reference为现有解析运动学足端点，目标是absolute position；z不是相对base高度。
不新增坐标旋转或重定向来“修正”现有controller。Quaternion元数据明确XYZW。

## 7.1 Camera source -> canonical -> model

| Runtime registry / alias | Source identity | Canonical key | Model role / existing Python key |
|---|---|---|---|
|front_camera / ego|go2_ego|observation.images.ego|ego / go2_ego|
|task_camera / task|d435i_rgb_task|observation.images.task_centric|task_centric / d435i_rgb_task|
|configured later|configured observer source|monitor.images.observer|NEVER_MODEL_INPUT / none|

Model role与Python输入key分开；阶段2的`go2_ego/d435i_rgb_task`输入保持不变。
Observer source key在后续实现中配置；本轮不创建或选择新的物理机位。它不能进入`observation.images.*`。

---

# 8. Requested, Execution Transform Chain and Confirmation

```text
requested_action -> explicitly enabled execution transform chain -> submitted_action
                 -> RAMBO hold -> execution confirmation -> executed history
```

Raw每个policy tick保存`action.requested`和`action.executed`；canonical训练列`action=action.executed`。
它们都是高层9D命令，不能用18D residual、关节目标或实际足端运动替换。

当前代码的精确顺序：

1. 标识/时间/9D shape、numeric dtype、finite及float32可表示范围检查；
2. 保留requested值；
3. 转成现有setter所需float32表示；
4. baseline force维6:9清零；
5. 已确认的identity/no-op高层filter；
6. 提交并保持两个100Hz控制步；
7. 区间完成后生成执行确认。

源数据按float32 schema保存；既有helper中的表示转换不构成新增限幅或平滑。
未来新增bounds/clamp/filter必须参数化、固定顺序、写config及transform/version hash，记录修改来源。
本版本没有新增workspace bounds、safety clamp、validity correction或command smoothing。
数据中executed与requested的修改必须能追溯到明确chain；当前canonical验证允许的值变化仅为force清零。
保留ordered-chain hash和force modification flag；不得以训练方便为由静默修改命令。

原controller内部的residual裁剪、IK、关节滤波/限幅保持原样，不属于新增高层链。
Residual observation和FL IK使用的source command ID/消费时间继续作为独立trace；不混成另一条9D history。

---

# 9. Force and Wrench Semantics

严格区分desired EEF force、实际contact measurement、QP force estimate和external injected wrench。
Baseline desired force维6:9为零；external force与external torque也均为零。接触力可以非零。

External wrench正式为6D：`external_force:float32[3]`、`external_torque:float32[3]`，必须记录frame及既有作用点信息。
当前旧控制路径会将negative desired force转换到FL body-local并作用于该body CoM；本轮不改动该路径。

经当前IsaacLab PhysX源码审计：

- `net_forces_w`：world frame下按body聚合的net normal contact force，不含独立friction字段。
- `force_matrix_w`：按配置body pair过滤的normal contact force；不等同于任意单接触点force。
- `friction_forces_w`：另行启用的切向/摩擦力总和。
- 当前ContactSensorData没有直接torque输出；缺失时标为不可用，不推导或填假零测量。
- `get_net_contact_forces(dt=physics_dt)`沿用原API调用；不额外做average/filter。
- history窗口或逐分量peak不应标成一次瞬时测量。

`actual_contact_force`在本profile绑定上述world-frame net normal量；对象pair接触必须明确其过滤配置。
proximity+net-force只能是推断指标，不得作为已确认FL-object接触。

---

# 10. Language / Text Contract

## 10.1 Task-level Text

标准 task instruction：

```text
task.text
```

例如：

```text
"Push the box into the target area."
```

这是主要 policy language condition。

## 10.2 Episode-level Text

每个 episode 保存：

```json
{
  "tasks": [
    "Push the box into the target area."
  ]
}
```

## 10.3 LingBot `action_config`

每个 episode 必须带 LingBot-compatible `action_config`。

Baseline 默认：

```text
one episode = one action segment
```

示例：

```json
{
  "episode_index": 0,
  "tasks": [
    "Push the box into the target area."
  ],
  "length": 412,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 412,
      "action_text": "Push the box into the target area."
    }
  ]
}
```

未来只有在确实需要 long-horizon subtask supervision 时才引入：

```text
approach object
make contact
push object
...
```

等多 segment annotation。

## 10.4 Text Embedding

Canonical dataset 只保存：

```text
UTF-8 raw text
```

禁止把以下内容当作 canonical data：

```text
T5 embedding
Wan text embedding
CLIP embedding
```

这些全部属于 WAM-Policy derived cache，必须位于独立cache_root，不写入RAMBO canonical tree。

---

# 11. Robot State Contract

Adaptation v2 baseline 暂时不把 proprioception 作为 LingBot input，但必须完整采集。

## 11.1 Canonical 50 Hz Snapshot

每个 policy tick 保存：

```text
observation.state.base.position          float32[3]
observation.state.base.orientation       float32[4]

observation.state.base.linear_velocity   float32[3]
observation.state.base.angular_velocity  float32[3]

observation.state.projected_gravity      float32[3]

observation.state.joint_position         float32[N_joint]
observation.state.joint_velocity         float32[N_joint]

observation.state.fl_ee_position         float32[3]
observation.state.fl_ee_velocity         float32[3]

observation.state.foot_contact           bool[4]

observation.state.fl_contact_force       float32[3]
```

当前：

```text
Canonical dataset: 保存
LingBot v2 input: 暂不使用
```

这样未来若需要 proprioception-conditioned variant，不需要重新采数据。

---

# 12. Controller Telemetry

RAMBO controller:

```text
100 Hz
```

属于 raw/audit modality，不进入当前 canonical 50 Hz training table。

推荐：

```text
controller_100hz.parquet
```

至少记录：

```text
timestamp_ns
controller_tick

high_level_command
reference_state
policy_residual
desired_joint_target
desired_joint_velocity
desired_joint_torque
safety_flags
controller_mode
```

最终字段以 RAMBO runtime 能稳定导出的 telemetry 为准。

---

# 13. Physics State and Sensor-native Contact Streams

`physics_500hz.parquet`保存500Hz simulator-native state：base pose/twist、joint position/velocity/torque、object pose/velocity和完整external wrench。
初始边界另可记录；必须区分physics steps数与包含初始状态的snapshot数。

`contact_sensor.parquet`每row只对应一次真实sensor buffer update：

```text
measurement_timestamp_ns
sensor_sequence_index
force
force_kind / frame
valid
torque / wrench if actually available
```

当前配置`update_period=0.005s`只代表标称200Hz。2ms物理步和lazy buffer更新机制不能证明实际严格200Hz。
实际节奏以fresh update时间戳/sequence为准；本轮不改sensor period、不提高查询频率来改变lazy update行为。
不得把旧contact值每2ms复制一次并称为新的500Hz measurement。

若physics snapshot附带latest contact，必须包含：

```text
contact_measurement_timestamp_ns
contact_measurement_is_fresh
```

用以明确sample-and-hold。新contact字段的net/per-pair、normal/friction、frame、是否available沿用第9节审计结论。
缺失数据显式标记；真实500Hz记录与sensor freshness仍需后续运行验收。

---

# 14. Task / Object State Contract

## First-task status and conditional Push Box observability — DATA-003

Approach-and-Push Box / Box 05 is **approved under DATA-004**. The current task requires full-footprint containment for 3 policy ticks; contact provenance is diagnostic only.

Push Box 已批准。以下 pair evidence 要求仅在可用时作为诊断解释；不阻塞任务冻结、success或采集：

- `task.contact.fl_object`必须来自可识别FL接触body与目标object的pair证据；`task.contact.body_object`独立记录非操作腿/机器人body与目标object的接触，并在task profile中明确body集合，不能把FL混入后者。
- 接触证据记录两端body/object身份、measurement timestamp/sequence、来源/API、坐标系、可用force分量，以及validity和不可用原因；法向力与可用摩擦分量分别解释，torque不可用时不伪造。
- Body-aggregated contact force（包括FL body净力）不能单独证明接触了目标物体，更不能单独证明任务由FL manipulation完成。距离+净力只作为inferred指标。
- 未观测到pair数据必须视为unknown，不能写成confirmed false。现有bool列不能单独表达unknown；证据应保存在Raw或具名extensions并由后续版本化task profile绑定，不能仅凭bool值通过任务归因/发布验收。
- 任务判定必须能区分FL-object、body-object及两者同时发生的情况。用户已明确：接触不作为success/collection gate；success只依赖整个XY footprint进入区域且机器人未摔倒，连续3个50Hz policy ticks。

这些是条件性的task contract要求，不表示sensor接线或真实pair观测已完成；现有通用schema与profile保持原版本和hash。

## Recorded task state


每个 50 Hz policy tick 同步保存：

```text
task.object.position
task.object.orientation
task.object.linear_velocity
task.object.angular_velocity

task.goal.position / region
task.progress
task.success

task.contact.fl_object
task.contact.body_object
```

用途：

```text
ground-truth success
failure diagnosis
evaluation metric
episode filtering
```

原则：

> task success MUST 来自 simulator ground-truth state，不允许通过 observer video 人工判断。

---

# 15. Success / Termination Contract

Episode-level:

```text
episode.status
```

枚举：

```text
success
failure
aborted
invalid
```

另外记录：

```text
success_timestamp_ns
termination_timestamp_ns
termination_reason
```

推荐 termination reason：

```text
task_success
timeout
robot_fall
workspace_violation
asset_failure
operator_abort
recording_failure
```

Raw archive 不删除失败 episode。

Training subset 可以 filter：

```text
status == success
```

---

# 16. Asset Metadata and Approval

每个 simulation asset 必须保存：

```text
asset_id
asset_name
source_url
source_repository
license
sha256
scale
collision_type
articulation_config
material_config
```

同时保存：

```text
approval.status
approval.timestamp
```

Production collection gate：

```text
if approval.status != approved:
    production_collection_allowed = false
```

每个新任务必须走：

```text
asset search
    ↓
candidate import
    ↓
physics validation
    ↓
three-camera preview
    ↓
USER APPROVAL
    ↓
task implementation / production collection
```

不允许 Codex 自主找到资产后直接开始 production collection。

---

# 17. Camera Calibration Metadata

每个 camera 必须记录：

```text
camera_id
role
policy_input
resolution
fps

fx
fy
cx
cy

horizontal_fov
vertical_fov

position
orientation
parent_link
extrinsic_matrix
```

同时记录：

```text
camera_config_hash
```

任何 policy camera pose / intrinsics / resolution 的 substantive change 都必须触发新的 camera config version。

---

# 18. Episode Provenance

每个 episode 至少记录：

```text
episode_uid
episode_index
task_id

dataset_schema_version
action_schema_version
camera_schema_version

random_seed

Isaac_Sim_version
Isaac_Lab_version

RAMBO_Data_commit
RAMBO_commit

task_config_hash
controller_config_hash
camera_config_hash

asset_ids
asset_hashes
```

建议同时记录：

```text
collection_timestamp
operator_id / collector_type
teleop_method
```

---

# 19. Configurable Resource Roots and Raw Layout

`datasets/`和`cache/`仅为逻辑结构，不表示大数据存入Git repository。

```text
RAMBO_DATA_ROOT/
  raw/v2/<task>/<episode>/
    manifest.json
    cameras/{ego,task_centric,observer}.mp4
    streams/{camera_index,policy_50hz,controller_100hz,physics_500hz,contact_sensor}.parquet
    metadata/{task,cameras,assets,provenance}.json
    review/{start,contact,success}.png
  canonical/lerobot_v2_1/<release>/

WAM_POLICY_CACHE_ROOT/
  latents/
  text_embeddings/
  normalization/
  manifests/
```

具体入口来自显式config字段`dataset_root`、`canonical_root`、`cache_root`，允许引用上述environment变量。
配置缺失或环境变量未解析时报错；源码中不猜测fallback路径。cache_root与canonical_root必须分离。
本地root取值沿用workspace资源层；不搬迁、删除或重写旧数据。
Git只保存schema/config/templates/converters/validators/tests及small fixtures，不保存正式大规模数据或cache。

---

# 20. Raw policy_50hz.parquet

每row对应完整20ms policy interval起点的observation/state和该区间的高层动作。

必须包含simulation_time_ns、policy tick或等价physics step、`action.requested:float32[9]`、`action.executed:float32[9]`、
50Hz robot/task state及execution confirmation/修改标记。完整chain hash写入配置/manifest。
Terminal state单独保存，不在policy表追加fake terminal action。
Canonical列名`action`显式映射自Raw的`action.executed`；名称转换不改数值。
实际未完成区间继续保留其partial记录，但不能作为完整20ms训练行。

---

# 21. Canonical 50 Hz LeRobot Dataset

Canonical dataset fps、两路主视频、action和state均为50Hz。第k行是动作前RGB/state与后续[t_k,t_{k+1})动作。
Main table有N行，主视频各N帧；terminal boundary通过额外metadata和两路PNG快照保留，不伪造action。

WAM随后执行固定相位0的50->12.5Hz RGB抽样，action保持50Hz。
它不改变canonical row count，不为VAE group缩短或补齐canonical数据。
Camera geometry、normalization和model input结构不因这个存储变化被重新设计。

---

# 22. Canonical LeRobot-compatible Layout

当前第一阶段优先兼容 LingBot 官方 post-training pipeline，而不是优先迁移到 LeRobot v3。

推荐：

```text
RAMBO_DATA_ROOT/
└── canonical/lerobot_v2_1/<release>/
    │
    ├── meta/
    │   ├── info.json
    │   ├── episodes_stats.jsonl  # LeRobot v2.1 reader使用
│   ├── stats.json            # 可选描述性聚合统计
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
    │
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       ├── episode_000001.parquet
    │       └── ...
    │
    └── videos/
        └── chunk-000/
            ├── observation.images.ego/
            │   ├── episode_000000.mp4
            │   └── ...
            │
            └── observation.images.task_centric/
                ├── episode_000000.mp4
                └── ...
```

Canonical额外包含terminal metadata与两路terminal PNG快照，供WAM在合法采样点使用；它们不新增LeRobot action row。

Observer camera 不进入该 training package。

如需公开 observer video，可作为 sidecar / review artifact 发布。

---

# 23. Canonical LeRobot Row Schema

正式训练 action：

```text
action = action.executed
```

shape:

```text
float32[9]
```

另外保留：

```text
action.requested
```

作为 audit-only feature。

Canonical row 至少包括：

```text
timestamp
frame_index
episode_index
task_index

action                         float32[9]
action.requested               float32[9]

observation.state.*
task.*
```

RGB 保存在对应 MP4，不嵌入 Parquet。

---

# 24. Observer Camera Exclusion Rule

禁止使用：

```text
observation.images.observer
```

作为 canonical training key。

推荐仅保留：

```text
raw/.../cameras/observer.mp4
```

或公开 release 中：

```text
review/
└── observer/
```

这样避免任何自动 camera enumeration 将 observer 意外送入模型。

---

# 25. LingBot `episodes.jsonl`

Baseline 示例：

```json
{
  "episode_index": 0,
  "tasks": [
    "Push the box into the target area."
  ],
  "length": 412,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 412,
      "action_text": "Push the box into the target area."
    }
  ]
}
```

第一阶段：

```text
one episode = one action segment
```

以后只有明确需要 long-horizon subtask annotation 时才允许拆分：

```json
{
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 140,
      "action_text": "Approach the box."
    },
    {
      "start_frame": 140,
      "end_frame": 412,
      "action_text": "Push the box into the target area."
    }
  ]
}
```

---

# 26. Dataset Statistics and Normalization

已核对lerobot==0.3.3默认保存`meta/episodes_stats.jsonl`并在读取时聚合；
下面的`meta/stats.json`是可选描述性统计，不替代该版本实际需要的episode stats文件。

Canonical dataset 可以在：

```text
meta/stats.json
```

保存描述性统计：

```text
mean
std
min
max
```

但训练 normalization MUST NOT 直接等同于 canonical stats。

WAM-Policy 应仅使用：

```text
training split
```

计算：

```text
per-dimension action quantiles
```

防止 validation / test leakage。

---

# 27. Train / Validation / Test Split

切分单位必须是：

```text
episode
```

禁止按：

```text
frame
window
latent chunk
```

随机切分。

更严格时按：

```text
initial configuration
random seed
task geometry
```

隔离。

示例：

```text
50 demos

40 train
5 val
5 test
```

正式论文 evaluation 最终应使用独立 evaluation episodes，而不是简单复用 demonstration test split。

---

# 28. WAM-Policy Model-specific Cache

Canonical dataset 进入 WAM-Policy 后：

```text
50 Hz ego video
50 Hz task-centric video
50 Hz 9D action
raw text
        ↓
video frame sampling
        ↓
12.5 Hz policy video
        ↓
Wan causal VAE
        ↓
multi-view latent composition
        ↓
LingBot SFT
```

推荐：

```text
WAM_POLICY_CACHE_ROOT/
├── latents/
├── text_embeddings/
├── normalization/
│   ├── action_quantiles.json
│   └── ...
└── manifests/
```

Cache metadata 至少应记录：

```text
source dataset version
source episode
source video FPS
target video FPS
sampled frame IDs
segment start/end
raw text
latent shape
camera composition rule
action schema version
normalization version
```

---

# 29. Deterministic Sampling, Causal Start and Episode Tail

50Hz source RGB固定选择`0,4,8,12,...`，sampling_phase=0；action保持50Hz。
Factor-4只决定选择哪些canonical rows。sampled timestamp必须逐项继承对应row的authoritative `simulation_time_ns`；选择terminal边界时继承terminal timestamp。尾部action timestamps同样读取source rows，不通过`index * 20 ms`合成。Nominal 20ms grid只用于validation，不能替代source clock；保留int64纳秒精度，不经float时间戳转换。
每个12.5Hz model RGB interval对应4个50Hz action；一个后续VAE group覆盖4个model RGB间隔，对应16个action slots。
不要把model RGB frame与VAE latent frame混用。

初始latent的历史action slots仅为结构占位：zero、mask=false，不参与loss，不能解释为真实executed action。
占位只出现在WAM derived preprocessing，绝不写入Raw/Canonical物理action表。

Raw和Canonical尾部全部保留；model cache仅使用完整合法group，manifest记录tail length、dropped indices、timestamps。
不padding fake actions。Canonical terminal快照只有落在固定抽样相位且能形成合法group时才使用。
例如N=64：64行/64个主视频帧+terminal64，模型选0..64步长4得到17RGB、5latent、64有效actions+16无效初始slots。
N=65仍保留65行，当前完整group使用前64动作，剩余动作在cache manifest中记录。

新episode重置两路causal VAE temporal state、pending frames、KV/history；不继承跨episode上下文。
若训练时pack多个episode，attention mask必须阻断跨episode attention。
本轮最小reader使用人工cache测试；正式VAE/T5生成与packing训练器未在本轮实现。

---

# 30. Dataset QA and Compatibility Gates

进入正式canonical前验证：逐stream timestamp/schedule、physics/controller/action连续性、50Hz policy视频decoded count与index、
observer缺失的显式原因、9D dtype/单位/有限值及当前transform chain、Raw N/N+1与canonical terminal、
robot/object state及quaternion、force/contact/wrench含义、资产批准、provenance、UTF-8 text和完整action_config。
MP4须满足第6节编码/close/validation要求；不得silent repair。修复必须产生新派生版本、记录过程并保留原episode。

Compatibility分层报告：

1. lerobot==0.3.3实际读取canonical两路key、MP4、Parquet、tasks/episodes/action_config、9D action；
2. 固定LingBot reader的原始行为对照；
3. 最小WAM适配后native9、ego-above-task、false初始mask、独立cache root及model batch符合阶段2接口。

不得只因目录类似就声明兼容。原reader默认把latent放dataset内、按RoboTwin做拼接/动作处理，必须通过明确适配复用已有v2规则。
第三方源码与阶段2模型保持不变。
CPU人工样例通过只表示格式/语义适配通过，不表示真实采集、传感器节奏、runtime或任务效果通过。

---

# 31. Versions and Historical Contracts

```text
dataset_schema_version: wam-quadruped-v2.1.0
LeRobot codebase_version: v2.1
LeRobot library version: 0.3.3
action semantic identity: rambo-native9-v2
```

这几个版本不可混同。旧PNG数据contract 2.0.0只供历史审计，不能原地修改profile内容后复用旧hash。
当前runtime消息协议2.0.0不因dataset落盘格式变化自动升级。
数据profile、camera mapping、encoding和transform chain分别固定hash；任何substantive变化使用新的dataset/profile版本。
正式release不可覆盖；历史run/manifests/模型身份保持不变。

---

# 32. End-to-end Data Flow

```text
                         Isaac Sim
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
    Cameras                RAMBO                Physics
   ego 50 Hz              100 Hz                500 Hz
   task 50 Hz
   observer 25 Hz
       │                     │                     │
       └──────────────┬──────┴──────────────┬─────┘
                      │                     │
                9D command 50 Hz        task/object state
                      │
                      ▼
              RAMBO_Data RAW ARCHIVE
                      │
        ┌─────────────┴──────────────┐
        │                            │
 multi-rate telemetry         observer/debug
        │
        ▼
     validation
        │
        ▼
 Canonical LeRobot 50 Hz
        │
        ├── ego RGB             MP4
        ├── task-centric RGB    MP4
        ├── action              Parquet
        ├── robot state         Parquet
        ├── task/object state   Parquet
        ├── task text           metadata
        ├── action_config       metadata
        └── episode metadata    JSONL
        │
        ▼
                  WAM-Policy
        │
        ├── RGB 50 → 12.5 Hz
        ├── action stays 50 Hz
        ├── 1 video : 4 actions
        ├── Wan causal VAE
        └── T5 / text encoding
        │
        ▼
  LingBot-specific latent/text cache
        │
        ▼
       SFT
```

---

# 33. Final Schema Boundary

## 33.1 Canonical 50 Hz Robot Dataset

逻辑上：

```text
Canonical sample at 50 Hz
=
{
    two policy RGB streams,
    executed 9D action,
    requested 9D action,
    robot state,
    task/object state,
    language text,
    timestamp,
    episode metadata
}
```

## 33.2 Current LingBot v2 Model Inputs

当前真正输入模型的只有：

```text
ego RGB @ 12.5 Hz
+
task-centric RGB @ 12.5 Hz
+
executed 9D action @ 50 Hz
+
language text
```

## 33.3 Stored but Not Used as Current Policy Input

以下内容必须保存，但不作为当前 baseline 输入：

```text
robot proprioception
contact state
task/object state
RAMBO controller telemetry @ 100 Hz
Isaac physics telemetry @ 500 Hz
observer video @ 25 Hz
requested 9D action
actual contact force
external wrench
```

---

# 34. Codex Implementation Requirements

Codex 在执行本 contract 时应遵守以下约束：

1. 先实现 schema / manifest / validation，再开始 production collection。
2. 不允许为了方便训练而丢弃 raw multi-rate telemetry。
3. 不允许把 model-specific cache 放进 RAMBO_Data canonical dataset。
4. 不允许将 observer camera 加入 policy observation namespace。
5. 不允许只保存 requested action；必须保存 requested + executed。
6. 不允许把 desired force / contact force / external wrench 混成一个字段。
7. 不允许在未明确 user approval asset 前开始正式数据采集。
8. 不允许 silent repair invalid episode。
9. 不允许按 frame/window 做 train/val/test split。
10. 不允许在 canonical dataset 中保存 T5 / VAE / CLIP 等 model-specific embedding。
11. 不允许修改 50 Hz → 12.5 Hz temporal sparsification rule，除非显式产生新 dataset/model preprocessing version。
12. 数据 contract 的任何 substantive change 必须通过 schema version bump 体现。

---

# 35. Implementation Milestones and Current Authorization

当前只实施schema/manifest/validation、最小读取/映射适配和CPU人工兼容性测试。
以下Recorder、Pilot、生产转换、真实VAE/T5步骤是后续gate，不因本文存在就自动授权执行。

Codex 应按以下顺序推进：

### Milestone A — Schema

完成：

```text
action schema
camera schema
time schema
robot-state schema
task-state schema
episode manifest schema
asset provenance schema
```

### Milestone B — Recorder

验证：

```text
50 Hz ego
50 Hz task-centric
25 Hz observer
50 Hz requested/executed 9D
100 Hz controller telemetry
500 Hz physics telemetry
```

全部在同一个 simulation clock 下稳定记录。

### Milestone C — One Pilot Episode

仅采 1 条 episode，用于验证：

```text
timestamps
frame counts
action/state completeness
observer isolation
force semantics
task success state
provenance
```

### Milestone D — Canonical Conversion

从 pilot raw episode 转出一条：

```text
LeRobot-compatible canonical episode
```

检查：

```text
50 Hz row alignment
two policy camera streams
task text
action_config
executed 9D action
state fields
```

### Milestone E — WAM-Policy Read Test

WAM-Policy 能够：

```text
load canonical dataset
sample 50 Hz RGB
deterministically downsample to 12.5 Hz
retain 50 Hz action
form 1 video : 4 action relation
generate VAE/text cache
```

完成以上 gate 后，才进入正式 3-episode / 10-demo data collection。

---

# 36. Reference Basis

本 contract 主要遵循以下设计来源：

- LingBot-VA / *Causal World Modeling for Robot Control*
  - 50 Hz action
  - 50 Hz source video → 12.5 Hz model video
  - action-per-video-frame temporal interleaving
  - LeRobot-based custom-data pipeline
  - `action_config`
  - Wan causal VAE latent preprocessing
- Hugging Face LeRobot
  - low-dimensional tabular data + MP4 video + metadata organization
- Current RAMBO_Data / WAM-Policy v2 engineering decisions
  - quadruped-only
  - single FL manipulation leg
  - 9D high-level RAMBO command
  - ego + task-centric policy cameras
  - observer-only third-person camera
  - raw/canonical/model-cache repository separation

This versioned document is the authoritative dataset specification for Adaptation v2. Specification acceptance does not assert production runtime or collection acceptance.


## Current DATA-004 valid replacement pilot

The prior 8.04s/402-action center-only pilot is invalid for current DATA-004 and excluded from future training. No legacy/compatibility path or history rewrite was introduced. The only current valid technical pilot is the new straight-push replacement under task profile `push-box-v2-2`.

The reviewed narrow `local_x_min` face is allowed. Actual mesh bounds define its center/normal and all corners. Box dimensions are about30.0763x17.4538x22.6464cm. Nominal geometric center is(0.690382,0.142000,0.133114)m, face center(0.540000,0.142000,0.133114)m and normal(-1,0,0), yaw0. Robot stays at native reset XY origin; FL's neutral lateral command aligns the push line. No yaw randomization, controller replacement or model architecture change.

The scripted expert approaches until the measured face distance is suitable, then approaches normally and advances the body with about0.38m FL forward reference. World-axis references are transformed into the existing projected-com native9 frame. This is demonstration generation using simulator state, not a learned-policy result. Intended face/commanded contact/actual FL EEF/box pose/yaw are recorded in audit-only task-review metadata.

Success requires the entire projected source-box bounding volume inside goal x=[0.95,1.35], y=[-0.10,0.40]m and robot not fallen for3 consecutive50Hz policy ticks. Four XY envelope corners are computed from all8 pose-transformed3D corners, including roll/pitch. Contact remains diagnostic/unknown and never gates success; yaw is reported without a success threshold.

The valid pilot lasts9.08s with454 completed actions. Full containment first appears at physics timestamp9036000000ns and policy timestamp9040000000ns; success ticks are9.04/9.06/9.08s. Terminal footprint X=[0.956747,1.297620],Y=[0.038445,0.257470]m is fully inside. Forward displacement0.436802m, net lateral displacement0.005958m. Yaw range[-12.405,1.875]degrees, final-7.396degrees; residual rotation is disclosed.

Real terminal-before-reset/reset isolation passed on the current implementation and final pilot. Raw/canonical validation and MP4 decode passed:455 dual50Hz Raw boundaries,454 canonical rows/frames plus separate terminal state/RGB,908 controller steps,4540 physics steps/4541 snapshots,909 actual sensor updates,228 observer25Hz frames. Requested/executed native9 and zero desired force/external wrench retained. Observer is never a model input or success ground truth. DATA-002 schema/profile bytes are unchanged.

WAM reads113 sampled RGB frames per view with inherited source timestamps, encodes real frozen VAE/T5, and reloads latents[1,48,29,24,20], actions/mask[1,9,29,16,1], text[1,512,4096]. Initial mask is false.448 actions enter complete groups;6 tail actions and their timestamps remain in Raw/canonical and are explicitly reported by cache metadata. Normalizer is diagnostic identity, not training statistics.

Evidence: workspace `runs/audit/v2/DATA-004/straight-push-20260916T042259Z`. Two failed development collection attempts are excluded (spawn XY/controller reference mismatch; expert overreach and lateral deflection). Only the replacement is the current valid pilot. WAM73 and RAMBO220 CPU tests passed; real data/model gates were executed separately. The user accepted this replacement pilot and authorized DATA-004 commit/push. Further batch demonstrations require separate authorization; formal training/release and model server closed-loop remain not_run.
