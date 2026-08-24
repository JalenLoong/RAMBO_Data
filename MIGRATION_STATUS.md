# RAMBO Isaac Sim 6 迁移状态

## 当前目标

- Isaac Sim: `6.0.1.0`
- Isaac Lab: `v3.0.0-beta2.patch1` / `ffff603eafc6b74264a5261cc0183d6a65390d78`
- Physics: **PhysX only**；Newton 包存在允许，Newton 实际运行失败。
- Renderer: Isaac RTX；headless `--viz none`，GUI `--viz kit`。

## 已确认事实

- 旧基线：`10154c6d2f20d33f51e5e126801c507e6fe206db`。
- 迁移分支：`adapt/isaacsim60-isaaclab30b2`。
- 迁移 worktree：`/workspace/rambo60`。
- 旧计划备份：`/workspace/migration-reference/rambo-isaacsim60-isaaclab30b2-adaptation-plan-ad85d4a5e99d39ee3e29023cfcb481b030b47967a59fd53602564b851a551eff.md`。
- 用户已明确接受 EULA；原生 pip 启动仅对需要的执行进程设置
  `OMNI_KIT_ACCEPT_EULA=Y`，不设置 `PRIVACY_CONSENT`，也不写入持久化
  `EULA_ACCEPTED` 文件。
- Host：Ubuntu 24.04.4、RTX 4090 24 GiB、driver 595.71.05、CUDA capability 8.9、Python 3.12.3、uv 0.11.19。
- Runtime artifact root：`/workspace/migration-output/isaac60/`；原定
  `/workspace/migration_logs/isaac60/` 对执行用户不可写。

## 进度

| 阶段 | 状态 | 证据 / 下一动作 |
|---|---|---|
| P0：备份权威计划 | 完成 | SHA256 `ad85d4a5e99d39ee3e29023cfcb481b030b47967a59fd53602564b851a551eff` |
| P0：fetch/核验远端 | 完成 | `origin/new_IsaacSim_IsaacLab` = `10154c6…` |
| P0：tag/worktree | 完成 | tag `isaac51-golden-10154c6`；worktree `/workspace/rambo60` |
| P0：持久化计划 | 完成 | commit `eb0958ad975133188866d1f92a104c59c850707c`；push 因无 GitHub 凭证待补 |
| M0 | 完成 | 删除已授权的损坏 venv，释放 20,005,949,440 B；证据见 `M0/20260824T100222Z-host-and-cleanup/` |
| M1：精确栈安装 | 完成 | Isaac Sim 6.0.1.0、tag `ffff603…`、Torch 2.10.0+cu128、RTX 4090 CUDA 已通过 |
| M2：官方 PhysX 门禁 | 完成 | Cartpole、Go2、RTX RGB、Kit GUI 与独立 clean restart 均通过；脚本 `scripts/rambo/official_physx_smoke.py` |
| M3：API inventory | 完成 | `MIGRATION_API_INVENTORY.md`；覆盖 PhysX、ProxyArray、XYZW、view/writer、Camera、Button、teleop 与 lifecycle |
| M4：核心 API / PhysX contract | 完成 | 共享 fail-closed PhysX contract、XYZW↔WXYZ 明确桥接、ProxyArray `.torch`、公开 writer/ContactSensor/composer API、Python 3.12 runtime 入口均已迁移；22 个静态/契约测试通过，四足与双足各完成真实 5-step PhysX smoke。 |
| M5：qpth / QP contract | 完成 | `qpth==0.0.18` 的 CPU/CUDA forward/backward、CPU gradcheck、KKT、残差和五次确定性均通过；未运行任何物理后端。 |
| M6：四足 checkpoint / 长 rollout | 待开始 | 等待以 `model_2000.pt` 做严格加载、16-env smoke、3000-step 无相机验收。 |
| M7：RGB camera | 进行中 | 正在并行替换旧 SensorBase 私有 cadence/旧 WXYZ camera offset，并准备独立 RGBD PhysX smoke。 |
| M8：Button / keyboard | 待开始 | 依赖 M6/M7 通过。 |
| M9：双足 checkpoint / 长 rollout | 待开始 | 依赖四足、RGB、Button 通过。 |
| M10：LingBot 数据合成 | 待开始 | 依赖 M8/M9 通过。 |
| M11：冻结 / Docker | 待开始 | native 运行全部通过后执行；Docker/Toolkit 缺失可延后记录。 |

## 最近命令与结果

1. 备份 authority plan，源/备份 SHA256 一致。
2. `git fetch origin refs/heads/new_IsaacSim_IsaacLab:refs/remotes/origin/new_IsaacSim_IsaacLab`：成功。
3. 创建 annotated baseline tag 与 migration worktree：成功。
4. `git push -u origin adapt/isaacsim60-isaaclab30b2`：失败，原因是当前环境没有 GitHub 用户名/凭证；本地 commit 保留。
5. 删除 `/workspace/venvs/rambo51`：成功；路径为普通目录、非链接、非 mountpoint，删除前大小 19,671,114,584 B，实际释放 20,005,949,440 B。
6. M1：clone exact Isaac Lab tag、创建 `/workspace/venvs/rambo60`、安装 `isaacsim[all,extscache]==6.0.1.0`：成功。
7. M1：按 tag 覆盖为 `torch==2.10.0+cu128`、`torchvision==0.25.0+cu128`，移除 torchaudio：成功。
8. M1：安装最小 editable 集合（含裸 `isaaclab_newton`，不含 optional extras/RL）：成功。CUDA tensor 测试确认 RTX 4090。
9. M1 `pip check`：仅四项已知差异：isaacsim-core 的 torchaudio/Torch/vision metadata 和 isaacsim-kernel coverage 7.4.4 vs Isaac Lab 7.6.1。
10. M2：为精确 tag 的 `isaaclab_tasks` 补装常规 `hydra-core==1.3.2` / `omegaconf==2.3.1`；未安装任何 Newton optional extra。
11. M2：以进程级 `OMNI_KIT_ACCEPT_EULA=Y` 启动隔离官方 gate。Cartpole 16-step、Go2 16-step、RTX RGB 640×480、Kit GUI Cartpole 8-step 与热缓存相机 clean restart 均通过，五个成功 artifact 的 exit code 均为 0。
12. M2：每个成功 summary 都同时记录 `PhysxCfg`、`PhysxManager`、`use_newton_actuators=false` 与 `cuda:0`。Kit GUI experience 按固定官方依赖图加载了 `isaaclab_newton` 扩展，但活跃 manager 仍是 `PhysxManager`，未选择或执行 Newton physics backend。首次 RTX 冷启动为 shader cache 初始化耗时约 156 秒；独立热缓存重启为 8 秒，满足 60 秒关闭/重启门槛。
13. M3：完成对所有 RAMBO 配置、入口、资产/传感器数据、QP、相机、Button、teleop、生命周期和打包 metadata 的只读 API inventory。确认现有 RAMBO 源码没有主动执行 Newton，但除官方 gate 外的运行路径仍需在 M4 显式选择并实际断言 PhysX。
14. M4：新增共享 `rambo.utils.physx`，每个 RAMBO config 均以 `PhysxCfg()` 覆盖 physics、设定 `use_newton_actuators=False`，并在环境构造后与 rollout 后核验实际 `PhysxManager` FQCN；生产/验证入口只允许显式 `--viz none|kit`。
15. M4：迁移四足、双足 QP 环境至 Isaac Lab 3 的 XYZW、ProxyArray `.torch`、公开 articulation data、indexed writers、`find_sensors` 与永久 wrench composer。双足 FL/FR 力现在以一次合并 composer 调用写入，避免后一足覆盖前一足。
16. M4：修复 DirectRLEnv 已弃用的 `num_actions`/`num_observations` aliases 后，真实最终证据为四足 `M4/20260824T111439Z-quadruped-space-contract-r2` 与双足 `M4/20260824T111439Z-biped-space-contract-r2`：各 5 control steps / 25 physics ticks、405/18 或 435/18、`dt=0.002`、decimation=5、运行前后均为 `PhysxManager`、Newton actuators=false；两个 `summary.json` SHA256 分别为 `fe43f0e3346fd192136459eb8d25e0147d55980ee631fa53b5b70d2f4e70bec9` 与 `2aef9f13f5dbd441d0600f22e16242afad040bfc6f147e011999c6858ef35c1c`。
17. M5：`/workspace/migration-output/isaac60/M5/20260824T105302Z-qpth-contract/results.json` 记录 qpth 0.0.18 在 Torch 2.10.0+cu128 / CUDA 12.8 上的 CPU/CUDA 解、梯度、gradcheck、KKT、约束残差与确定性均成功；CUDA 仅预加载 PyTorch 自带 `libtorch_cuda_linalg.so`，未改变 solver 算法或安装 Newton optional extras。

## 当前 blocker

无技术 blocker。GitHub push 需要凭证但不阻止本地迁移；Docker/Toolkit 不存在是 M11 的可延后事项。M7 camera 代码正在并行迁移，完成前不启动依赖 RGB 的 checkpoint 验收。

## 下一精确动作

完成 M7 camera 后，从 M6 开始以四足 `model_2000.pt` 的 SHA256 allow-list 和严格 state-dict 加载，依次执行 1-env/16-env/3000-step PhysX 验收。
