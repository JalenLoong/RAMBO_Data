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
- 用户已明确接受 EULA；只对需要的执行进程设置 `ACCEPT_EULA=Y`，不设置 `PRIVACY_CONSENT`。
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
| M2–M11 | 未开始 | 见 `CODEX_EXECUTION_PLAN.md` |

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

## 当前 blocker

无技术 blocker。GitHub push 需要凭证但不阻止本地迁移；Docker/Toolkit 不存在是 M11 的可延后事项。

## 下一精确动作

提交本次 M1 状态更新，然后实现并运行 M2 的隔离官方 PhysX/CUDA/RTX/RGB/clean-restart smoke；该阶段不得 import RAMBO。
