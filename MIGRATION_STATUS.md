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

## 进度

| 阶段 | 状态 | 证据 / 下一动作 |
|---|---|---|
| P0：备份权威计划 | 完成 | SHA256 `ad85d4a5e99d39ee3e29023cfcb481b030b47967a59fd53602564b851a551eff` |
| P0：fetch/核验远端 | 完成 | `origin/new_IsaacSim_IsaacLab` = `10154c6…` |
| P0：tag/worktree | 完成 | tag `isaac51-golden-10154c6`；worktree `/workspace/rambo60` |
| P0：持久化计划 | 进行中 | 将本文件和 `CODEX_EXECUTION_PLAN.md` 提交并 push |
| M0 | 未开始 | 核验后删除已授权 `/workspace/venvs/rambo51`，记录空间变化；69 GiB 是硬下限、70 GiB 是目标 |
| M1–M11 | 未开始 | 见 `CODEX_EXECUTION_PLAN.md` |

## 最近命令与结果

1. 备份 authority plan，源/备份 SHA256 一致。
2. `git fetch origin refs/heads/new_IsaacSim_IsaacLab:refs/remotes/origin/new_IsaacSim_IsaacLab`：成功。
3. 创建 annotated baseline tag 与 migration worktree：成功。

## 当前 blocker

无。Docker/Toolkit 不存在是 M11 的可延后事项，不阻止原生迁移。

## 下一精确动作

在 `/workspace/rambo60` review 并提交 P0 文档，然后执行 M0 的目标路径/空间预检；确认后删除
唯一已授权的损坏 venv `/workspace/venvs/rambo51`。
