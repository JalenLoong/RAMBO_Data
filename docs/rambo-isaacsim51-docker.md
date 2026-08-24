# RAMBO Isaac Sim 5.1 Blackwell Docker image

本文说明如何构建和验证 `rambo-isaac51-blackwell:r1`。镜像用于把已经在 Vast.ai RTX 5080 上验证的 RAMBO / Isaac Sim 5.1 runtime 固化下来；checkpoint、dataset、验证输出和凭据不进入镜像。

## Runtime contract

| 项目 | 镜像约定 |
| --- | --- |
| Base image | `vastai/linux-desktop:cuda-12.1-ubuntu-22.04` |
| Python | deadsnakes stable PPA 提供的 Python 3.11；不替换 base 的系统 Python 3.10 |
| Python venv | `/opt/venvs/rambo51` |
| RAMBO reference copy | `/opt/rambo`，以 editable mode 安装 |
| 用户 workspace | `/workspace`，可安全挂载 storage |
| PyTorch / CUDA runtime | `2.7.0+cu128` / 12.8 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.2 |
| QP stack | qpth 0.0.18、CVXPY 1.5.4、OSQP 0.6.7.post3、ECOS 2.0.14 |

2026-08-24 检查到该 base tag 的 registry digest 为 `sha256:ebe4705b9a7a323994c656aac686dc54691f58d676b2f41d3811c616fe9e16a2`。Runtime contract 中保留可读的 tag 名，Dockerfile 实际使用完整的 `tag@sha256:digest` 引用；这样即使 upstream tag 发生漂移，重新构建仍会使用这次已经验证的 exact base image。

Base image 的入口为 `/opt/instance-tools/bin/entrypoint.sh`，负责 Vast.ai Linux Desktop / Selkies 启动。`Dockerfile.rambo51` 没有声明 `ENTRYPOINT` 或 `CMD`，也不删除或替换 desktop、Jupyter 和 Selkies runtime，因此继承该入口。镜像不安装 NVIDIA kernel driver；GPU driver 始终来自 Vast.ai host。

该 base 同时预置 deadsnakes stable 和 nightly apt source，但没有安装 Python 3.11。构建会保留 stable source、显式禁用 nightly source，再安装并检查 Python minor version，避免开发频道 package 漂移。

## Build

建议预留至少 40 GB 可用 Docker 存储空间，并使用支持 cache mount 的 BuildKit。首次构建需要下载数十 GB 的 base、Torch 和 Isaac wheels。

```bash
docker build \
  -f Dockerfile.rambo51 \
  -t rambo-isaac51-blackwell:r1 \
  .
```

不要默认使用 `--no-cache`。构建分为以下缓存边界：

1. 先安装 Python 3.11；
2. 只复制 `scripts/setup_isaacsim51.sh` 和两个 `requirements/isaacsim51.*` manifest；
3. 以 `RAMBO_SKIP_LOCAL_EDITABLE=1` 安装 Torch、Isaac Sim、Isaac Lab、official `isaaclab_assets` 和 QP stack；
4. 用 `requirements/isaacsim51.lock` 对齐验证机的完整 third-party freeze，关闭 transitive version drift；
5. 最后复制 repository 到 `/opt/rambo`，并对 `source/crl2`、`source/rambo` 执行不解析依赖的 lightweight editable install。

因此普通 RAMBO Python 修改不会使重型依赖层重新下载。apt 和 pip 下载使用 BuildKit cache mount，下载缓存不会写入最终镜像层。

## 基础静态验证（不需要 GPU）

下面的检查只验证 baked Python 环境、distribution metadata 和依赖一致性，不启动 Isaac Sim，也不要求 `torch.cuda.is_available()` 为 true：

```bash
docker run --rm \
  --entrypoint /bin/bash \
  rambo-isaac51-blackwell:r1 \
  -lc '
    python --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    python -c "import importlib.metadata as m, importlib.util as u; print(\"isaaclab\", m.version(\"isaaclab\")); print(\"isaacsim\", m.version(\"isaacsim\")); print(\"isaaclab_assets\", u.find_spec(\"isaaclab_assets\").origin); print(\"qpth\", m.version(\"qpth\"))"
    python -c "import crl2, qpth, rambo; print(\"RAMBO_REFERENCE_IMPORTS_OK\")"
    python -m pip check
  '
```

预期包括 Python 3.11、Torch `2.7.0+cu128`、`torch.version.cuda == 12.8`、Isaac Lab 2.3.2、Isaac Sim 5.1.0.0、可解析的 `isaaclab_assets` 和 qpth 0.0.18，以及 `No broken requirements found.`。没有 GPU 的 WSL/build host 上 `torch.cuda.is_available()` 为 false 是正常结果，不代表 Vast GPU runtime 有问题。

可额外确认最终镜像仍继承 Selkies 入口，且没有 image-level command 覆盖：

```bash
docker image inspect rambo-isaac51-blackwell:r1 \
  --format 'entrypoint={{json .Config.Entrypoint}} workdir={{json .Config.WorkingDir}}'
```

入口应为 `["/opt/instance-tools/bin/entrypoint.sh"]`，工作目录应为 `/workspace`。该 base entrypoint 会把普通 `docker run IMAGE bash -lc ...` 中的参数解释为 desktop flags，而不是执行 shell；因此一次性检查显式使用 `--entrypoint /bin/bash`。这个 CLI override 只作用于该临时容器，不修改镜像入口或 Vast.ai 的默认启动行为。

## 使用 `/workspace` 中的开发代码

镜像启动后的 reference baseline 位于 `/opt/rambo`，但开发 checkout 应放在不会遮住 runtime 的 `/workspace/rambo`：

```bash
cd /workspace/rambo
git switch new_IsaacSim_IsaacLab

/opt/venvs/rambo51/bin/pip install -e /workspace/rambo/source/crl2
/opt/venvs/rambo51/bin/pip install -e /workspace/rambo/source/rambo
/opt/venvs/rambo51/bin/python -m pip check
```

这会让同一个 venv 使用 workspace 中正在开发的包。挂载 `/workspace` 不会遮住 `/opt/venvs/rambo51` 或 `/opt/rambo`。

## Vast RTX 5080 端验证

首次启动 Isaac Sim 时由操作者阅读并接受 NVIDIA EULA。不要在 image build 中自动接受 EULA。只有操作者已经合法接受后，自动化 session 才应设置 `OMNI_KIT_ACCEPT_EULA=YES`。

以下命令在 RTX 5080、R580 host driver、GPU/Vulkan 已暴露给容器后执行。先保存 host fingerprint：

```bash
nvidia-smi
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader
vulkaninfo --summary

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("compute_capability", torch.cuda.get_device_capability(0))
print("architectures", torch.cuda.get_arch_list())
PY
```

预期 driver 是已验证的 `580.159.03`（最低目标 580.65.06），GPU capability 为 `(12, 0)`，Torch arch list 包含 `sm_120`。

从 baked reference copy 运行 official Go2、RGB 和 qpth gate：

```bash
cd /opt/rambo

scripts/rambo/run.sh scripts/rambo/smoke.py --headless --steps 100

scripts/rambo/run.sh scripts/rambo/smoke.py \
  --headless --steps 100 --enable-camera \
  --camera-output /tmp/rambo-go2-smoke-rgb.npy

RAMBO_REQUIRE_CUDA_QPTH=1 scripts/rambo/run.sh \
  -m pytest -q tests/rambo/test_qpth_contract.py
```

checkpoint 不在 image 中。准备并核对 allow-listed checkpoint 后，再做短 rollout：

```bash
export RAMBO_QUAD_CKPT=/workspace/rambo-checkpoints/quadruped/model_2000.pt
export RAMBO_BIPED_CKPT=/workspace/rambo-checkpoints/biped/model_4000.pt

sha256sum "$RAMBO_QUAD_CKPT" "$RAMBO_BIPED_CKPT"

scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Quadruped-Go2-v0 \
  --checkpoint "$RAMBO_QUAD_CKPT" \
  --steps 16 --enable_cameras \
  --output-dir /workspace/rambo-validation/quadruped-short-01

scripts/rambo/run.sh scripts/rambo/validate.py \
  --task Isaac-RAMBO-Biped-Go2-v0 \
  --checkpoint "$RAMBO_BIPED_CKPT" \
  --steps 16 --enable_cameras \
  --output-dir /workspace/rambo-validation/biped-short-01
```

已验证 checkpoint SHA256 分别应为：

```text
1cc5f68fe15e37ccabae26060d79a26a8c078ed465a81f6f729c2009b67ca706  model_2000.pt
c16e64bf1ca2dc16878c386b742cd303e65040f52e8744cd0c96c540c595b2a6  model_4000.pt
```

最后运行 loco-manip smoke：

```bash
scripts/rambo/run.sh scripts/rambo/teleop_loco_manip.py \
  --headless --smoke-loco-manip \
  --checkpoint "$RAMBO_QUAD_CKPT"
```

短 gate 全部通过后，按 `rambo-isaacsim51-blackwell-runbook.md` 继续执行两种 3000-step RGB validation、GUI playback 和真实 viewport keyboard loco-manip 验证。CPU-only container 检查不能替代这些 CUDA/Vulkan/RTX 验收。

## Future GHCR commands

这里只记录命令；本流程不登录 GHCR，也不执行 push：

```bash
docker tag \
  rambo-isaac51-blackwell:r1 \
  ghcr.io/jalenloong/rambo-isaac51-blackwell:r1

docker push \
  ghcr.io/jalenloong/rambo-isaac51-blackwell:r1
```

## Secrets and build context

`.dockerignore` 排除 `.git`、`.agents`、`.codex`、本地 venv、cache、checkpoint、weights、datasets、RGB/trajectory/validation outputs、logs 和常见凭据文件。不要通过临时修改 `.dockerignore` 把 GitHub/Hugging Face/Vast token、SSH key、registry credential、private dataset credential 或本地 `.env` COPY 进 image；需要认证的数据应在 runtime 通过 Vast secrets 或只读 mount 注入。
