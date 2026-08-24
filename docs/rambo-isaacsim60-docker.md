# RAMBO Isaac Sim 6 Docker specification (deferred)

本文档记录未来可复现的容器化路径；它**不是**已完成的 Docker 验收报告。截至
2026-08-24，原生 RTX 4090 验收主机没有 Docker Engine 或 NVIDIA Container
Toolkit，因此本仓库没有执行 `docker build`、`docker run` 或发布镜像。原生
PhysX 验收仍是唯一有效的运行证据，状态以
[`MIGRATION_STATUS.md`](../MIGRATION_STATUS.md) 为准。

`Dockerfile.rambo60` 是 fail-closed 模板。没有给出已验证的 Ubuntu base
digest 时，它会直接拒绝 build；不要为了让 build 继续而改用可变 tag。

## 固定 runtime contract

| 项目 | 要求 |
| --- | --- |
| Base image | 由操作者提供、预先核验的 Ubuntu 24.04 `tag@sha256:digest`；必须兼容 host R595/NVIDIA Container Toolkit |
| Python | 3.12 |
| Isaac Sim | `6.0.1.0` |
| Isaac Lab | `v3.0.0-beta2.patch1` / `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| PyTorch | `2.10.0+cu128` / CUDA 12.8 userspace |
| Physics | 配置和实际 manager 均为 PhysX；`use_newton_actuators=false` |
| Renderer | Isaac RTX，目标 GPU 为 RTX 4090 |
| Checkpoint / artifacts | runtime read-only mount / output mount；绝不 COPY 进 image |

镜像不安装 NVIDIA kernel driver，也不安装任何 Newton optional extra。精确
Isaac Sim / Isaac Lab 依赖图所要求的裸 Newton 相关 package 可以存在，但
RAMBO config、smoke、validation、recorder 和 production execution 都必须在
运行前后记录实际 `PhysxManager`，不能选择或执行 Newton backend。

base image 还必须是 x86_64、以 root 构建、没有继承的 `PYTHONPATH`，并提供
Ubuntu 的 `libEGL`、`libGL` 与 Vulkan loader userspace。Dockerfile 对前 3 项
fail-closed 检查，并安装后 3 项的 loader；这不等同于把 NVIDIA driver 放进
image。host 的 NVIDIA Container Toolkit / driver passthrough 仍是运行时前提。

## EULA boundary

不要在 Dockerfile、image ENV、image label 的可执行环境变量、shell profile、
`.env`、secret layer 或 EULA marker 中接受 EULA。Docker build 中不应有
`OMNI_KIT_ACCEPT_EULA`、`PRIVACY_CONSENT` 或 EULA acceptance 文件。

用户已合法接受 EULA 后，未来的运行命令只能通过 `docker run -e
OMNI_KIT_ACCEPT_EULA=Y` 把 consent 传给**该容器进程**。它不会进入 image，
也不应被导出到宿主 shell。没有有效同意时，应停止而不是绕过该边界。

## 在具备 Docker 的独立主机上准备

在尝试 build 前，操作者必须先记录以下证据：

1. Docker Engine 与 BuildKit 可用，且 NVIDIA Container Toolkit 已配置；
2. host driver、`nvidia-smi`、Vulkan/RTX 和 Docker GPU passthrough 均在
   RTX 4090 上可用；
3. 选定的 Ubuntu 24.04 CUDA userspace base 是 immutable digest，而不是
   只写 tag；
4. 当前 checkout 是 RAMBO migration branch，Isaac Lab source 的实际 commit
   为 `ffff603eafc6b74264a5261cc0183d6a65390d78`；
5. 已明确保留 checkpoint、dataset、cache、凭据与历史 migration artifact 在
   build context 之外。

不要升级或在 image 内安装 host NVIDIA driver，不要使用 `--privileged`，也不
要把 host 的旧 Isaac Sim cache、Python path 或 EULA marker bind mount 进去。

## Deferred build command

以下命令仅供满足上述前提的未来操作者使用；在当前 migration host 上**不要执行**。
先以 registry 的独立核验结果填入 `RAMBO_BASE_IMAGE`。该变量必须包含 tag 与
digest，例如 `registry.example/ubuntu-rtx:24.04-cuda12.8@sha256:<digest>`；
这里的示例不是一个被本仓库认可的真实 digest。

```bash
RAMBO_BASE_IMAGE='registry.example/ubuntu-rtx:24.04-cuda12.8@sha256:<verified-digest>'

DOCKER_BUILDKIT=1 docker build \
  --build-arg RAMBO_BASE_IMAGE="$RAMBO_BASE_IMAGE" \
  --build-arg ISAACLAB_REPOSITORY='https://github.com/isaac-sim/IsaacLab.git' \
  --build-arg ISAACLAB_COMMIT='ffff603eafc6b74264a5261cc0183d6a65390d78' \
  --file Dockerfile.rambo60 \
  --tag rambo-isaac60-physx:deferred \
  .
```

`Dockerfile.rambo60` clones only the detached exact Isaac Lab commit, then runs
`scripts/setup_isaacsim60.sh --docker-build-metadata-only`。它会安装精确 Isaac
Sim / Torch / Isaac Lab 路径和必要的 bare transitive packages，并消费
`requirements/isaacsim60.lock` 强制非 editable wheel 版本，但不会请求 Isaac
Lab Newton optional extra。该 build mode 只核验 Python、包、lock、Isaac Lab
editable provenance 与 `pip check` allowlist：Docker build 本身通常没有 GPU，
所以它**不能**成为 CUDA、RTX renderer 或 PhysX runtime 通过的证据。它不接受
EULA。

Build 后先执行不启动 Kit 的 metadata 检查，再决定是否进行 GPU runtime
验收：

```bash
docker run --rm --entrypoint /bin/bash rambo-isaac60-physx:deferred -lc '
  python --version
  python scripts/verify_isaacsim60_install.py \
    --isaaclab-source /opt/IsaacLab-3.0.0-beta2.patch1 \
    --requirements-input requirements/isaacsim60.in \
    --requirements-lock requirements/isaacsim60.lock \
    --installer-script scripts/setup_isaacsim60.sh \
    --metadata-only
  python -c "import crl2, rambo; print(\"RAMBO_IMPORT_OK\")"
'
```

此命令不启动 Isaac Sim，因此不需要 EULA 环境变量。若版本、import 或
dependency metadata 与固定 contract 不一致，应将该 image 标记为失败，不应继续
替代 native evidence。

## Deferred GPU/PhysX validation

完成 metadata gate 后，先在有 GPU 的容器中运行默认 verifier；它必须看见 CUDA
设备才会成功。随后使用只读 checkpoint mount 和可写 artifact mount 运行最小官方
gate。下面也是未来操作者的命令，当前主机没有执行：

```bash
mkdir -p /workspace/container-artifacts

docker run --rm --gpus all \
  --env OMNI_KIT_ACCEPT_EULA=Y \
  --mount type=bind,src=/workspace/rambo-go2-policies,dst=/checkpoints,readonly \
  --mount type=bind,src=/workspace/container-artifacts,dst=/artifacts \
  --entrypoint /bin/bash \
  rambo-isaac60-physx:deferred -lc '
    cd /opt/rambo
    python scripts/verify_isaacsim60_install.py \
      --isaaclab-source /opt/IsaacLab-3.0.0-beta2.patch1 \
      --requirements-input requirements/isaacsim60.in \
      --requirements-lock requirements/isaacsim60.lock \
      --installer-script scripts/setup_isaacsim60.sh
    scripts/rambo/run60.sh scripts/rambo/official_physx_smoke.py \
      --scenario cartpole --steps 16 \
      --output-dir /artifacts/official-cartpole \
      --viz none
  '
```

`--viz none` 仍会启动必要的 Kit/RTX renderer path；它不是 `--headless` 的
别名。RAMBO commands 必须显式使用 `--viz none` 或 `--viz kit`，并拒绝
`--headless`、自定义 `--experience`、`--kit_args` 和未审计的 visualizer。

容器 runtime 的最小成功条件是 artifact 同时记录：

- configured `PhysxCfg` 和 `use_newton_actuators=false`；
- environment 创建后与结束前均为
  `isaaclab_physx.physics.physx_manager.PhysxManager`；
- RTX 4090 / CUDA / renderer evidence，而不是 CPU fallback；
- checkpoint allow-list、405/435 observation、18 action、时间和内存 contract；
- 3000-step RGB validation 时为 15,000 physics ticks 和 375 个 640×480 帧；
- Button recorder 时 action/observation/post-state/RGB/timestamp/checksum
  完整，并通过离线 validator。

容器内的成功不能取代原生 acceptance。只有 native 和 container 各自完整通过官方
gate、Quadruped、Biped、RGB、Button recorder、offline validator、clean restart
和实际 PhysX manager 断言后，才可以更新 Docker 状态或发布 image。

## 不执行的事项

- 不在当前 host 安装 Docker、Toolkit 或 driver；
- 不 build、run、tag、push 或发布任何 image；
- 不把 EULA、checkpoint、credentials、dataset、cache 或 validation output 写入 image；
- 不因为 image 中存在 Newton 相关 transitive package 而认定失败，也不允许实际
  Newton runtime；
- 不将 Docker success 声称为已经完成，直到上述独立验收 artifact 存在。
