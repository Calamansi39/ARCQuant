# 三台服务器分别跑三个模型

目标分工：**当前服务器跑 DeepSeek，另一台 H20 跑 MAGI-2，A800 跑 MiniMax H3。**
各机器独立下载、等待空卡和保存结果，代码都来自这个 ARCQuant 仓库。

## 先确认机器够不够用

| 机器任务 | 当前生成路径 | 模型文件体积约数 | 推荐预留磁盘 |
| --- | --- | --- | --- |
| 当前机器 / DeepSeek | 保留现有两卡任务与 TEAL/WINA 队列 | 使用已有权重 | 使用现有目录 |
| H20 / MAGI-2 | **两张约 140 GiB H20**；每卡空闲 ≥142000 MiB；Hopper 扩展 | 307 GB | ≥800 GB，包含缓存、环境和视频 |
| A800 / MiniMax H3 | **一张 80 GB A800**；空闲 ≥79000 MiB；模型分阶段 CPU 卸载 | 145 GB，仅 T2VA 所需组件 | ≥500 GB，包含缓存、环境和视频 |

生成结束后释放生成进程，评分另等一张空闲 ≥16000 MiB 的卡。两套视频任务均需大量主机内存，部署时检查剩余 RAM、磁盘和驱动；表中是目前代码路径的门槛，不是任何型号都能成功运行的保证。

**96 GB H20 不能直接使用当前两卡 MAGI 路径；40 GB A800 不能直接使用当前单卡 MiniMax 路径。** 若实际卡型不同，需要另外调整并验证分片/卸载配置，不能只调低空闲显存阈值。

## 1. 在每台新机器克隆完整仓库

```bash
cd /your/workspace
git clone https://github.com/Calamansi39/ARCQuant.git
cd ARCQuant
export ARCQUANT_ASSET_ROOT=/your/large-disk/arcquant-assets
```

目录必须是 `/your/workspace/ARCQuant/`。权重和输出不进入 Git。
控制 CLI 用 Python 3.10+；下载资产需要 `huggingface-hub`，可单独安装到控制环境。
视频环境按第 3 步准备，根目录旧版 `requirements.txt` 用于原来的模型路径。

## 2. 准备模型、提示词、评分权重和评测器

### 评测器迁移包

`base_eval` 是另一个固定版本的 Git 仓库，当前远程访问未验证通过。
源机器可从已有授权副本导出完整 Git bundle：

```bash
python -m benchmarks.export_backend --output runs/deployment/base_eval.bundle
```

输出 `.bundle` 和同名 `.json`（版本、字节数、SHA256）。导出工具会在临时目录还原一份，验证可以独立克隆；不会修改源评测器。文件已存在时拒绝覆盖。
**私下把两份文件传到新机器**，如 `/your/transfer/base_eval.bundle`，无需公开上传评测器，也无需拷贝整套旧服务器路径。

在目标机器校验收到的包：

```bash
python - /your/transfer/base_eval.bundle <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); m = json.loads(p.with_suffix('.json').read_text())
assert p.stat().st_size == m['bytes']
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['sha256']
print('bundle checksum OK:', m['revision'])
PY
```

### 每台机器只准备它负责的模型

```bash
# 在 A800 机器执行
python arc.py provision --model minimax-h3 --benchmark vbench \
  --backend-source /your/transfer/base_eval.bundle --execute

# 在 H20 机器执行
python arc.py provision --model magi-2-preview --benchmark vbench \
  --backend-source /your/transfer/base_eval.bundle --execute
```

工具按 [`experiments.json`](../configs/experiments.json) 的固定版本下载，检查已有文件，补齐 VBench 提示词、官方评分源码、约 9.4 GB 评分权重和离线缓存。再次运行会跳过已就绪项。MAGI 还会准备官方源码。

若两台服务器挂载同一共享盘，优先复用完整权重：

```bash
# 在对应机器设置其中一项；路径必须在该机器真实存在
export ARCQUANT_MINIMAX_MODEL=/shared/models/MiniMax-H3
export ARCQUANT_MAGI2_MODEL=/shared/models/MAGI-2-preview
```

若通过 rsync 传权重，用 `rsync -aL --partial --append-verify SOURCE/ user@HOST:DEST/`，只传目标模型；`-L` 复制链接所指的文件。
当前源服务器的 `assets/models/` 多数是软链接，复制链接本身会在新服务器失效。评分缓存也有软链接；可以由 `provision` 在新机器下载并重建缓存映射。无需迁移整份 Hugging Face 缓存、DS 权重或本机虚拟环境。

## 3. 分开准备生成环境和评分环境

新工具 `python -m benchmarks.setup_video_env` 默认打印安装计划；加 `--execute` 才安装。
它基于目标机器**已有、能正确导入的 CUDA Torch + torchvision 环境**创建独立 venv，锁定并复用原来的 Torch 版本，按固定源码安装模型依赖。目标目录必须是新的，或由该工具以相同配置创建的目录。

### A800：MiniMax 和评分

```bash
python -m benchmarks.setup_video_env --role h3 \
  --base-python /path/to/torch-env/bin/python \
  --destination /your/envs/h3 --execute

python -m benchmarks.setup_video_env --role scorer \
  --base-python /path/to/torch-env/bin/python \
  --destination /your/envs/vbench --cuda-arch 8.0 \
  --cuda-home /path/to/matching-cuda --execute

export ARCQUANT_H3_PYTHON=/your/envs/h3/bin/python
export ARCQUANT_VBENCH_PYTHON=/your/envs/vbench/bin/python
```

评分使用独立的 Transformers 4.33.2，并编译固定提交的 Detectron2。CUDA toolkit 的主次版本要与 Torch 一致；工具会提前检查。安装依赖和编译不占 GPU，但会消耗 CPU、内存及网络。不同 Torch/驱动组合仍须通过单视频验证；工具不会替机器安装驱动。

### H20：MAGI 的 Hopper 环境

MAGI 需要 Python 3.12、`flash_attn_interface`、`magi_attention`、`magi_compiler`，不能只安装通用 Transformers。
有现成兼容运行环境时：

```bash
python -m benchmarks.setup_video_env --role magi \
  --base-python /path/to/hopper-runtime/bin/python \
  --destination /your/envs/magi --execute
export ARCQUANT_MAGI2_PYTHON=/your/envs/magi/bin/python
```

没有现成环境时，使用 `external/MAGI-2-preview/Dockerfile` 中固定的 Hopper 构建方案，先构建镜像：

```bash
docker build -t arcquant-magi-runtime external/MAGI-2-preview
```

随后在具备 NVIDIA Container Toolkit 的容器中挂载 ARCQuant、资产、输出目录，完成上述环境准备和队列启动。队列需要在容器中看到 `nvidia-smi` 和 GPU。与宿主机其他 ARCQuant 队列共用卡时，把宿主机 `/tmp/arcquant-gpu-locks-UID` 挂到容器同名路径并使用相同 UID，以共享协作锁。
镜像编译、驱动兼容和第一次生成仍需在目标 H20 上验证，本轮没有把它标作“已经部署”。

H20 的评分环境使用相同 `--role scorer` 命令，将架构改成 `--cuda-arch 9.0`，也设置 `ARCQUANT_VBENCH_PYTHON`。
两台机器都要有 `ffmpeg` 和 `ffprobe`；不在 PATH 时设置 `ARCQUANT_FFMPEG_BIN=/path/to/ffmpeg/bin`。
把本机环境变量保存进 `external/local_env.sh`，以后先 `source external/local_env.sh`。该文件不提交 Git。

## 4. 一条队列命令：预检 → 等卡 → 单视频 → Tiny → 评分

```bash
# A800：先打印计划，不占 GPU
python arc.py video-queue --model minimax-h3 --out-dir runs/h3-hif4-tiny
# 确认环境已准备后启动，自动等待一张空卡
nohup python -u arc.py video-queue --model minimax-h3 \
  --out-dir runs/h3-hif4-tiny --execute > h3-queue.log 2>&1 < /dev/null &

# H20：自动等待两张空卡，再使用 torchrun 启动 MAGI
nohup python -u arc.py video-queue --model magi-2-preview \
  --out-dir runs/magi-hif4-tiny --execute > magi-queue.log 2>&1 < /dev/null &
```

默认只跑 **HiF4 W4A4、不加稀疏**，不自动遍历其他 FP4 格式。MiniMax 默认 50 步，MAGI 默认 **50+5 步**，用于匹配已有 BF16 Tiny 实验；MAGI 原生默认 100+5 步，若使用 `--steps 100`，基线和方法都要使用一致设置并新建输出目录。

- `--gpu-ids 0,1,2,3`：允许使用的物理卡号，队列从中选空卡；默认检查全部可见卡。
- `--poll-seconds 30`：轮询间隔。只有无计算进程、利用率 ≤5%、空闲显存足够的卡才会入选；取得本机协作锁后再次核查。
- `--profile smoke`：只生成并评分 1 个视频，用于调试。
- `--profile tiny`：固定 **108 条去重提示词 / 740 个视频 / 16 维评分**；沿用固定评测器里的 Tiny manifest。
- `--profile official`：944 条去重提示词 / 6220 个视频的官方完整集。Tiny 与官方全集分别标注。
- `--stage generate` / `--stage score`：只生成或只评分；同一配置、同一输出目录可分阶段继续。

这些锁只协调同机器同账号的 ARCQuant 任务，不能向集群保证预约优先级。不会杀掉他人进程，也不通过申请显存来持续空占卡。中断会结束本队列的子进程并释放锁；启动失败写入错误日志。

## 5. 看进度、续跑、区分方法

```bash
cat runs/h3-hif4-tiny/status.json
tail -n 30 runs/h3-hif4-tiny/tiny-generate.log
```

`status.json` 记录主机、PID、阶段、使用的 GPU、错误信息。视频生成和评分的原始进度在各阶段日志及其输出目录中；最终评分在 `tiny/vbench_summary.json` 的 `tiny_scores`，官方全集在 `official/vbench_summary.json` 的 `official_scores`。
完全相同的命令可以重启续跑，底层评测器会核验缓存；改模型、方法、稀疏度、步数或入口实现时必须使用新输出目录，以免串结果。

MiniMax 可以显式尝试 WINA 选择规则的 DiT 适配：

```bash
python arc.py video-queue --model minimax-h3 --method wina --sparsity 0.5 \
  --out-dir runs/h3-hif4-wina50 --execute
```

WINA 在这里按每个 token 的 `abs(x) × 权重列范数` 选通，再执行 HiF4 假量化；50/60/70/80/90% 是非结构化激活置零目标，**没有施加 2:4**。它是视频架构适配，不能称为已经完整复现原论文。先过 GPU 单视频检查，再检查实际置零与质量，最终才能形成可比较成绩。
MAGI 的 WINA grouped refiner 以及视频 TEAL 尚未完整接好，新队列会明确拒绝这些组合。HiF4 当前使用假量化和稠密 GEMM，结果用于质量比较，不代表硬件 FP4/稀疏内核加速。

## 本轮已验证到哪里（2026-11-05）

仓库已加入队列、环境准备和评测器导出工具；测试覆盖单/双卡选择、卡容量不足、异常释放锁、输出身份与完整性检查、Git bundle 独立还原。本机两种模型通过真实文件与 CPU dry-run 检查，评分资产含缓存映射共 29 项验证通过；环境基本导入通过，但 MAGI 提示可选通信扩展警告，仍需 GPU 单视频验证。新环境安装工具在本轮只验证了计划生成，没有在另一台机器实际安装。

实际新服务器上的下载、安装和 GPU 运行，需要拿到两台机器的 SSH 地址、登录用户和部署目录后完成。源机器上的 DeepSeek 任务保持运行。
