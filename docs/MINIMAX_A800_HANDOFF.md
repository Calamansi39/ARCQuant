# A800 上的 MiniMax H3：部署包说明与实验要求

更新：2026-11-05。给接手的助手：你的任务是部署 MiniMax H3，完成 HiF4 W4A4 下的 VBench 验证和稀疏方法实验。DeepSeek 留在原机器，MAGI 由另一台 H20 负责。

**先验证单视频，再做稀疏方法，最后跑完整 Tiny。** 不自动遍历所有 FP4 格式，不把未完成方法或单视频分数写成完整实验成绩。

## 1. 这个包包含什么

包内 `README.md` 就是本说明；`MANIFEST.json` 记录代码版本、文件大小、SHA256 与资产范围。

| 文件 | 用途 |
| --- | --- |
| `restore.py` | 校验包内文件，把两个 Git 仓库还原到指定工作目录 |
| `bundles/ARCQuant.bundle` | 本次提交的完整可克隆 Git 历史；还原后目录名为 `ARCQuant`，可继续修改和提交 |
| `bundles/base_eval.bundle` | 固定版本外部评测器，解决它的 GitHub 地址无法直接读取的问题 |
| `reference_results/bf16/`、`reference_results/hifp4-w4a4/` | 已有 MiniMax Tiny 的汇总和 16 维评分 JSON，供核对；不是新机器的运行结果 |
| `asset_sources.json` | MiniMax、VBench 和评分资产的固定来源；实际下载清单仍在仓库 `configs/` |
| `MANIFEST.json` | 包内文件校验、仓库提交和历史结果来源 |

**包不含约 143.45 GB 模型权重、约 9.4 GB 评分资产、生成的视频或 Python/CUDA 环境。** 这些通过共享盘复用、单独传输或按清单下载。代码和评测器可离线还原；安装依赖和下载缺失资产仍需网络。原 ARCQuant 的 CUDA 子模块没有打进包，本次 HiF4 数值评测路径不需要编译它们。

评测器 bundle 供你的服务器之间私下传输，不要把整包或评测器历史再上传到公开 GitHub。包不包含服务器 SSH 密钥、登录凭据或旧虚拟环境。

## 2. 硬件与协议要求

| 项目 | 本轮设置 |
| --- | --- |
| GPU | 至少一张 **80 GB A800**，生成启动阈值 79000 MiB 空闲；40 GB 卡不能直接套用 |
| 评分 GPU | 空闲至少 16000 MiB；生成进程退出后再运行评分 |
| 主机内存 | 模型分阶段 CPU 卸载，必须检查可用 RAM；建议为单个 worker 预留约 256 GB 主机内存余量，实际峰值以单视频实测为准 |
| 磁盘 | 建议可用 ≥500 GB，用于模型、缓存、环境和视频；多个方法累积视频后需继续监控 |
| 模型 | `MiniMaxAI/MiniMax-H3`；T2VA 所需组件，不下载额外的 `transformer_ref` |
| 量化 | **HiF4 W4A4**，命令写 `--quant hifp4`；本轮不扩展 NVFP4/MXFP4 |
| 生成 | T2VA 流程只评视频；1344×768、124 帧、24 fps（约 5.17 秒）、50 步、seed 0 |
| 卸载 | `model_cpu`，文本编码器 → DiT → 视频 VAE 按组件切换；不是整模型同时放 GPU |
| Tiny | 固定 `vbench-tiny10-balanced-v1`，16 维、108 条去重提示词、**740 个视频** |
| 全集 | 944 条去重提示词、6220 个视频；另标 `standard16`，先完成 Tiny |
| 稀疏目标 | TEAL/WINA 各 50%、60%、70%、80%、90%，指非结构化激活元素置零目标 |

Tiny 每维 10 条提示词，普通提示词生成 5 个样本，闪烁维度每条先生成 25 个候选再筛选。沿用固定 manifest，不重新随机抽样。MiniMax 的画幅/帧数与 MAGI 的 1080p、10 秒不同，不要照抄 MAGI 的生成参数。

固定版本：

- MiniMax：`42ed227ee7df40d41602854ae760620d6eb651fe`。
- 评测器：`c452d6d3b5a5765ab66871bb1d18d8244f6f2dec`。
- VBench：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。
- Tiny manifest SHA256：`5823e6399ba7e31ba5303dc6d63526c30656e2c45a6ae1c7ade2482a1d23e958`。

当前 HiF4 是 DiT Linear 的数值假量化，权重仍用浮点张量保存；文本编码器/VAE 等保留其实现精度。它用于评估量化和稀疏后的生成质量，不能据此宣称真正 FP4 压缩存储或硬件稀疏加速。A800 不需要安装 MAGI 的 Hopper 专用扩展。

## 3. 解包和恢复代码

把 `.tar.gz` 和同名 `.sha256` 一起传到新机器。下面例子假设文件名为 `MiniMax_A800_20261105.tar.gz`：

```bash
sha256sum -c MiniMax_A800_20261105.tar.gz.sha256
tar -xzf MiniMax_A800_20261105.tar.gz
cd MINIMAX_A800_TRANSFER
python3 restore.py --check-only
python3 restore.py --workspace /your/workspace
cd /your/workspace/ARCQuant
git log -1 --oneline
```

需要 Python 3.10+、Git。还原工具只校验并克隆代码，不下载模型、不安装环境、不使用 GPU。它创建 `/your/workspace/ARCQuant/` 和其中的 `external/base_eval/`，不会把仓库子目录散落到工作路径；目标 `ARCQuant` 已存在时拒绝覆盖。

还原后位于 `deployment/minimax-a800` 开发分支，origin 指向 `https://github.com/Calamansi39/ARCQuant.git`；推送时使用目标机自己的授权方式，不改服务器账号。`MANIFEST.json` 中是这次打包的精确提交，不需要为了开跑先 pull 最新 main。

## 4. 准备模型、提示词和评分资产

先确定一个容量充足的位置。控制环境只负责下载和启动，生成/评分使用各自独立环境：

```bash
cd /your/workspace/ARCQuant
python3 -m venv external/envs/control
external/envs/control/bin/python -m pip install huggingface-hub
export ARCQUANT_ASSET_ROOT=/your/large-disk/arcquant-assets
# 若网络需要镜像，可在下载前设置：
# export HF_ENDPOINT=https://hf-mirror.com
```

如果 A800 能访问原共享盘，优先核对并复用：

```bash
export ARCQUANT_ASSET_ROOT=/mnt/kpfs_qyang/LBC/ARCQuant/assets
export ARCQUANT_MINIMAX_MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MiniMax-H3
test -r "$ARCQUANT_MINIMAX_MODEL/model_index.json"
```

共享挂载可选，不假设 A800 一定也能看到它。无法访问时使用前面自己的资产目录，**不设置旧服务器的模型路径**，然后执行：

```bash
external/envs/control/bin/python arc.py provision \
  --model minimax-h3 --benchmark vbench
external/envs/control/bin/python arc.py provision \
  --model minimax-h3 --benchmark vbench --verify-scorer-hashes --execute
```

已还原的评测器无需访问它的 GitHub 地址。`provision` 检查已有资产、下载缺失模型/提示词/评分权重并建立离线缓存；共享已有资产通过检查时跳过。模型清单含 61 个文件，总计 143,445,767,410 字节；不只是 DiT 权重，还包括文本编码器、VAE 等依赖。

如需从原机器传权重，使用 `rsync -aL --partial --append-verify SOURCE/ user@HOST:DEST/`，`-L` 复制软链接指向的实际文件。传完后设置 `ARCQUANT_MINIMAX_MODEL=DEST` 并执行资产检查。不能只传旧 `assets/models/minimax-h3` 软链接。大文件不放 Git。

## 5. 创建 A800 的生成和评分环境

先检查 `nvidia-smi`、`free -h`、`df -h`、Python 和 CUDA toolkit。需要目标机器已经可用的 CUDA Torch + 匹配 torchvision；驱动要支持这个 Torch CUDA 版本。

源机器的生成依赖包括 diffusers 0.40.0、transformers 5.9.0；评分固定 transformers 4.33.2，因此必须分环境。不要把根目录旧版 `requirements.txt` 装进视频环境。

```bash
cd /your/workspace/ARCQuant
python3 -m benchmarks.setup_video_env --role h3 \
  --base-python /path/to/working-torch/bin/python \
  --destination /your/envs/minimax-h3 --execute
python3 -m benchmarks.setup_video_env --role scorer \
  --base-python /path/to/working-torch/bin/python \
  --destination /your/envs/vbench-a800 \
  --cuda-arch 8.0 --cuda-home /path/to/matching-cuda --execute
export ARCQUANT_H3_PYTHON=/your/envs/minimax-h3/bin/python
export ARCQUANT_VBENCH_PYTHON=/your/envs/vbench-a800/bin/python
```

工具保留基础 Torch/torchvision 版本，安装其它依赖到新 venv，并编译固定 Detectron2。`nvcc` 的 CUDA 主次版本必须和基础 Torch 一致。没有合适基础 Torch 时，先按本机驱动安装兼容版本，再执行上述命令；本包不承诺任意旧驱动都能直接运行。

安装系统 `ffmpeg`、`ffprobe`；若不在 PATH，设置 `ARCQUANT_FFMPEG_BIN=/path/to/ffmpeg/bin`。环境变量存到新仓库的 `external/local_env.sh`，以后先 source。不要直接复用旧机器虚拟环境的路径或安装到共享环境。

检查资产和导入，不占 GPU：

```bash
external/envs/control/bin/python arc.py doctor \
  --model minimax-h3 --benchmark vbench \
  --native-dry-run --hash-scorer --skip-gpu
```

`--skip-gpu` 的总状态可为 `unchecked`、退出码 2，表示还没有检查空卡；逐项确认无其它 blocked。这个检查不能代替实际单视频运行。

## 6. 等空卡运行：先基线 smoke，再 WINA smoke

下面不指定 GPU 编号时，会在本机所有可见卡中选一张无计算进程、显存达标的空卡。可增加 `--gpu-ids 0,1,2,3` 限定候选范围。只协调本机同账号 ARCQuant 的协作锁，不杀他人进程，也不是管理员级资源预约。

```bash
cd /your/workspace/ARCQuant
mkdir -p runs/a800-hif4-smoke-v1
nohup external/envs/control/bin/python -u arc.py video-queue \
  --model minimax-h3 --quant hifp4 --method none --profile smoke --steps 50 \
  --out-dir runs/a800-hif4-smoke-v1 --execute \
  > runs/a800-hif4-smoke-v1/launcher.log 2>&1 < /dev/null &
echo $! > runs/a800-hif4-smoke-v1/launcher.pid
```

去掉 `--execute` 就只打印计划。smoke 为一个真实视频和一个调试维度的评分，检查画面、有限数值、峰值显存和日志。基线成功后再运行：

```bash
mkdir -p runs/a800-wina50-smoke-v1
nohup external/envs/control/bin/python -u arc.py video-queue \
  --model minimax-h3 --quant hifp4 --method wina --sparsity 0.5 \
  --profile smoke --steps 50 --out-dir runs/a800-wina50-smoke-v1 --execute \
  > runs/a800-wina50-smoke-v1/launcher.log 2>&1 < /dev/null &
echo $! > runs/a800-wina50-smoke-v1/launcher.pid
```

**不要同时盲目运行上面两条长作业。** 第一项验证成功后才运行第二项。当前 WINA 包装会在生成日志中打印 `masked_elements`、`total_elements`；确认实际经过选通且比率合理，不能只凭参数非零就认为稀疏已经生效。

方法验证通过后，使用新目录，将 profile 改为 `tiny`：

```bash
external/envs/control/bin/python arc.py video-queue \
  --model minimax-h3 --quant hifp4 --method wina --sparsity 0.5 \
  --profile tiny --steps 50 --out-dir runs/a800-wina50-tiny-v1 --execute
```

它依次进行单视频生成检查、740 视频生成、16 维评分。随后按 0.6、0.7、0.8、0.9 用不同目录补齐，其它条件保持一致。同一配置、同一源码可重启续跑；改代码、环境身份、步数或方法时用新目录。`status.json` 和 `*-generate.log` / `*-score.log` 是进度入口。

本轮尚无 A800 真正启动/安装的验证结论；包内检查只证明代码能还原、命令可解析以及源机器上的资产预检有效。

## 7. 已有结果和适配欠账

包内历史 MiniMax Tiny 参考分数：

| 精度 | 方法/稀疏度 | 总分 | 质量 | 语义 |
| --- | --- | ---: | ---: | ---: |
| BF16 | 无 / 0% | 81.2504% | 84.9953% | 66.2711% |
| HiF4 W4A4 | 无 / 0% | 80.8803% | 84.7099% | 65.5619% |

来源为旧服务器 `base_eval/results/MiniMax-H3/vbench_standard/<tag>/tiny/scoring/static-coverage-v2/`。本次可读取汇总和各维评分 JSON，但没有找到同目录上游的完整 `generation_config.json` / `gen_summary.json`，原始视频也不在本包内。因此这些是**历史参考**；在把它们与新方法作为严格同协议对照前，需要找回并核对原始生成配置。若找不到，建立本轮同配置的 HiF4 对照，必要时补 BF16；不要直接假设历史参数与新默认值完全相同。

当前方法状态：

- **WINA 已有选择规则适配入口，尚无本轮正式稀疏成绩。** `model/video_sparse_runtime.py` 在权重量化前缓存原始输入列 L2 范数，在线按每个 token 的 `abs(x) × norm` 选 top-k，再交给原 HiF4 量化层。包装入口为 `benchmarks/run_minimax_sparse.py`。
- **TEAL 视频路径尚未实现完整校准与接线。** 要对真实 MiniMax 去噪过程的层/投影/timestep 采样，生成可复用直方图和阈值，再在线置零；不要复用 DS 文本直方图，也不要把简单 top-k 命名为 TEAL。
- 两者都是非结构化激活稀疏，不能用 2:4 替代。WINA 当前是 DiT 适配，不自动等价于原论文完整 transform/greedy 设置。
- 接手助手应检查 WINA 的覆盖清单、掩码顺序、eval 模式、0% 时与同量化基线一致性、实际零率统计和开销；测试不能仅验证输出形状。再逐步补视频 TEAL，最后完成五档各自的完整 Tiny。

## 8. 结果怎么交付

只有 740 个视频通过校验、16 维评分和闪烁筛选覆盖完整，才把 `tiny/vbench_summary.json` 的 `tiny_scores.total/quality/semantic` 填进结果表。不要使用简单维度平均数代替官方聚合；Tiny 必须标为 Tiny，不能冒充官方全集。

表头：`模型 | benchmark/协议 | 精度 | 方法 | 稀疏度 | 总分 | 质量 | 语义`。另外保存代码提交、模型/数据哈希、生成与评分配置、覆盖/实际零率、GPU 型号、峰值显存和耗时。代码与说明提交到你的开发分支，权重、视频和环境不进 Git。

若 A800 也能看到 LBC，在 `codexx1.md` 追加实际进展；看不到时在本机维护 `A800_PROGRESS.md`，连同小体积结果 JSON/CSV 回传。其它机器任务保持运行。

## 可直接发给 A800 助手的任务

> 请读取解包后的 README.md，先校验并用 restore.py 恢复到完整 ARCQuant 目录。你负责 MiniMax H3：核对 A800 是否为 80GB、准备模型/评分资产与独立环境，先跑 HiF4 单视频，再验证 WINA 50%。结合现有实现完善方法覆盖、零率统计和必要测试，补齐视频 TEAL 校准与接线，逐步跑 HiF4 下 50/60/70/80/90% 的完整 VBench Tiny。只等空卡，不停止其他人的任务。历史评分只作参考，严格对照需确认生成配置一致；将实现、完整结果与实际进度留档并提交到你的开发分支。
