# MAGI：共享存储下的另一台 H20 评测交接

更新时间：2026-11-05 16:48（Asia/Shanghai）。运行数字是此时快照，接手后先刷新。

**给接手的助手：请在你所在的 H20 服务器执行。目标是复用共享盘资产，推进 MAGI-2 的 HiF4 基线及稀疏方法适配，最终得到完整、可比较的 VBench 成绩。先读这份文档，无需翻历史对话。**

用户已说明两台机器共享挂载存储，`/mnt/kpfs_qyang/LBC` 应该可见。优先核实并复用已有文件；不要默认重新下载几百 GB 权重。当前服务器 `10-0-2-5` 专注 DeepSeek，不要从另一台机器改动或停止它的队列。

## 1. 当前实验分工与实际进度

| 机器/模型 | 任务 | 当前状态 | 接下来做什么 |
| --- | --- | --- | --- |
| 当前机器 `10-0-2-5` / DeepSeek V4 Flash | HiF4 W4A4 + TEAL/WINA，50/60/70/80/90% | WINA 50% MATH-500 为 **120/500**；另外 9 组由连续双卡队列等卡 | 原机器继续跑；本交接不重复启动 DS |
| 另一台 H20 / MAGI-2-preview | VBench Tiny 同协议 BF16、HiF4、后续稀疏方法 | BF16 Tiny 已完整；HiF4 Tiny **399/740 条完成记录**，仍有 worker 在生成；TEAL/WINA 尚无完整结果 | 先识别现有任务，补齐/接续 HiF4；利用另一批卡完善方法 |
| A800 / MiniMax H3 | VBench Tiny HiF4、后续 WINA/TEAL 适配 | 迁移入口已准备；本轮未核验 A800 上部署状态 | 另行由 A800 负责，本机不重复加载 MiniMax |

DS 当前仅这个连续批次覆盖 MATH-500；未来 GPQA Diamond、LongBench v2、LiveCodeBench 的方法全量成绩仍需补齐。不能把 MATH-500 排队描述成所有文本 benchmark 已排队。

MAGI 已有 BF16 Tiny：总分 **80.0008%**，质量 **84.8416%**，语义 **60.6373%**。这是完整 Tiny 子集的 VBench 分数，不是准确率，也不是官方全提示词榜单成绩。HiF4 尚无最终分数。

## 2. 本轮必须保持的评测设置

| 项目 | 要求 |
| --- | --- |
| 模型 | `sand-ai/MAGI-2-preview`，固定版本见下表 |
| 任务 | 文本生成视频；输出 1920×1080，10 秒，25 fps |
| 量化 | **HiF4 W4A4**，CLI 写作 `hifp4` / `w4a4`；不要自动遍历 NVFP4、MXFP4 |
| 对照 | 已有 `bf16-50p5`；无需无故重跑 |
| 采样步数 | preview **50** + refiner **5**；匹配现有 BF16。原生默认 100+5 不能混入本组比较 |
| 并行 | 每个生成 worker 两张 H20，CP=EP=2 |
| 随机数 | `seed=0`，沿用 `sha256-prompt-sample-v1` 的逐视频 seed |
| 提示词 | 不做 prompt enhancement；沿用固定 Tiny manifest，不能重新随机抽样 |
| Tiny 规模 | 16 维，每维 10 条提示词；去重 108 条；普通提示词 5 个样本，闪烁维度每条先生成 25 个候选；共 **740 个视频** |
| Tiny 标识 | `vbench-tiny10-balanced-v1`；不是 108 个视频或只做一个维度 |
| 全集 | 944 条去重提示词、6220 个视频；需要扩展时另记为 `standard16` |

当前 HiF4 覆盖 DiT 的 dense/grouped/routed GEMM；router/mHC 控制、embedding/adapter、文本编码器、VAE 保留原精度。它是**数值假量化**：权重仍占 BF16/FP32 存储，运算仍主要使用稠密 GEMM。可以比较量化质量，不能据此报告硬件 FP4 或稀疏内核的加速。

后续稀疏网格：TEAL、WINA 各 **50%、60%、70%、80%、90%**。百分比表示目标激活元素置零率，**不施加 2:4**。SOW 的 2:4 / share-index 是另外的实验线，单独实现和命名。

## 3. 已有文件在哪里

| 内容 | 共享路径 |
| --- | --- |
| ARCQuant 主仓库 | `/mnt/kpfs_qyang/LBC/ARCQuant`，交接时提交 `ebe36a2`，GitHub `Calamansi39/ARCQuant` |
| MAGI 权重 | `/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview` |
| ARCQuant 固定评测器 | `/mnt/kpfs_qyang/LBC/ARCQuant/external/base_eval`，提交 `c452d6d3b5a5765ab66871bb1d18d8244f6f2dec` |
| 正在产出旧结果的评测器 | `/mnt/kpfs_qyang/zhaolianchao/base_eval`，交接时提交 **`7a3992265000fbd03fac2e2361ec1a7d85573715`**，含动态 worker |
| MAGI 官方源码 | `/mnt/kpfs_qyang/LBC/ARCQuant/external/MAGI-2-preview`；另有 `/mnt/kpfs_qyang/zhaolianchao/work/MAGI-2-preview` |
| VBench 官方源码/提示词 | `/mnt/kpfs_qyang/zhaolianchao/data/vbench/VBench` |
| VBench 评分权重 | `/mnt/kpfs_qyang/zhaolianchao/data/vbench_cache` |
| Hugging Face 离线缓存 | `/mnt/kpfs_qyang/zhaolianchao/data/hf_home` |
| Tiny manifest | `/mnt/kpfs_qyang/zhaolianchao/base_eval/vbench_tiny_manifest.json`；ARCQuant 固定评测器中有相同副本 |
| 旧结果根目录 | `/mnt/kpfs_qyang/zhaolianchao/base_eval/results/MAGI-2-preview/vbench_standard` |
| 已完成 BF16 | 上述目录下 `bf16-50p5/tiny/vbench_summary.json` |
| 未完成 HiF4 | 上述目录下 `hifp4-w4a4-50p5/tiny/`，视频缓存在同一 tag 下 `_cache/` |
| 旧 MAGI 调度状态 | `/mnt/kpfs_qyang/zhaolianchao/base_eval/results/MAGI-2-preview/.dynamic_magi2/tiny16/status.json` |
| DS 连续队列状态 | `/mnt/kpfs_qyang/LBC/ARCQuant/runs/continuous_sparse_sweep_20261105_161852/status.json` |
| 迁移预检证据 | `/mnt/kpfs_qyang/LBC/ARCQuant/runs/deployment/validation/` |

两个 `base_eval` 目录**不是相同提交**。`arc.py video-queue` 校验 `c452d6d...`；不能把它的 `--base-eval-root` 随意指向 `7a39922...` 来绕过版本检查。旧视频续跑应使用原来生成它的版本和配置；不要改缓存身份文件来强行兼容。

固定版本：

- 模型：`2dea51b64db47ee5b4402d36fd90829a0c58913b`。
- MAGI 官方运行源码：`f68a0f9bbccbea56e0177a9bd912abc3b18ffe61`。
- VBench：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。
- Tiny manifest SHA256：`5823e6399ba7e31ba5303dc6d63526c30656e2c45a6ae1c7ade2482a1d23e958`；本次核对两个副本一致。

## 4. 先在目标 H20 检查这几件事

```bash
cd /mnt/kpfs_qyang/LBC/ARCQuant
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,compute_cap --format=csv
free -h
df -h /mnt/kpfs_qyang/LBC /tmp
readlink -f assets/models/magi-2-preview
readlink -f assets/datasets/vbench
test -r /mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview/preview/model.safetensors.index.json
```

需要两张约 **140 GiB H20**，每卡空闲至少 **142000 MiB**；评分另需一张空闲至少 16000 MiB 的卡。96 GB H20 不能直接运行当前两卡方案，调低启动阈值不能降低实际显存需求。

主网络原始权重约 212.44 GiB，两卡切分后每卡约 111.14 GiB；refiner 约 12.56 GiB、文本编码器约 51.75 GiB。组件分阶段卸载，但 CPU 内存也会有权重及备份，不应只看 GPU；新机器必须记录可用 RAM，并通过真实单视频测出峰值。权重整体约 307 GB，已有完整共享副本时无需再下载。

共享的是文件，不是 GPU、进程、`/tmp`、驱动或本地 `/opt` 环境。同一个 GPU 编号和 PID 在不同主机上不是同一资源，不能根据共享状态文件的 PID 直接在新机器 `kill`。

### 环境优先复用，但必须验证

```bash
source /mnt/kpfs_qyang/LBC/ARCQuant/external/local_env.sh
export ARCQUANT_ASSET_ROOT=/mnt/kpfs_qyang/LBC/ARCQuant/assets
export ARCQUANT_MAGI2_MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview
export PATH="$ARCQUANT_FFMPEG_BIN:$PATH"
export PYTHONDONTWRITEBYTECODE=1
command -v ffmpeg
command -v ffprobe
test -x "$ARCQUANT_MAGI2_PYTHON"
test -x "$ARCQUANT_VBENCH_PYTHON"
CUDA_VISIBLE_DEVICES='' "$ARCQUANT_MAGI2_PYTHON" -c '
import sys, torch, flash_attn_interface, magi_attention, magi_compiler
from transformers import Qwen3_5TextModel
from magi_attention.api import flex_flash_attn_func
from magi_attention import magi_attn_ext
assert sys.version_info[:2] == (3,12)
print(sys.executable, torch.__version__, torch.version.cuda)
'
python arc.py doctor --model magi-2-preview --benchmark vbench \
  --native-dry-run --hash-scorer --skip-gpu
```

`doctor --skip-gpu` 通常以 `unchecked` / 退出码 2 结束，表示尚未检查空卡；阅读逐项状态，不能把它简单当成全部失败，也不能忽略其中真正的 `blocked`。

**已发现的具体迁移风险：** 共享的 `.../base_eval/.venvs/magi2-h20/bin/python` 指向 `/opt/miniconda3/envs/sglang-ready/bin/python`。目标机器没有这个本地基础解释器，或者基础环境版本不同，就不能直接复用它。不要在共享 venv 上强行重装或改链接，这会影响其他服务器。改用目标机自己的环境并设置 `ARCQUANT_MAGI2_PYTHON`；环境安装方案见 `ARCQuant/docs/MULTI_SERVER_DEPLOYMENT.md` 和 `benchmarks/setup_video_env.py`。MAGI 需要官方 Hopper 扩展，评分需独立 Transformers 4.33.2 与兼容的 Detectron2。

本机检查已通过 70 个索引权重分片、108 prompts / 740 videos 计划、29 项评分资产与缓存映射。MAGI 导入有可选通信扩展警告，不能只凭“import 成功”就省略目标机器的 GPU 验证。

## 5. 先识别正在运行的 MAGI，再决定接续还是做适配

```bash
cat /mnt/kpfs_qyang/zhaolianchao/base_eval/results/MAGI-2-preview/.dynamic_magi2/tiny16/status.json
ps -eo pid,args | rg 'magi2_vbench_dynamic|magi2_vbench_worker|gen_vbench_magi2'
```

本次共享状态是 `running`，两组 worker，399 条完成记录，`generation_done=false`、`evaluation_done=false`。状态没有主机名，不能据此断言它在当前机器；需要将 worker 状态里的 GPU UUID 与本机 `nvidia-smi` 比对，再检查实际进程和日志。`completion_record_count` 是中途计数，最终还要逐视频验证；`verified_complete_videos=0` 也不等于没有生成视频。

按查到的状态处理：

1. **原任务仍在正常更新：**保留它。另一批卡优先做独立单视频验证、WINA/TEAL 适配与校准；不要无故从头再生成 740 个相同 HiF4 视频。
2. **原调度器及其 worker 已退出：**优先使用原评测器、原 tag、原步数和原缓存接续；重启前确认锁已释放。原数据完整保留。
3. **需要另一台机器参与同一缓存加速：**旧 `magi2_vbench_worker.py` 有逐视频领取锁，但总调度器有共享的 `.scheduler.lock`，不能启动第二个总调度器强抢同一目录。必须先验证共享文件系统的跨主机 `flock`、生成身份一致、唯一 worker ID、设备锁及统一最终发布流程，再接入独立 worker。当前 ARCQuant 新队列不是开箱即用的跨主机合并调度器，不要假设两个命令会自动分担任务。

原调度器的参数参考如下；这段**仅做 CPU 预检**，不占卡、不改旧输出：

```bash
export MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview
export MAGI2_REPO=/mnt/kpfs_qyang/LBC/ARCQuant/external/MAGI-2-preview
export MAGI2_PYTHON="$ARCQUANT_MAGI2_PYTHON"
export VBENCH_PYTHON="$ARCQUANT_VBENCH_PYTHON"
export TAGS=hifp4-w4a4-50p5
export HF_HOME=/mnt/kpfs_qyang/zhaolianchao/data/hf_home
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VBENCH_REPO=/mnt/kpfs_qyang/zhaolianchao/data/vbench/VBench
export VBENCH_CACHE_DIR=/mnt/kpfs_qyang/zhaolianchao/data/vbench_cache
export TORCH_HOME="$VBENCH_CACHE_DIR/torch_hub"
export PYTHONPATH="$VBENCH_REPO:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' "$MAGI2_PYTHON" \
  /mnt/kpfs_qyang/zhaolianchao/base_eval/magi2_vbench_dynamic.py \
  all 0,1 --tiny --num-inference-steps 50 --refiner-num-inference-steps 5 --dry-run
```

若 Git 报跨用户 ownership，用对子进程生效、限定这些确切目录的 `safe.directory` 配置；不要全局信任 `*`。上面卡号只是 CPU 预检占位，真正运行前按目标机器可用卡选择。

原调度器启动时要求指定卡组已经空闲；虽然名字含 dynamic，它**不会在启动阶段替你等待忙卡**。去掉 `--dry-run` 接续前，由接手助手加合适的等待层，并确认不会与同机其他队列重复分配。ARCQuant 的本机 `/tmp/arcquant-gpu-locks-UID` 和旧评测器的 `.gpu_locks/host-*` 是两套锁协议，不能只看其中一套锁就断言没有冲突。

## 6. 新实验怎样启动

新环境先用独立目录做一个 HiF4 视频；既不覆写旧缓存，也不重复全量生成。以下命令默认只打印计划，加 `--execute` 才等待本机空卡并运行。

```bash
cd /mnt/kpfs_qyang/LBC/ARCQuant
MAGI_HOST=$(hostname -s)
MAGI_RUN=/mnt/kpfs_qyang/LBC/magi_runs/$MAGI_HOST/hif4-smoke-50p5-v1
mkdir -p "$MAGI_RUN"
python arc.py video-queue --model magi-2-preview \
  --quant hifp4 --method none --profile smoke --steps 50 --refiner-steps 5 \
  --out-dir "$MAGI_RUN"
nohup python -u arc.py video-queue --model magi-2-preview \
  --quant hifp4 --method none --profile smoke --steps 50 --refiner-steps 5 \
  --out-dir "$MAGI_RUN" --execute > "$MAGI_RUN/launcher.log" 2>&1 < /dev/null &
echo $! > "$MAGI_RUN/launcher.pid"
```

默认从本机所有可见物理卡中等待两张合格空卡；也可以两条命令都加 `--gpu-ids 0,1,2,3` 限定候选卡。检查 `status.json`、`smoke-generate.log`、`smoke-score.log`；预期产生一个真实视频及 subject_consistency 调试分数。通过不等于完整 Tiny 成绩，也不保证 16 维评分全通过。

**只有确认需要新的独立完整基线时**，使用新目录把 `--profile smoke` 改成 `--profile tiny`，其他设置保持一致。它会完成单视频生成检查、740 视频生成和完整评分。不要把新队列的 `--out-dir` 指向旧 `hifp4-w4a4-50p5` 目录；新旧输出身份格式不兼容。也不要只改 `--quant bf16` 就无故重跑已有对照。

共享运行时 `/tmp` 的编译缓存和 GPU 锁各机器独立；不同实验、主机使用独立输出目录。相同实验续跑必须保留相同参数和代码版本；实现变动用新目录。

## 7. 另一批 H20 上优先做哪些方法适配

**现状必须说清楚：MAGI 稀疏方法还没完整实现，不能直接承诺已经能跑十组完整成绩。**

| 文件 | 现有内容/用途 |
| --- | --- |
| `ARCQuant/model/magi2_sparse_runtime.py` | WINA 试验实现：普通 Linear、preview routed experts；grouped refiner 标为 pending |
| `ARCQuant/benchmarks/run_magi2_sparse.py` | 给原生 MAGI 生成器安装 WINA 包装；属于开发入口 |
| `ARCQuant/tests/test_magi2_sparse_runtime.py` | 目前主要检查专家矩阵方向和非法稀疏率，远未覆盖整模型正确性 |
| `external/base_eval/magi2_quant.py` | dense/grouped/routed HiF4 接线、量化边界 |
| `external/base_eval/magi2_batched_moe.py` | routed experts 批量执行与累加口径，替换时要保持语义 |
| `external/MAGI-2-preview/inference/model/` | 官方 preview/refiner 架构，适配以其真实 forward 为准 |

建议顺序：

1. **先核对 WINA 覆盖和语义。**弄清 preview/refiner 所有投影，区分 `[out,in]` 与专家 `[in,out]`，补齐 grouped/refiner；核对 router、adapter 等排除项，输出明确的覆盖清单。WINA 的目标是按 `abs(x) × 输入列范数` 选通；原论文的共享掩码和投影选择要逐项查证，不能因为写了 top-k 就声称完整原生实现。
2. **补好离线和在线边界。**权重范数原则上应从明确指定的原始权重获取并缓存；当前包装先 `apply_magi2_quant` 再安装 WINA，会使范数来源及计算开销需要重新核对。在线先对激活选通再量化，不能只对已量化激活随意追加置零。分组/专家布局、路由概率、激活函数、clamp、累加精度均保持原实现语义。
3. **修正统计再跑真实验证。**当前报告把统计 Counter 提前转成 dict，不能将安装时零计数当作运行统计。增加按模块/步数统计，记录目标稀疏度、实际零率、覆盖、NaN/Inf、显存峰值和时间。先测稀疏度 0 时与相同量化基线一致，再测 50% 的真实单视频，检查画面和数值，随后扩展其它档位。
4. **TEAL 需要视频专用离线校准。**采集真实去噪过程中各层/投影、不同 timestep 的激活，保留与原生阈值规则一致的统计和校准身份，再在线按阈值置零；不能直接拿 DS 的文本直方图套到 MAGI。校准提示词与评测提示词分开并记录，定义预算后再开长作业。
5. **方法完整性通过后才开放队列。**目前 `arc.py video-queue` 明确拒绝 MAGI WINA，且没有视频 TEAL 选项，这是防止漏层实现冒充完整方法的门槛。补齐代码、必要测试和真实单视频验证，再扩展这个入口，随后逐方法跑 50/60/70/80/90% 的完整 Tiny。

最先应交付：现有 HiF4 Tiny 的完整评分，以及至少一个经过覆盖/零率验证的 MAGI 稀疏单视频。然后逐步补齐同协议的完整方法表。真正 FP4 压缩存储和单卡 MAGI 是后续优化方向，不应成为当前双卡质量评测的前置条件。

## 8. 共享仓库如何开发，避免影响 DS

纯评测可以只读使用共享 `ARCQuant`。开发 MAGI 适配时使用独立 clone；不要在 DS 正在使用的目录中切分支、pull、改通用文件或运行安装命令。

```bash
MAGI_HOST=$(hostname -s)
MAGI_DEV=/mnt/kpfs_qyang/LBC/magi_work/$MAGI_HOST/ARCQuant
mkdir -p "/mnt/kpfs_qyang/LBC/magi_work/$MAGI_HOST"
git clone --no-hardlinks /mnt/kpfs_qyang/LBC/ARCQuant "$MAGI_DEV"
cd "$MAGI_DEV"
git remote set-url origin https://github.com/Calamansi39/ARCQuant.git
git switch -c "magi/$MAGI_HOST-adaptation"
export ARCQUANT_ASSET_ROOT=/mnt/kpfs_qyang/LBC/ARCQuant/assets
python arc.py fetch --backend base-eval \
  --source /mnt/kpfs_qyang/LBC/ARCQuant/external/base_eval
python arc.py fetch --runtime magi-2-preview \
  --source /mnt/kpfs_qyang/LBC/ARCQuant/external/MAGI-2-preview
```

这会创建完整、原名为 `ARCQuant` 的独立仓库，权重继续共享，无需重复下载；代码副本只占很小空间。目标目录存在时先查是否已有可用开发分支，不要覆盖。Git 若需要推送认证，使用目标机已授权方式，不更改服务器账号或复制私钥。

源码、文档和测试提交到对应分支；结果放独立目录。基线评测器与官方源码保持固定版本，优先通过仓库包装层适配。若必须修改固定实现，明确登记新版本并用新输出，保留旧证据。共享环境变量文件不随意覆盖；目标机专属配置存到它自己的开发目录或 `/mnt/kpfs_qyang/LBC/magi_runs/<hostname>/`。

## 9. 什么才算完成，如何汇报

完整 Tiny 结果至少满足：740 个视频的完成校验、16 维评分、闪烁候选筛选覆盖、`gen_summary.json` 与 `vbench_summary.json` 可追溯，模型/提示词哈希、50+5 步、精度和方法身份一致。方法结果还要有覆盖和实际置零统计。

展示表使用：`模型 | benchmark/协议 | 精度 | 方法 | 稀疏度 | 总分 | 质量 | 语义`。只把完整 Tiny 的 `tiny_scores` 填进去；官方全集使用 `official_scores` 并另外标注。不要把简单维度平均数当总分，不把视频中途计数或 smoke 分数当最终分数。

接手后请记录主机名、GPU UUID/型号/显存、环境、代码提交、队列 PID、输出目录、最新进度和失败原因。实际启动了什么就汇报什么；只完成准备时不要说已开跑。将进度追加到 `/mnt/kpfs_qyang/LBC/codexx1.md`，多人同时工作时先读最新内容，避免覆盖别人的记录。

## 可直接给另一台机器助手的指令

> 请读取 `/mnt/kpfs_qyang/LBC/MAGI_H20_HANDOFF.md`。你当前负责另一台 H20 上的 MAGI-2 工作。先核实共享挂载、硬件和环境，再识别共享盘中已有的 HiF4 Tiny 任务，避免重复生成或破坏缓存。在适当的独立代码目录内补齐 MAGI 的 WINA/TEAL 适配并完成真实验证；已有基线能接续时优先补齐它。沿用 HiF4 W4A4、1080p、10 秒、50+5 步和固定 Tiny manifest，不把 2:4 套进 TEAL/WINA。不要停止 DS 或他人的 GPU 进程，等待合格空卡运行。把可复现实现提交到你的分支，把实际结果和下一步写回 LBC 工作记录。
