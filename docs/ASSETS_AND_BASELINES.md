# 本机模型与评测数据：检查结果和使用方法

本页记录 2026-10-24 在这台服务器上核对的文件。模型权重和数据集放在仓库外，Git 只保存路径、检查规则和启动入口。`python arc.py assets` 不申请 GPU，会重新检查本机文件、权重分片索引及数据条数。

## 先看文件是否齐全

| 模型 | 本机目录 | 核对结果 |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 | `/mnt/kpfs_qyang/djs/models/DeepSeek-V4-Flash-0731` | 与 Hugging Face 镜像的 74 个文件逐项匹配大小；48 个权重分片齐全；两卡转换权重另在 `models/DeepSeek-V4-Flash-0731-mp2`，两片和转换记录齐全。 |
| MiniMax H3 | `/mnt/kpfs_qyang/zhaolianchao/models/MiniMax-H3` | **当前 VBench 文本生成视频（T2VA）所需文件齐全**：本机既有下载清单的 61 个文件大小匹配，文本编码器、transformer、VAE 的索引分片均在。官方完整仓库另有参考条件生成所用 `transformer_ref`、原始 `FL2VA/` 和 `Ref2VA/` 等 196 个未下载文件，约 354 GB；它们不参与当前 T2VA 评测。 |
| MAGI-2 preview | `/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview` | 与 Hugging Face 镜像的 89 个文件逐项匹配大小；preview、refiner 和文本编码器的索引分片、视频和音频 VAE 均在。 |

这里的“齐全”指文件存在、大小与已知清单一致，索引能找到全部分片，Safetensors 文件头有效。它不等于已逐字节计算数百 GB 权重的哈希，也不等于 ARCQuant 已支持该模型架构。

| Benchmark | 固定版本与本机数据 | 核对结果 |
| --- | --- | --- |
| GPQA Diamond | `Idavidrein/gpqa`，快照 `633f5ee` | 198 题 |
| MATH-500 | `HuggingFaceH4/MATH-500`，快照 `6e4ed1a` | 500 题 |
| LongBench v2 | `THUDM/LongBench-v2`，快照 `2b48e49` | 503 题，`data.json` 可解析 |
| LiveCodeBench v6 | `livecodebench/code_generation_lite`，快照 `0fe84c3` | `test6.jsonl` 为 175 题，文件大小与本机固定清单一致 |
| VBench | 本机 `data/vbench/VBench` | 官方元数据含 944 条唯一英文提示词；21 个评分模型及 BERT 文件的哈希已核对，合计约 9.43 GB |
| Terminal-Bench | 官方 `terminal-bench-core==0.1.1`，提交 `91e1045` | 已下载到 `data/terminal-bench-core-0.1.1/tasks`，86 个任务都含 `task.yaml` |

上表中的 `data/` 目录均位于 `/mnt/kpfs_qyang/zhaolianchao/data/`。完整路径、快照号和预期条数记录在 [`configs/experiments.json`](../configs/experiments.json)。Terminal-Bench 还有其他版本；本次固定的是 `0.1.1` 任务集。它的运行还需要终端代理、容器和模型服务，当前统一 CLI 只检查任务数据。

## 用统一 CLI 查看和启动已有基线

```bash
cd /mnt/kpfs_qyang/LBC
git status
python arc.py assets

# 只生成计划，不申请 GPU
python arc.py plan --backend base-eval --model deepseek-v4-flash \
  --benchmark gpqa-diamond --profile smoke --method none \
  --quant-type HiF4 --gpu-ids 0,1

python arc.py plan --backend base-eval --model minimax-h3 \
  --benchmark vbench --profile tiny --method none \
  --quant-type HiF4 --gpu-ids auto

python arc.py plan --backend base-eval --model magi-2-preview \
  --benchmark vbench --profile tiny --method none \
  --quant-type HiF4 --gpu-ids 0,1
```

把 `plan` 改为 `run` 才会启动实际实验。`smoke` 是少量文本题目；VBench `tiny` 是 108 条唯一提示词，按官方采样规则仍需每种精度生成 740 条视频。选择 `full` 或 `official` 前请看 [GPU 预算](GPU_BUDGET.md)。实验清单写到仓库的 `runs/`，DeepSeek 基线结果也写到 `runs/baseline_results/`；视频评分和视频缓存由现有评测器保存在其 `results/` 下。

`--backend base-eval` 调用本机已有、提交为 `c452d6d` 的 [`base_eval`](https://github.com/xiaodiandian-AI/base_eval) 评测实现。它已有 DeepSeek 的 GPQA、MATH-500、LongBench v2、LiveCodeBench v6 入口，以及 MiniMax H3、MAGI-2 的 VBench 入口。可用 `ARCQUANT_BASE_EVAL_ROOT` 或 `--base-eval-root` 指定其它副本。

**结果含义要分清：**这个后端的 `HiF4` 是已有基线的 W4A4 量化后反量化计算；它没有使用 ARCQuant 的增强残差通道，也没有接入 TEAL/WINA。`--backend arcquant` 才走本仓库原有的量化和 Llama 稀疏实现。新模型接入 ARCQuant 核心量化和稀疏层，仍需要分别实现它们的投影、校准与推理路径。

## 已做的 CPU 预检

DeepSeek 的 GPQA、LongBench v2、LiveCodeBench v6，MiniMax H3 的 VBench tiny，以及 MAGI-2 的 VBench tiny，均通过了已有评测器的 CPU 预检。它们检查模型与数据路径、提示词或任务协议、生成参数；没有加载完整权重或占用 GPU。本次没有新跑精度、视频生成或速度实验。
