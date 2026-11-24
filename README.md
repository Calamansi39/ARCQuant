# ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs

## 从这里开始

| 你想看什么 | 入口 |
| --- | --- |
| **实验结果、汇报用数据** | **[彩色 Excel](reports/verified/实验结果.xlsx?raw=true)** · [结果范围与缺项](reports/verified/README.md) |
| **代码怎么用、方法怎么配置** | **[中文快速开始](docs/QUICKSTART.md)** |
| **服务器目录和文件在哪里** | **[工作区导航](docs/WORKSPACE_GUIDE.md)** |
| **DS 当前评测与续跑安排** | [调度记录](docs/DEEPSEEK_DENSE_PRIORITY.md) |

**DeepSeek ARC uniform 对照队列：** [启动、校准与历史部署状态](docs/DEEPSEEK_ARC_QUEUE.md)；当前 greedy 目标使用下面的独立入口。

**DeepSeek TEAL/WINA greedy 预算：** [原生搜索规则、离线准备、推理加载与迁移命令](docs/DEEPSEEK_GREEDY.md)。
统一入口 `python arc.py deepseek-greedy --help`；先验证两种方法，再搜索 50%–90% 五档。
完整 43 层策略以产物的 `complete` 标记为准；WINA transform 尚未启用。

**DS dense 优先续跑：** [GPQA/MATH 全量、旧答案复用与 MAGI 检查点交接](docs/DEEPSEEK_DENSE_PRIORITY.md)。

**后续卡时与组会计划：** [首批申请、完整网格预算及实测依据](docs/GPU_BUDGET.md)（2026-11-05 更新）。

**预算受限时的实验收缩方案：** [先选稀疏度，再做正式评测](docs/EXPERIMENT_RESCOPING.md)。

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ARCQuant-2601.07475-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2601.07475)
 <br>

</h5>

![arcquant](kernels/ARCQuant_Visualize.png)

**ARCQuant** is a high-performance quantization framework designed to resolve the conflict between accuracy and inference efficiency in low-bit LLMs.

While fine-grained quantization (e.g., Block-wise/NVFP4) effectively isolates quantization noise, activation outliers still degrade performance in critical channels. Traditional mixed-precision methods address this by splitting computations into separate branches, which introduces significant kernel launch overhead and memory fragmentation.

ARCQuant takes a different approach. Instead of treating outliers separately, we leverage the structural sparsity of quantization errors in fine-grained settings. We capture the quantization residuals of these critical channels and fuse them back into the computation as **Augmented Residual Channels (ARC)**.

To do: 

- [x] Release arxiv version of [ARCQuant](https://arxiv.org/abs/2601.07475).
- [x] Release code for reproducing results.
- [x] Release CUDA kernels on NVFP4.
- [x] Release calibration and preprocessing scripts.
- [x] Support [vLLM](https://github.com/vllm-project/vllm) integration.
- [ ] **Model Support**: Add support for more model families:
    - [ ] Qwen3
    - [x] Mixtral
    - [ ] Wan2.2


## 1. Installation
```bash
conda create -n arcquant python=3.11 -y
conda activate arcquant
```
For the original native NVFP4 kernels, use the matching [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) toolchain. For the HiF4 PyTorch evaluation path, install a PyTorch build compatible with your GPU driver; see the [Chinese quickstart](docs/QUICKSTART.md).
```bash
git clone --recurse-submodules https://github.com/Calamansi39/ARCQuant.git
cd ARCQuant
# Install the PyTorch build matching your CUDA driver first; see docs/QUICKSTART.md.
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Building Kernels
```bash
sudo apt-get update
sudo apt-get install python3-dev
```
```bash
conda install pybind11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
cd kernels/
bash remake.sh
```

This might take a few minutes.

### 2.2 Preprocessing

Reorder_indices, select_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 128 --seqlen 2048 --act_sort_metric max
```
Results are saved in ./saved/

### 2.3 Accuracy Evaluation

ARCQuant supports multiple formats, including NVFP4, MXFP4, HiF4, and INT4. You can modify the `quant_type` parameter if needed.
```bash
bash evaluate.sh /PATH/TO/YOUR/MODEL/
```

## 3. Efficiency Evaluation

FlashInfer:
```bash
cd third-party/flashinfer
python -m pip install -v .
```
We will release our vLLM evaluation very soon.

## 中文快速指南：方法整合与迁移

**三台服务器分工部署：** [部署与占卡评测指南](docs/MULTI_SERVER_DEPLOYMENT.md)
说明如何让当前机器保留 DeepSeek、另一台 H20 跑 MAGI、A800 跑 MiniMax；
新增 `arc.py video-queue` 自动完成环境预检、等空卡、单视频验证、VBench 生成和评分。
两台机器共用挂载存储时，直接看 [MAGI H20 交接](docs/MAGI_H20_HANDOFF.md)：
现有资产与结果位置、避免重复生成、运行要求和方法适配任务均已列出。
向 A800 传包部署 MiniMax 时，使用 [MiniMax A800 交接](docs/MINIMAX_A800_HANDOFF.md)，
含完整 Git 仓库恢复、模型下载、环境安装与 HiF4 稀疏评测要求。

**普通 2:4 结构化路径：** 新增 `arc.py structured-plan` / `structured-run`，将
HiF4 W4A4 与每组四取二的激活掩码组合，并单独统计线性投影覆盖率。它与原生
TEAL/WINA 元素稀疏率扫描是不同的实验。当前完成了 CPU 测试及真实权重的单卡
投影验证；整模型质量、share index 和硬件加速仍待验证。使用方法与边界见
[结构化稀疏指南](docs/STRUCTURED_SPARSITY.md)。

**当前实验目标是 DeepSeek V4 Flash、MiniMax H3 和 MAGI-2。** 换机器先按 [模型评测迁移指南](docs/MIGRATION_EVALUATION.md) 准备文件和环境，再用 `arc.py doctor` 查看逐项状态。[统一结果目录](reports/verified/README.md) 保存已核查的 DS Tiny、DS 全量 dense 和 MAGI Tiny，一份彩色 Excel 对应三页；计数与来源见 [审计快照](reports/verified/audit.json)。[目标模型进度](docs/TARGET_MODELS.md) 区分模型、数据和方法适配；[稀疏率要求核查](docs/SPARSITY_REQUIREMENT_AUDIT.md) 说明目标定义。DeepSeek 的实现见 [适配说明](docs/DEEPSEEK_ADAPTER.md)，固定来源见 [实验清单](configs/experiments.json)。

外行读法：`provision` 准备模型权重、题库、评分器和源码；`doctor` 逐项检查能否启动；`plan` 展示将执行的命令；`run` 才真正占用 GPU。大文件在 Git 忽略的 `assets/`，本机专用环境路径在 `external/local_env.sh`，新机器可按 [`configs/runtime.example.sh`](configs/runtime.example.sh) 填写。TEAL/WINA 是先离线确定稀疏规则、再在线稀疏投影输入；DeepSeek 已有 ARC/稀疏的 GPQA/MATH Tiny 结果和各 benchmark 的生成式方法入口；视频模型已有独立 VBench 基线，ARC 稀疏结果仍待验证。

```bash
python arc.py list
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark vbench
python arc.py provision --model deepseek-v4-flash --benchmark gpqa-diamond
python arc.py doctor --model deepseek-v4-flash --benchmark gpqa-diamond --gpu-ids 0,1 --native-dry-run
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark gpqa-diamond \
  --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
```

DeepSeek 的生成式稀疏入口（LongBench-v2、LiveCodeBench、MATH-500）直接沿用
`base_eval` 的原生 tokenizer、thinking、生成和评分协议。先用少量题目检查：

```bash
torchrun --nproc_per_node=2 benchmarks/run_deepseek_sparse_extended.py \
  --base-eval-root /path/to/base_eval --sparse-method wina --sparsity 0.5 \
  --benchmark longbench-v2 --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 --quant hifp4 --mode w4a4 \
  --limit 1 --max-new-tokens 4 --out-dir validation/longbench-wina

torchrun --nproc_per_node=2 benchmarks/run_deepseek_sparse_math.py \
  --quantization arc --calibration /path/to/formal-calibration \
  --base-eval-root /path/to/base_eval --sparse-method wina --sparsity 0.5 \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 --quant hifp4 --mode w4a4 \
  --limit 8 --out-dir validation/math-wina
```

要一次生成原生 TEAL/WINA 的 50%、60%、70%、80% 和 90% 元素稀疏率对照命令，可用
`sparse-plan`；这些点不保证 2:4，也不表示稀疏层覆盖率。`sparse-plan` 只写
计划，不占用 GPU；使用 `--profile baseline_sweep` 时会输出这五档的 WINA/TEAL
smoke 命令：

```bash
python arc.py sparse-plan --benchmark math500 \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration runs/deepseek_formal/calibration \
  --limit 1 --max-new-tokens 128 --gpu-ids 0,1
```

稀疏运行时会把目标稀疏率、可掩码元素数、实际置零元素数和实际零值比例写入
`runtime`。目标值和真实观测值会同时保留，不会把 prefill 中未处理的前缀或模型
原有零值误报成精确的目标百分比。

TEAL 还要加入与 TP rank 对应的 `--calibration runs/.../rank*.pt`。输出中的
`runtime.sparse_masked_elements` 是真实掩码计数；smoke 结果只用于检查链路，不能
填入正式 benchmark 表。MiniMax H3 也提供可迁移的 DiT WINA 启动器：
`benchmarks/run_minimax_sparse.py`，它不会改动文本编码器和 VAE。MAGI-2 也提供
`benchmarks/run_magi2_sparse.py`，当前覆盖 dense 与 routed preview 层；分组 refiner
层仍会在报告中明确列为待校准，不会把 DeepSeek 的统计套过去。

想先看到真实模型里的激活分布，不必等待完整 benchmark，可以用两张卡跑一
个短 prompt 的诊断探针。它只保存统计量，不保存完整激活张量：

```bash
python arc.py probe --model deepseek-v4-flash \
  --base-eval-root /path/to/base_eval \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --gpu-ids 0,1 --output runs/deepseek_activation_probe.json \
  --quant hifp4 --max-length 64
```

在确认统计正常后，可直接让探针用同一批激活估计并应用一次稀疏掩码，查看
logits 变化：加上 `--method teal` 或 `--method wina --sparsity 0.5`。

探针会报告各投影的激活范围、零值比例，以及 TEAL/WINA 在几个目标稀疏率
下的实际保留率和输入能量损失。它用于校准和 debug，不能替代正式精度结果；
完整的实测记录见 [DeepSeek smoke run](docs/DEEPSEEK_SMOKE.md)。

`assets` 检查文件，`plan` 生成启动计划；两者不运行 GPU。DS 的 ARC 0% 与 TEAL/WINA 40%–90% 已完成 GPQA/MATH Tiny，原有 BF16/普通 HiF4 已完成四项全量基线；ARC 全量结果仍需补齐。MAGI 已有 BF16/普通 HiF4 的完整 Tiny，但未包含 ARC 补偿；MiniMax 的本机证据尚不完整。可复现入口见 [DeepSeek 实验](docs/DEEPSEEK_EXPERIMENT.md)，最新成绩见 [统一结果](reports/verified/README.md)，方法来源见 [实现核查](docs/METHOD_PROVENANCE.md)，迁移依赖见 [资产指南](docs/ASSETS_AND_BASELINES.md)。

## 4. Citation
```
@article{meng2026arcquant,
  title={ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs},
  author={Meng, Haoqian and Luo, Yilun and Zhao, Yafei and Liu, Wenyuan and Zhang, Peng and Ma, Xindian},
  journal={arXiv preprint arXiv:2601.07475},
  year={2026}
}
```

## 5. Acknowledagement
Our code is built on the following repos, thank you for your contributions to community:
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

