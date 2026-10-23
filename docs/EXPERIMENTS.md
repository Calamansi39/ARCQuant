# ARCQuant 实验指南

## 先理解仓库在做什么

ARCQuant 原本负责**量化**：把模型的权重（W）和中间激活（A）按较低精度处理。本次在学姐维护的 `joannahuadu/ARCQuant` 主线代码上加入了另一种可选步骤：**激活稀疏**，即在计算前把一部分激活置零。执行顺序是：

```text
模型权重 → TEAL / WINA 稀疏激活 → ARCQuant 重排与量化 → 评测
```

`--quant-type HiF4` 是本仓库的 W4A4 数值模拟路径。`--method` 选择稀疏规则，二者是独立的选项。当前量化实现仍可能在内存中持有普通浮点张量；稀疏值为零也不保证底层矩阵乘法已经跳过这些计算。**精度、真实显存占用与推理速度需要分开测量。**

本次采用学姐 `main` 为代码基准，同时合并你的 `Calamansi39/ARCQuant` 主线提交 `e4ab5a2`，因此你的仓库可以保留原有提交历史后再接收整合分支。学姐加入的 `x_mask`、softmax alpha 和联合校准代码保留在主流程；你原有的 `model/bridge.py`、预处理兼容修正和快捷脚本也保留。对于 `model/eval.py`、`model/quantize.py` 的冲突，使用学姐当前实现，避免旧版量化接口覆盖新接口。实验入口在根目录的 `arc.py`，登记表在 `configs/experiments.json`，Llama 稀疏实现位于 `model/sparse_utils.py` 和 `model/qLlamaLayer.py`。

## 已接入的方法

| `--method` | 做什么 | 额外输入 | 当前范围 |
| --- | --- | --- | --- |
| `none` | 只运行 ARCQuant 量化 | 无 | 原有 Llama / Qwen / Mixtral 适配代码 |
| `teal` | 按校准直方图得到阈值，低于阈值的激活置零 | 每层注意力和 MLP 的 `histograms.pt` | Llama |
| `wina` | 每个 token 按分数取 top-k 激活，其余置零 | 无；分数使用原权重输入列范数 | Llama |
| `xmask` | 调用学姐主线已有的 2:4 掩码路径 | 已校准的 `x_mask` checkpoint | 统一 CLI 中的 Llama |

TEAL 的直方图来自之前的 `feature/arc-teal` 实现，文件目录形如 `layer-0/self_attn/histograms.pt` 与 `layer-0/mlp/histograms.pt`。WINA 接入的是**均匀稀疏率**版本：K 投影和 MLP gate 投影用真实权重输入列的 L2 范数，其他投影用 1；默认在预填充阶段只稀疏最后约 99% 的 token，解码阶段逐 token 稀疏。尚未接入原 WINA 的逐矩阵 greedy 稀疏率分配。参考代码为 [Calamansi39/wina](https://github.com/Calamansi39/wina) 的 `04ac8794aa4ae345d09414733ceb76470d3b6a3e`，上游为 [microsoft/wina](https://github.com/microsoft/wina)。

学姐主线已有另一种 `x_mask` 2:4 激活掩码。`--method xmask --x-mask-ckpt /path/to/checkpoint.pt` 会转发给原有 `model/main.py --use_x_mask` 路径，可选 `--x-mask-r-thr`、`--rec` 和 `--softmax-alpha-ckpt`。当前不允许在同一次运行中把 TEAL/WINA 和 `x_mask` 叠加，因为两种掩码会改变同一量化输入，尚未完成联合校准。

## 第一次运行

在仓库根目录使用已安装依赖的环境。模型权重不提交到 Git；通过 `--model-path` 或登记表中的环境变量提供路径。

```bash
python arc.py list
export ARCQUANT_LLAMA_MODEL=/path/to/Llama-3.1-8B

# 准备重排索引、选择通道数与激活统计，写入 saved/
python arc.py prepare --model llama-3.1-8b --dry-run
python arc.py prepare --model llama-3.1-8b
```

`prepare --dry-run` 只打印命令。真正的 `prepare` 会调用原有 `reorder_indices.py` 并使用 GPU。默认校准集是 WikiText-2，128 个样本，每个 2048 token，激活排序指标是 `max`。同一模型换了校准集或排序指标，需要重新生成对应缓存。

先用 `plan` 核对将执行的命令和缓存路径：

```bash
python arc.py plan --model llama-3.1-8b --method none \
  --quant-type HiF4 --benchmark wikitext2

python arc.py plan --model llama-3.1-8b --method teal --sparsity 0.5 \
  --histogram-path /path/to/teal/histograms \
  --quant-type HiF4 --benchmark gpqa-diamond --profile smoke

python arc.py plan --model llama-3.1-8b --method wina --sparsity 0.5 \
  --quant-type HiF4 --benchmark math500 --profile smoke

python arc.py plan --model llama-3.1-8b --method xmask \
  --x-mask-ckpt /path/to/x_mask.pt --quant-type HiF4 --benchmark wikitext2
```

把 `plan` 换成 `run` 才会加载权重并评测。`run` 在开始前检查权重目录、ARCQuant 缓存和 TEAL 各层直方图文件；记录文件写入 `runs/`。lm-eval 评测结果也写到该目录。`saved/` 和 `runs/` 均不提交到 Git。

`--sparsity 0.5` 是目标稀疏率；WINA 的逐 token top-k 比例由它直接决定。TEAL 的实际零值比例受校准直方图与输入分布影响。加 `--disable-prefill-sparsity` 可以仅对解码 token 启用稀疏。

## 模型和评测状态

“已登记”只表示名称和路径已经进入配置，**不表示模型能够被 ARCQuant 加载**。

| 模型 | 当前状态 |
| --- | --- |
| Llama 3.1 8B | 原有量化适配器；TEAL/WINA 已接入该路径 |
| Qwen 2.5 7B、Mixtral 8x7B | 保留原有量化适配代码；未接入 TEAL/WINA，需按实际依赖版本验证 |
| DeepSeek V4 Flash | 已登记本机权重；尚无该架构的 ARCQuant 量化适配器 |
| MiniMax H3、MAGI-2 preview | 已登记本机权重；视频模型接口与量化适配器尚未接入 |

| Benchmark | 当前状态 |
| --- | --- |
| WikiText-2、现有零样本任务、MMLU | 通过原有 perplexity / lm-eval 入口 |
| GPQA Diamond、MATH-500 | 已配置当前 lm-eval 中的 `gpqa_diamond_zeroshot`、`hendrycks_math500` 任务；提供每任务 8 题的 `smoke` 档 |
| LongBench v2、LiveCodeBench、Terminal-Bench、VBench | 已登记；尚未接好各自的输入、生成和评分流程 |

仓库已有 `model/eval_longbench.py`，它针对原 LongBench 任务，**不等于 LongBench v2 适配器**。VBench 的一次提示可能对应多个生成视频；配置中的 `tiny`/`official` 只是规划档位，不代表已经具备正式评分能力。

当前新模型和生成式 benchmark 的路径可在 `configs/experiments.json` 修改，也可用环境变量覆盖。新增模型时先实现模型结构和 ARCQuant 投影适配，再验证量化输出；新增 benchmark 时按官方协议接入生成和评分，最后再把状态改为可运行。

## 已验证到哪一步

已验证 CLI 参数与当前 `model/main.py` 对齐、HiF4 下 TEAL/WINA 的计划生成及稀疏函数的 CPU 数值行为。仓库的 `requirements.txt` 锁定 `transformers==4.46.3`；检查时本机 `arcquant` 环境是 5.9.0，旧 Llama 包装层在这个版本下无法构造。`prepare`/`run` 会先检查版本并提示更换环境。这里还缺少可直接回归的 Llama 权重、校准缓存与 TEAL 直方图，故**尚未完成真实模型精度和速度回归**。在组会或论文中使用结果前，应分别跑 `none`、`teal`、`wina` 的相同模型与 benchmark，并另外测量端到端延迟。
