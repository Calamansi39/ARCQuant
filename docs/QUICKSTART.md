# ARCQuant 整合版：从零开始

这份指南面向第一次接触仓库的人。仓库分开管理三件事：**方法代码**、**模型与数据的来源清单**、**本机下载的权重和评测结果**。GitHub 保存前两项；权重、视频、直方图、运行缓存通常太大，放在 `assets/`、`saved/`、`runs/`，不会被 Git 上传。

这条整合分支以 `joannahuadu/ARCQuant` 的量化实现为基准，保留 `Calamansi39/ARCQuant` 原有的预处理修正，再加入 TEAL/WINA 接线和统一 CLI；代码提交到 `Calamansi39/ARCQuant`。

**本阶段要评测的模型是 DeepSeek V4 Flash、MiniMax H3、MAGI-2。** 请先看 [目标模型进度](TARGET_MODELS.md)。下面的 Llama 章节解释已有方法原型，不能作为这三个目标模型已适配的证明。

## 先看能做什么

| 入口 | 当前可运行范围 | 说明 |
| --- | --- | --- |
| `arc.py --backend arcquant` | 原有 Llama / Qwen 2.5 / Mixtral 量化代码；TEAL/WINA 仅 Llama | `HiF4` 是 ARCQuant 的 W4A4 数值模拟；真实 4-bit 稀疏加速内核尚未接入 |
| `arc.py --backend base-eval` | DeepSeek V4 Flash 的 GPQA、MATH-500、LongBench v2、LiveCodeBench；MiniMax H3、MAGI-2 的 VBench | 调用固定版本的外部评测器；其 BF16/HiF4 是基线，**没有** ARCQuant 残差通道或 TEAL/WINA |
| `arc.py list` / `assets` / `fetch` | 所有登记的模型和数据 | 能查看、检查或下载；登记不代表有 ARCQuant 适配器 |

Terminal-Bench 已登记任务集和下载源，但还没有终端代理评测接线。视频模型的生成和 VBench 评分需要专门运行环境与高显存 GPU；只克隆本仓库不会自动安装它们。

## 1. 在新机器安装

```bash
git clone git@github.com:Calamansi39/ARCQuant.git
cd ARCQuant
conda create -n arcquant python=3.11 -y
conda activate arcquant

# 按新机器的 NVIDIA 驱动和 CUDA 版本安装 PyTorch；下面是 CUDA 12.8 示例
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python arc.py list
```

如果只查看 `list`、`assets`、`fetch --dry-run`、`plan`，系统 Python 即可。真正运行原 ARCQuant 量化路径要求 `transformers==4.46.3`；CLI 会在加载大权重前检查版本。`NVFP4` 的高性能 CUDA 路径还需要按仓库原有的 `kernels/README.md` 编译内核；`HiF4` 数值实验可以先走现有的 PyTorch 模拟路径。

## 2. 告诉仓库资产放在哪里

默认是仓库内但被 Git 忽略的 `assets/`。磁盘空间不够时，只需设置一个路径：

```bash
export ARCQUANT_ASSET_ROOT=/large-disk/arcquant-assets
```

目录约定为 `models/<模型名>`、`datasets/<数据集名>`、`calibration/<模型名>/<方法>`、`hf_home/`。`fetch` 将 Hugging Face 文件只下载到 `hf_home/hub` 一次，再在 `models/` 或 `datasets/` 建一个相对链接；移动整套资产目录后链接仍有效。单项已有资产可通过 `--model-path` 或 `configs/experiments.json` 记载的环境变量覆盖。`assets` 检查本机文件，不会下载，也不使用 GPU。

```bash
python arc.py fetch --model deepseek-v4-flash --dry-run
python arc.py fetch --model minimax-h3 --dry-run
python arc.py fetch --model magi-2-preview --dry-run
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark terminal-bench --benchmark vbench
```

实际下载时每次只对一个模型或数据运行 `fetch`，并确认磁盘空间与许可。已有权重可用清单记载的环境变量或 `--model-path` 指定。模型源、固定版本、文件检查规则在 [`configs/experiments.json`](../configs/experiments.json)。`main` 版本会在下载时记录实际 SHA；严格复现应固定该 SHA。

## 3. 目标模型与 benchmark

先下载被登记的资产：

```bash
python arc.py fetch --model deepseek-v4-flash --dry-run
python arc.py fetch --benchmark gpqa-diamond
python arc.py fetch --benchmark math500
python arc.py fetch --benchmark longbench-v2
python arc.py fetch --benchmark livecodebench
python arc.py fetch --benchmark terminal-bench
python arc.py fetch --benchmark vbench
python arc.py fetch --backend base-eval
```

当前登记的 `base_eval` GitHub 地址对这台机器的现有身份不可读取；要迁移外部生成评测器，需要先取得该仓库权限，或用 `python arc.py fetch --backend base-eval --source /已获授权的本地Git副本`。详情见 [资产与基线说明](ASSETS_AND_BASELINES.md)。

DeepSeek 的外部基线还需把官方权重转换为两卡版本；固定评测器提供 `convert_deepseek_v4.py`。MiniMax H3、MAGI-2 还需要各自的生成环境、VBench 评分依赖及 GPU。先用以下命令检查计划，再根据 [`资产与基线说明`](ASSETS_AND_BASELINES.md) 安装对应运行环境：

```bash
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark gpqa-diamond --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
python arc.py plan --backend base-eval --model minimax-h3 --benchmark vbench --profile tiny --method none --quant-type HiF4 --gpu-ids auto
```

DeepSeek 已有[投影级适配代码](DEEPSEEK_ADAPTER.md)，但尚无目标模型校准产物与整模型推理接线；视频模型仍未适配。因此不能把外部 `base-eval` 的 HiF4 结果称为 `ARCQuant + TEAL/WINA`。要在 DeepSeek 或视频模型上比较稀疏方法，还需完成校准、真实生成接线与 GPU 验证。

## 4. 已有的 Llama 方法原型

先准备 ARCQuant 自己的通道重排和量化缓存。它需要模型权重和 GPU，输出在 `saved/`：

```bash
python arc.py prepare --model llama-3.1-8b --dry-run
python arc.py prepare --model llama-3.1-8b
```

比较时保持模型、校准集、量化格式、benchmark 相同，只改 `--method`。`plan` 只显示将执行的命令；改为 `run` 才加载模型并生成 `runs/` 结果。

```bash
# ARCQuant 自身，不加激活稀疏
python arc.py plan --model llama-3.1-8b --method none --quant-type HiF4 --benchmark math500 --profile smoke

# WINA：每个 token 计算分数并保留前 50%；uniform 不需要离线 WINA 校准
python arc.py plan --model llama-3.1-8b --method wina --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke

# TEAL：先离线统计每层激活，再用统计结果运行
python arc.py calibrate --model llama-3.1-8b --method teal --allocation uniform --sparsity 0.5
python arc.py plan --model llama-3.1-8b --method teal --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke
```

若想让不同投影使用不同稀疏率，再离线生成 `greedy` 策略。它比 `uniform` 慢得多，因为会反复试验七个投影的层输出误差：

```bash
python arc.py calibrate --model llama-3.1-8b --method wina --allocation greedy --sparsity 0.5
python arc.py plan --model llama-3.1-8b --method wina --allocation greedy --sparsity 0.5 --quant-type HiF4 --benchmark gpqa-diamond --profile smoke
```

TEAL 也可把两条命令中的 `wina` 改为 `teal`。默认直方图和策略保存在 `assets/calibration/...`，`plan/run` 自动查找；自定义位置用 `--output-dir` 与 `--histogram-path` / `--strategy`。`--disable-prefill-sparsity` 可只对解码 token 启用稀疏。`smoke` 只是少量样本的接线检查，正式结果用 `--profile full`。实际运行会在 `runs/` 单独记录各投影观测到的零值比例；它包含本来就是零的激活，且不能当成加速比。

## 5. 两种方法到底做什么

量化把数字改成低精度表示；稀疏把某些数字变成零。在本仓库 Llama 路径中，顺序是 **投影输入 → TEAL/WINA 掩码 → ARCQuant 通道重排和 HiF4 量化 → 矩阵乘法**。

| 方法 | 离线（评测前做一次） | 在线（每次推理） |
| --- | --- | --- |
| TEAL | 用校准文本统计各层注意力和 MLP 的 `h1/h2` 激活直方图；可选 greedy，为七个投影分配不同稀疏率 | 从直方图读阈值，把幅度小于阈值的投影输入置零，再量化 |
| WINA | 均匀率只读取固定权重列范数；可选 greedy，离线试验各投影的稀疏率 | 每个 token 用 `|激活| × 权重列范数` 打分，保留 top-k，再量化 |

例如四个输入 `[1,2,3,4]`，目标保留两个：TEAL 若离线阈值为 `2.5`，保留 `[0,0,3,4]`。WINA 若第一个通道的权重列范数是 `10`、其余是 `1`，分数是 `[10,2,3,4]`，会保留 `[1,0,0,4]`。两种掩码都不强制 2:4；仓库原有的 `xmask` 才是另一条 2:4 路径。当前乘法仍是密集计算，零值不自动带来速度提升。

代码来源、与论文复现配置的区别和验证范围见 [`METHOD_PROVENANCE.md`](METHOD_PROVENANCE.md)。
