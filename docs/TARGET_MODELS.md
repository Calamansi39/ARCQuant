# 三个目标模型：现在到底做到哪一步

更新于 2026-10-27。这里的“文件齐全”只表示清单中的文件、分片和数据条数通过检查；“已有基线结果”指这台服务器原有的 `base_eval` 实验。两者都不表示 ARCQuant 稀疏方法已经在目标模型上跑通。

| 目标模型 | 本机权重 | 对应评测数据 | 已有独立基线 | ARCQuant HiF4 + TEAL/WINA |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash | 原版 48 分片、两卡转换版均检查通过 | GPQA Diamond 198、MATH-500 500、LongBench v2 503、LiveCodeBench v6 175 题均检查通过 | BF16、NVFP4、MXFP4、HiFP4 的完整结果保存在服务器原有 `base_eval/results` | **投影级代码和 CPU 检查通过；未接入整模型，未运行评测** |
| MiniMax H3 | 文生视频所需组件检查通过；参考视频组件不在本次范围 | VBench 的 944 条唯一英文提示词检查通过 | 原有 `base_eval` 的 VBench Tiny 四精度结果 | **未接入，未运行** |
| MAGI-2 preview | preview、refiner、文本编码器、VAE 文件检查通过 | 同上 | 原有两卡、短步数生成记录；没有可称为完整 VBench 的分数 | **未接入，未运行** |

Terminal-Bench core 0.1.1 的 86 个任务文件也检查通过，但模型服务、终端代理和容器评分还没有接线。VBench 的评分网络权重与 Python 环境另需按 [资产指南](ASSETS_AND_BASELINES.md) 检查；提示词齐全不代表评分环境可运行。

DeepSeek 的投影级实现、官方 FP8/FP4 权重切片检查和后续接线步骤见 [DeepSeek 适配进度](DEEPSEEK_ADAPTER.md)。统一 CLI 仍会拒绝在 DeepSeek 上启动未完成的 ARCQuant 整模型评测。

## 已有结果是什么

服务器原有报告位于 `/mnt/kpfs_qyang/zhaolianchao/base_eval/results/REPORT.md`，逐题或逐视频结果位于其 `results/` 下。这些数字出自**独立的 BF16/FP4 基线**，不含 ARCQuant 增强残差通道，也不含 TEAL/WINA；本仓库没有重新跑这些大实验。摘取与当前任务最直接相关的 HiFP4 数据：

| 模型 / 任务 | 原有 BF16 | 原有 HiFP4 |
| --- | ---: | ---: |
| DeepSeek / GPQA Diamond | 86.87% | 78.79% |
| DeepSeek / MATH-500 | 98.60% | 98.60% |
| DeepSeek / LongBench v2 | 50.30% | 47.51% |
| DeepSeek / LiveCodeBench v6，64K | 84.57% | 73.14% |
| MiniMax H3 / VBench Tiny 总分 | 81.25 | 80.88 |

这些协议的生成长度、采样与评分规则见原报告。LiveCodeBench 的 128K 是基于 64K 截断回答的续写补充实验，不能当作从头生成的原生 128K。MiniMax 的 Tiny 是固定提示词子集，不等于 VBench 官方全集。

## 在这台服务器怎样检查

我已把现有权重、两卡转换权重和数据集以**仅在本机有效的软链接**接到 Git 忽略的 `assets/`，并将固定提交的外部评测器和 MAGI-2 官方运行代码检出到 Git 忽略的 `external/`。GitHub 只保存来源、版本、检查规则和启动入口；不会上传数百 GB 权重或现有基线结果。换服务器时按 [快速指南](QUICKSTART.md) 的 `fetch` 或把资产目录整体迁走，再重新检查。

本机 CPU 预检：DeepSeek 的 GPQA 8 题生成计划及两卡转换清单通过，MiniMax H3 的 VBench Tiny 108 提示词/740 视频计划与生成、评分环境导入通过；MAGI-2 的模型头、采样计划和 21 个 VBench/BERT 资产哈希通过。MAGI-2 的完整启动预检仍会申请两卡并受显存占用影响，本次没有完成该环节。以上整模型检查均未产生目标模型的 ARCQuant + TEAL/WINA 评测输出。本次另用 DeepSeek 官方权重切片做了投影级 CPU 数值检查，不能当作模型精度或加速结果。

```bash
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark terminal-bench --benchmark vbench

# 只生成独立基线的启动计划，不占 GPU
python arc.py plan --backend base-eval --model deepseek-v4-flash \
  --benchmark gpqa-diamond --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
```

如果在目标模型上尝试 `--backend arcquant --method teal` 或 `wina`，CLI 会明确拒绝；它没有偷偷退回 Llama 或外部基线。`base-eval` 也会拒绝这两种稀疏方法。

## 为什么现有 Llama 接口不能直接套用

DeepSeek 使用 43 层、低秩 Q 与 O 投影、压缩注意力、超连接和 MoE；其官方路由专家权重是打包 FP4。TEAL 需要在各个实际投影的输入处采集直方图并确定阈值；WINA 若使用列范数，需要按真实权重解包和按张量并行切片计算。尤其 `wo_a` 通过分组 `einsum` 直接读取权重，普通线性层钩子覆盖不到。要保留 ARCQuant 的增强残差通道，还必须针对这些投影重新做通道选择、权重量化和校准，不能把独立基线的 HiFP4 改个名字。

MiniMax H3 是 50 层 Diffusers 视频 Transformer；MAGI-2 还使用自己的分组线性层和 MoE。两者的扩散步、文本条件和视频评分协议与自回归语言模型不同。因此它们需要各自的投影映射、校准数据、推理入口和完整步数生成验证。当前仓库没有这两套适配器。

当前 8 张 H20 在本次检查时各只剩约 8–27 GiB 空闲显存，无法据此宣称目标模型已在本仓库完成 GPU 冒烟测试。排期与卡时估计见 [GPU 预算](GPU_BUDGET.md)。
