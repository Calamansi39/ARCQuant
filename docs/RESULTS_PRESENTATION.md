# DeepSeek 量化与稀疏评测结果

模型：DeepSeek-V4-Flash-0731。分数单位为 %。进度快照：2026-11-05T09:03:13+00:00

| Benchmark | 评测口径 | 题数 | 目标稀疏度 (%) | BF16 (0%) | HiF4 W4A4 (0%) | ARC (0%) | TEAL (%) | WINA (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPQA Diamond | 生成式 Pass@1 | 198 | 50% | 86.87 | 78.79 | 未完成 | 未完成 | 未完成 |
| MATH-500 | 生成式 Pass@1 | 500 | 50% | 98.60 | 98.60 | 未完成 | 等待 GPU | 已写入 121/500 |
| MATH-500 | 生成式 Pass@1 | 500 | 60% | 98.60 | 98.60 | 未完成 | 等待 GPU | 等待 GPU |
| MATH-500 | 生成式 Pass@1 | 500 | 70% | 98.60 | 98.60 | 未完成 | 等待 GPU | 等待 GPU |
| MATH-500 | 生成式 Pass@1 | 500 | 80% | 98.60 | 98.60 | 未完成 | 等待 GPU | 等待 GPU |
| MATH-500 | 生成式 Pass@1 | 500 | 90% | 98.60 | 98.60 | 未完成 | 等待 GPU | 等待 GPU |
| LongBench v2 | 生成式 Accuracy | 503 | 50% | 50.30 | 47.51 | 未完成 | 未完成 | 未完成 |
| LiveCodeBench v6 64K | 生成式 Pass@1 | 175 | 50% | 84.57 | 73.14 | 未完成 | 未完成 | 未完成 |
| LiveCodeBench v6 128K 续写 | 生成式 Pass@1 | 175 | 50% | 94.29 | 93.14 | 未完成 | 未完成 | 未完成 |
| GPQA Diamond | 标签似然诊断 Accuracy | 198 | 50% | 50.00 | 46.46 | 51.52 | 43.94 | 40.91 |

- 分数单位为 %。目标稀疏度只针对 TEAL/WINA；BF16、HiF4、ARC 对照均为 0%。重复的 MATH 基线复用同一份结果。
- TEAL/WINA 均在 HiF4 W4A4 下评测。GPQA 标签似然诊断的 TEAL/WINA 另含 ARC，WINA 为 mask-only；不能与生成式 GPQA 混比。
- 50%–90% 是非结构化激活置零率目标，不是 2:4 或层覆盖率。未完成项目显示进度，不用部分题目计算最终分数。
- LiveCodeBench 为固定 test6 分片 175 题；128K 是 64K 到限题续写。生成式基线为本项目固定协议，不是官方榜单复现。
- 蓝色＝完整生成式基线；紫色＝完整标签似然诊断；黄色＝待完成。其他 benchmark 暂列 50% 槽位，60%–90% 当前只登记 MATH 网格。

[彩色 Excel](completed_benchmarks.xlsx) · [CSV](completed_benchmarks.csv) · [TSV](completed_benchmarks.tsv) · [审计快照](results_audit.json)

从仓库快照重新导出：`python benchmarks/present_results.py --snapshot-only`。
有新进度时加 `--math-summary <完整的 summary.json 路径>` 和/或 `--sweep-status <队列 status.json 路径>`；两项都可重复指定。
