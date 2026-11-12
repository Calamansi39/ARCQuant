# DeepSeek 五种稀疏度 Tiny Bench 结果

截至：2026-11-12 10:40 CST

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | GPQA | MATH | LongBench | LiveCodeBench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 50% | HiF4 W4A4 + ARC补偿 | 46.9%（15/32）；截断14 | 94.0%（94/100）；截断6 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 50% | HiF4 W4A4 + ARC补偿 | 37.5%（12/32）；截断17 | 93.0%（93/100）；截断7 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 60% | HiF4 W4A4 + ARC补偿 | 9.4%（3/32）；截断28 | 71.0%（71/100）；截断28 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 60% | HiF4 W4A4 + ARC补偿 | 3.1%（1/32）；截断31 | 57.0%（57/100）；截断43 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 70% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32）；截断31 | 0.0%（0/100）；截断100 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 70% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32）；截断32 | 评测中（80/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 80% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32）；截断27 | 0.0%（0/100）；截断83 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 80% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32）；截断31 | 0.0%（0/100）；截断97 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 90% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32） | 0.0%（0/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 90% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32） | 0.0%（0/100） | 无 | 无 |

GPQA 为 Diamond 固定前 32 题，MATH 为 MATH-500 固定前 100 题。
全部任务使用 ARC HiF4 W4A4、8192 token 上限和生成式评分；Tiny 结果不代表全量 benchmark。
完整成绩经题目覆盖、协议、ARC 身份和两个 rank 的 greedy 策略证据核验。截断题保留在分母内并按评测规则计分。
LongBench 与 LiveCodeBench 不属于本轮 Tiny 横评，记为“无”。“评测中”后为已完成题数/计划题数；中途分数不作为准确率展示。


## 结果分析

当前 50%–90% 网格有 **19/20 个 benchmark 单元完成核验**，其中每个稀疏度和方法应有 GPQA、MATH 两个单元。未完成单元只保留题目进度，不把中途结果当作准确率。

待补单元：WINA（greedy） 70% MATH：评测中（80/100）。

- **50% 是这五档中当前优先保留的候选。** 两种方法在两个任务上的50% 成绩都不低于其余已完成档位；这不代表 50% 是所有可能稀疏度的最优点，也不代表相对无稀疏基线已经无损。
- **50% → 60% 的变化：** MATH 中，TEAL 从 94.0% 到 71.0%（下降 23.0 个百分点）；WINA 从 93.0% 到 57.0%（下降 36.0 个百分点）。
- **TEAL 与 WINA 的差异：** 答对题数差（TEAL − WINA）为 50% 时 GPQA +3 题、MATH +1 题；60% 时 GPQA +2 题、MATH +14 题。GPQA 只有 32 题，一题就是 3.125 个百分点；50% MATH 的一题差距也很小，不能仅据此宣称方法显著优越。
- **70%–90%：** 已完成 11/12 个单元，其中 11 个为 0 分。大量长度触顶与低分同时出现；仅凭这些结果不能断定增大 token 上限就能恢复能力，也不能把退化唯一归因于稀疏算法本身。
- **90% 的失效形式：** 两种方法共 264 次生成，长度截断 0 次，判为思考未闭合（`incomplete_thinking`）264 次。提前结束不等于成功作答，也不能据此证明推理加速有效。

### 后续优先事项

1. 补齐剩余单元，保留原始输出、当前预算和评分规则；40% 补充实验仍使用单独表格。
2. 补同题、同提示词、同生成预算的 ARC HiF4 W4A4 无稀疏（0%）对照，再比较候选档的精度损失。BF16 或不同协议的旧成绩不能替代这一对照。
3. 结合独立开发题确认生成预算和候选稀疏度，再固定配置完成 GPQA-Diamond 198 题、MATH-500 500 题。当前 Tiny 是测试题前缀，选点使用了这些题，正式报告须说明重叠，不能把它们称作独立验证。
4. 在无其他作业干扰的同等硬件条件下另测速度、显存和卡时；当前共享 GPU 的运行时间不能证明稀疏推理加速，当前 DeepSeek 路径仍主要是激活置零后的 dense GEMM 精度验证。

本表是当前 DeepSeek 适配版 TEAL/WINA greedy 的生成式 pass@1（单次回答准确率），不是选项似然的 acc_norm，也不代表完整原生 WINA transform 已实现。评分把长度触顶题保留在分母并判错；GPQA 另有 7168 token 思考预算及强制结束思考机制，需与 MATH 的生成过程区分。

数据来源：`runs/deepseek_tiny_sweep_20261106/<方法>-<稀疏度>-<任务>/arcquant-*/` 下的 `summary.json`、`outputs.jsonl`、`protocol.json` 和 greedy 双 rank 证据。

重新生成：`python tools/build_deepseek_tiny_results.py`（需要本轮 runs 数据）。
