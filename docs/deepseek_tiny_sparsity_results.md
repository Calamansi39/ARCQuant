# DeepSeek 五种稀疏度 Tiny Bench 结果

截至：2026-11-08 18:41 CST

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | GPQA | MATH | LongBench | LiveCodeBench |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 50% | HiF4 W4A4 + ARC补偿 | 46.9%（15/32）；截断14 | 94.0%（94/100）；截断6 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 50% | HiF4 W4A4 + ARC补偿 | 37.5%（12/32）；截断17 | 93.0%（93/100）；截断7 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 60% | HiF4 W4A4 + ARC补偿 | 评测中（21/32） | 71.0%（71/100）；截断28 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 60% | HiF4 W4A4 + ARC补偿 | 3.1%（1/32）；截断31 | 57.0%（57/100）；截断43 | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 70% | HiF4 W4A4 + ARC补偿 | 评测中（0/32） | 评测中（20/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 70% | HiF4 W4A4 + ARC补偿 | 0.0%（0/32）；截断32 | 评测中（20/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 80% | HiF4 W4A4 + ARC补偿 | 评测中（排队） | 评测中（0/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 80% | HiF4 W4A4 + ARC补偿 | 评测中（0/32） | 评测中（41/100） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 90% | HiF4 W4A4 + ARC补偿 | 评测中（排队） | 评测中（排队） | 无 | 无 |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 90% | HiF4 W4A4 + ARC补偿 | 评测中（排队） | 评测中（排队） | 无 | 无 |

GPQA 为 Diamond 固定前 32 题，MATH 为 MATH-500 固定前 100 题。
全部任务使用 ARC HiF4 W4A4、8192 token 上限和生成式评分；Tiny 结果不代表全量 benchmark。
完整成绩经题目覆盖、协议、ARC 身份和两个 rank 的 greedy 策略证据核验。截断题保留在分母内并按评测规则计分。
LongBench 与 LiveCodeBench 不属于本轮 Tiny 横评，记为“无”。“评测中”后为已完成题数/计划题数；中途分数不作为准确率展示。
