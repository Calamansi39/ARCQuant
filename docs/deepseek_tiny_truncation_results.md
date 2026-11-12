# DeepSeek Tiny GPQA/MATH 非截断准确率与截断率

截至：2026-11-12 11:40 CST

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | tinyGPQA 非截断准确率 | tinyGPQA 截断率 | tinyMATH 非截断准确率 | tinyMATH 截断率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 50% | ARC HiF4 W4A4 | 15/18（83.3%） | 14/32（43.8%） | 94/94（100.0%） | 6/100（6.0%） |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 50% | ARC HiF4 W4A4 | 12/15（80.0%） | 17/32（53.1%） | 93/93（100.0%） | 7/100（7.0%） |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 60% | ARC HiF4 W4A4 | 3/4（75.0%） | 28/32（87.5%） | 71/72（98.6%） | 28/100（28.0%） |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 60% | ARC HiF4 W4A4 | 1/1（100.0%） | 31/32（96.9%） | 57/57（100.0%） | 43/100（43.0%） |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 70% | ARC HiF4 W4A4 | 0/1（0.0%） | 31/32（96.9%） | —（无自然EOS） | 100/100（100.0%） |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 70% | ARC HiF4 W4A4 | —（无自然EOS） | 32/32（100.0%） | 未完成（80/100） | 未完成（80/100） |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 80% | ARC HiF4 W4A4 | 0/5（0.0%） | 27/32（84.4%） | 0/17（0.0%） | 83/100（83.0%） |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 80% | ARC HiF4 W4A4 | 0/1（0.0%） | 31/32（96.9%） | 0/3（0.0%） | 97/100（97.0%） |
| DeepSeek-V4-Flash-0731 | TEAL（greedy） | 90% | ARC HiF4 W4A4 | 0/32（0.0%） | 0/32（0.0%） | 0/100（0.0%） | 0/100（0.0%） |
| DeepSeek-V4-Flash-0731 | WINA（greedy） | 90% | ARC HiF4 W4A4 | 0/32（0.0%） | 0/32（0.0%） | 0/100（0.0%） | 0/100（0.0%） |

口径：非截断准确率只统计 `finish_reason=eos` 的题；截断率为 `finish_reason=length` 的题数/计划题数。自然 EOS 但未形成有效答案仍计入非截断准确率分母，并按评测器的正确性判定计分。`未完成` 行不是正式成绩。
GPQA 为固定前 32 题，MATH 为固定前 100 题；两者均为 Tiny 筛查结果，不等于完整 benchmark。
数据使用 ARC HiF4 W4A4、TEAL/WINA greedy 配置和 8192 token 生成上限。
