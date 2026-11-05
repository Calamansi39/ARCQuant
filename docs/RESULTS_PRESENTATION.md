# 实验结果展示：已核查的完整评测

核查时间：2026-11-04T05:55:17+00:00。本页由原始结果导出；本次整理没有新增模型评测。

[下载彩色 Excel](completed_benchmarks.xlsx?raw=true) · [CSV](completed_benchmarks.csv) · [TSV](completed_benchmarks.tsv) · [核查快照与证据哈希](results_audit.json)

**按目标格式筛选：完整文本成绩 15 项；视频 Tiny 参考 3 项。** 保留 BF16 参考和指定的 HiF4 W4A4。“完整”表示固定题集有完整输出与评分，不表示稀疏方法已达到最终目标。

NVFP4/MXFP4 来自服务器已有的历史结果；本次整理没有为它们新增推理任务。旧版把这些额外格式也放入了主表，现已移除。原始核查快照保留历史记录供追查，不计入本任务展示数量。

## 1. DeepSeek 文本基线

模型为 `DeepSeek-V4-Flash-0731`。只看 BF16 参考与指定的 HiF4 W4A4，后者在外部结果文件中命名为 HiFP4。下面尚未加入 TEAL/WINA；分数单位为 %，越高越好。

| Benchmark / 协议 | 题数 | BF16 参考 | HiF4 W4A4（目标） |
|---|---:|---:|---:|
| GPQA Diamond | 198 | 86.87 | 78.79 |
| MATH-500 | 500 | 98.60 | 98.60 |
| LongBench v2 | 503 | 50.30 | 47.51 |
| LiveCodeBench v6 64K | 175 | 84.57 | 73.14 |
| LiveCodeBench v6 128K 续写 | 175 | 94.29 | 93.14 |

- GPQA / MATH：0-shot thinking，8192 生成 tokens；分别用 `gpqa_final_letter_v4` / `math500_format_v3` 判分。
- LongBench：503 题，chat 模式，输入上限 131072 tokens，中间截断；输出上限 8192。
- LiveCodeBench：固定 release-v6 `test6` 分片 175 题，public + private 测试执行；不泛指所有版本题库。128K 为 64K 到限题续写，保留此前已结束的答案。
- BF16 是已发布混合 FP8/FP4 checkpoint 反量化后的计算参考；不是恢复出原始 BF16 训练权重。FP4 是指定投影的 W4A4 数值模拟/重建后 GEMM，不能据此声称硬件 FP4 或稀疏加速。
- 截断、答错、空代码及执行失败均留在完整分母中，未为提高分数删题。每项截断数见 Excel 明细。

## 2. GPQA 内部方法诊断（不是最终 benchmark 口径）

**这是四个选项标签的条件似然诊断，不能与上面的 GPQA 生成式 Pass@1 横向比较，也不应作为最终 GPQA 成绩。** 全部为 198 题。

实现是在 `Answer:` 后分别给 `(A)`、`(B)`、`(C)`、`(D)` 打分，选择概率最高的标签；它没有让模型先生成思考，也没有给四个完整选项文本分别评分。生成式基线允许模型先思考，再输出最终选项，所以分数更高并不矛盾。题目本身是同一批 198 题，差异来自答题协议和推理路径。

| 方法 | 正确题数 | 准确率 | 较同协议 BF16 |
|---|---:|---:|---:|
| BF16 | 99/198 | 50.00% | +0.00 pp |
| HiF4 W4A4 | 92/198 | 46.46% | -3.54 pp |
| HiF4 + ARC | 102/198 | 51.52% | +1.52 pp |
| HiF4 + ARC + TEAL 50% | 87/198 | 43.94% | -6.06 pp |
| HiF4 + ARC + WINA mask-only 50% | 81/198 | 40.91% | -9.09 pp |

这组结果来自记录内的历史代码快照（版本和实现哈希见核查文件），不能解释为后续修复代码已完成重跑。TEAL/WINA 的 50% 是非结构化输入掩码；WINA 为 mask-only，没有 optional transform；当前使用 dense GEMM。它们只能作为同一诊断协议内的相对对照，不是正式 GPQA 成绩，也不是 2:4 结构化稀疏或端到端加速成果。正式结果应把 TEAL/WINA 按生成式协议重新跑完，再与生成式 BF16/HiF4 比较。

## 3. 视频 Tiny 附录（不计入完整 benchmark 主表）

固定 Tiny manifest：108 个 prompts、计划 740 次生成、16 个维度。以下来自已保存的评分报告；本次未逐个验证全部视频文件。**没有完整 VBench 官方 suite 成绩。**

| 模型 | 配置 | Tiny 聚合总分 ×100 | 闪烁覆盖 | 状态 |
|---|---|---:|---:|---|
| MiniMax-H3 | bf16 | 81.25 | 46/50 | Tiny 覆盖不足；仅参考 |
| MiniMax-H3 | hifp4-w4a4 | 80.88 | 50/50 | 仅 Tiny 参考；非全量 |
| MAGI-2-preview | bf16-50p5 | 80.00 | 50/50 | 仅 Tiny 参考；非全量 |

总分采用报告中的 `tiny_scores.total × 100`。MAGI 旧表的 `0.6372` 是维度简单平均，已纠正为该报告的聚合总分；MiniMax BF16 闪烁维度不足，不用于完整同条件差值。

## 4. 离最终方法结果表还缺什么

| 模型 | Benchmark | 方法 | 状态 / 待补全 |
|---|---|---|---|
| DeepSeek | GPQA 生成式 | TEAL / WINA | 未完成：目前只有条件似然历史对照；需相同生成式协议的全量结果 |
| DeepSeek | MATH-500 | TEAL / WINA | 未完成：此前短测失败；修复后的任务已建队列，不能用 smoke 分数填表 |
| DeepSeek | LongBench v2 / LiveCodeBench | TEAL / WINA | 未完成：需完成整套生成、逐题评分和覆盖核查 |
| MiniMax-H3 | VBench | 全部 | 仅 Tiny 参考：BF16 闪烁覆盖 46/50；无全量官方 suite 和稀疏方法成绩 |
| MAGI-2-preview | VBench | 全部 | 仅 BF16 Tiny 参考：缺其他精度及 TEAL/WINA 的完整结果 |
| 适用模型 | Terminal-Bench | 全部 | 未完成：缺完整 agent / 环境 / 任务评分报告 |
| 目标模型 | 全部 | HiF4 + 2:4 | 未完成：当前结构与 smoke 检查不构成完整 benchmark 成绩 |

## 5. 核查与分享说明

- 文本：核对唯一题号、数量、逐题评分和总分；LCB 从已保存的执行结果重新汇总；条件似然检查有限数值、argmax 和对错标签。各精度间核对题集与协议。没有重新执行模型或判题器。
- 视频：核对 16 维报告与 summary、Tiny manifest 哈希和闪烁覆盖；仅作参考。
- 修正旧表：MATH-500 的来源改为 `math500_generative/...`；旧路径 `math500/...` 属于另一套 4-shot 协议，不能给 0-shot 分数作证。
- Excel 默认显示阅读说明和三张成绩页；四张核查/待补全附页默认隐藏，可在表格软件中取消隐藏。CSV/TSV 与主展示范围一致，只承载数据，没有颜色。
- 可直接分享本 GitHub 页面。需要多人编辑时，下载 XLSX，在腾讯文档、飞书或 Excel Online 中选择导入，再设置平台内分享权限。本次交付未创建这些平台的在线文档。
- 只发布汇总指标和相对来源路径/哈希；权重、题目正文、逐题生成内容及本机私人文件不包含在共享包内。

### 更新方法

```bash
pip install -r requirements-report.txt
# 有原始结果的机器：重新核查并导出
python benchmarks/present_results.py --base-eval /path/to/base_eval
# 其他机器：从仓库已核查快照重新导出，不需要模型/GPU/原始题目
python benchmarks/present_results.py --snapshot-only
```

核查快照锁定本次来源文件的 SHA256。原始结果变化后应重新核查；有缺失或分数不一致的条目会进入拒绝清单，不能自动填入主表。
