# 当前模型、数据集和方法评测状态

更新时间：2026-11-04。

两种 baseline 的交付进度与队列说明见 [TEAL/WINA 实验交付表](BASELINE_PROGRESS.md)。
DeepSeek 生成接线已修正，CPU 回归检查通过；整模型恢复情况仍待 GPU 验证。

新增普通 2:4 路径：`structured-plan` / `structured-run` 已接 DeepSeek 的 HiF4 W4A4
运行时和四个文本 benchmark 入口，四项 CPU dry-run 通过；真实权重的单卡投影测试
6/6 输出有限，55,296 个四元组结构检查无违规。当前仍无该路径的整模型正式分数，
share index 和硬件加速也未实现。详情见[结构化稀疏指南](STRUCTURED_SPARSITY.md)。
下表历史 TEAL/WINA 分数属于非结构化方法对照，不能表示 2:4 或层覆盖验收已完成。

展示以[已核查结果文档](RESULTS_PRESENTATION.md)及[彩色 Excel](completed_benchmarks.xlsx)为准。
2026-11-04 重新核查后，完整文本主表为 25 项，5 项视频 Tiny 放入参考附页；
旧 30 行混合表不再作为完整 benchmark 清单。可下载 [TSV](completed_benchmarks.tsv) 或 [CSV](completed_benchmarks.csv)。

## 1. 模型和 benchmark 的对应关系

| 模型 | 主要 benchmark | 完整规模 | tiny/smoke 规模 | 当前实际状态 |
| --- | --- | ---: | ---: | --- |
| DeepSeek V4 Flash | GPQA Diamond | 198 题 | smoke 8 题 | 独立基线已完成；ARCQuant 的 HiF4、ARC、TEAL、WINA 已完成 GPQA 全量 198 题 |
| DeepSeek V4 Flash | MATH-500 | 500 题 | smoke 8 题 | 独立 BF16/FP4 基线已完成；WINA 50% 单题和 20 题试跑均退化并截断，TEAL 没有有效结果 |
| DeepSeek V4 Flash | LongBench v2 | 503 条 | smoke 8 条 | 独立 BF16/FP4 基线已完成；旧 WINA 单题仅有运行记录，生成质量待修正后重新验证，无正式方法结果 |
| DeepSeek V4 Flash | LiveCodeBench v6 | 175 题 | smoke 2 题 | 独立 BF16/FP4 基线已完成；WINA 生成入口已接入，需 GPU smoke 后才能形成正式结果 |
| DeepSeek V4 Flash | Terminal-Bench core 0.1.1 | 86 个任务 | smoke 1 个，small 5 个 | 数据已检查，代理、终端服务和容器评分尚未接入 |
| MiniMax H3 | VBench | 944 个官方 prompt | tiny 108 个唯一 prompt，740 条视频/精度 | Tiny 的 BF16、NVFP4、MXFP4、HiFP4 独立基线已评分；DiT WINA 启动器已接入，尚无正式稀疏成绩 |
| MAGI-2 preview | VBench | 944 个官方 prompt | tiny 108 个唯一 prompt，740 条视频/精度 | BF16 Tiny 已评分；HiFP4 W4A4 生成任务有记录；WINA 启动器已通过 dry-run，分组 refiner 仍待校准 |

Terminal-Bench 按任务性质应归入 DeepSeek 这一类文本/代码模型评测，但它目前是
额外登记的 benchmark，不在 DeepSeek 已完成的四项主表里。

## 2. VBench tiny 到底有多少条

`/mnt/kpfs_qyang/zhaolianchao/base_eval/vbench_tiny_manifest.json` 固定了本次
tiny 子集：

- 16 个 VBench 维度，每个维度选择 10 个长度分层 prompt；
- 由于不同维度共享 prompt，最终是 **108 个唯一 prompt**，不是 160 个独立问题；
- 普通维度每个 prompt 采样 5 条视频；
- `temporal_flickering` 的 10 个 prompt 先生成每个 25 条候选，再按 VBench 的
  静态筛选和选择规则保留；
- 因而每种模型、每种精度的 tiny 任务量是 **740 条视频**。

所以 VBench 的“108”是 prompt 数，“740”才是一次精度评测需要生成和评分的视频数。
它不是语言模型那种 108 道题。

## 3. 我们已经实际跑了什么

### DeepSeek 的 ARCQuant 方法结果

`runs/deepseek_formal/` 中已经完成同一份 GPQA Diamond 198 题的五组结果：

| 变体 | 实际路径 | 数量 |
| --- | --- | ---: |
| BF16 | BF16 对照 | 198 |
| HiF4 | HiF4 W4A4 | 198 |
| ARC | HiF4 W4A4 + ARC 残差通道 | 198 |
| TEAL | HiF4 W4A4 + ARC + 50% TEAL 输入掩码 | 198 |
| WINA | HiF4 W4A4 + ARC + 50% WINA 输入掩码 | 198 |

这是当前唯一已经完成“FP4 数值路径 + ARC/TEAL/WINA 方法”的整模型 benchmark
对照。该入口使用 HiF4 值重建和 dense PyTorch GEMM，能验证精度路径，不能直接
当作真实稀疏 kernel 的速度结果。Llama 路径已加入 WINA 论文中的可选全模型
`transform`；DeepSeek 不使用该 Llama 坐标变换。

### 服务器已有的独立基线

`/mnt/kpfs_qyang/zhaolianchao/base_eval/results/` 中已有：

- DeepSeek 的 BF16、NVFP4 W4A4、MXFP4 W4A4、HiFP4 W4A4，覆盖 GPQA、MATH-500、
  LongBench v2 和 LiveCodeBench；
- MiniMax H3 的 VBench tiny 四种精度基线；
- MAGI-2 的 BF16 Tiny 评分结果，以及一条尚未完成评分的 HiFP4 W4A4 生成任务。

这些 `base_eval` 结果没有 ARC 残差通道，也没有 TEAL/WINA 激活稀疏。仓库的
`base-eval` 后端明确只允许 `--method none`，因此不能把它们当作方法结果。

## 4. 是否已经满足“FP4 + 对应方法”

结论分模型看：

- **DeepSeek × GPQA：已满足。** 已跑 `HiF4 W4A4 + ARC`、`+ TEAL`、`+ WINA`，并且
  使用同一校准目录和同一 198 题数据。
- **DeepSeek × MATH：入口已接通但尚未形成正式结果；LongBench/LiveCodeBench：WINA/TEAL
  入口已接通但尚未形成正式结果。** 目前只有短上下文、单题调试结果，不能加入正式
  成绩表；完整任务仍需检查协议、覆盖率、续跑和显存峰值。
- **MiniMax H3 × VBench：尚未满足。** 已有 HiFP4 W4A4 基线，但没有 ARC/TEAL/WINA。
- **MAGI-2 × VBench：尚未满足。** 目前没有完整的 FP4+方法结果；BF16 Tiny 已评分，
  HiFP4 W4A4 任务仍未形成完整评分表。

另外，DeepSeek 自定义入口的 GPQA 是四选项条件似然协议，而 `base_eval` 的
DeepSeek GPQA 是生成式 Pass@1；两者题目数相同，但指标协议不同，不能直接把
百分数放在同一列比较。

## 5. 最新 MATH-500 WINA 诊断

WINA 50% 的单题对照生成 512 tokens 后达到长度上限，输出出现重复符号和乱码；同一题在 WINA 0% 下生成 80 tokens 并正确结束。随后在两张 H20 上启动的全量任务只完成首批 20/500 题，20 题全部达到 8192 tokens 上限，正确数为 0，已停止并从正式统计中排除。这个结果说明基础 W4A4 生成链路可用，但当前 DeepSeek WINA 掩码接线不能直接用于正式评测。

下一步应先按 attention、FFN/MoE、grouped `wo_a` 分组做 10%/25%/50% 掩码诊断，重新核对离线校准和逐投影重要性，再用 1 题和 8 题 smoke 验证 EOS、有限 logits、答案格式和速度。修复前不应恢复全量 WINA，也不应把这 20 题写入结果表。

## 6. 稀疏率要求核查

当前正式表里的 TEAL/WINA 都是 50% 置零率。原生 WINA 的复现实验脚本还包含
65% 这一标准点，greedy 默认有效目标约为 66%；原生 TEAL 的公开主结果集中在
40%–50%。因此 50% 不是 WINA 唯一可能的最终点，但仓库中没有证据证明 TEAL 和
WINA 必须统一到 65%。详细证据、定义和后续矩阵见[稀疏率要求核查](SPARSITY_REQUIREMENT_AUDIT.md)。
