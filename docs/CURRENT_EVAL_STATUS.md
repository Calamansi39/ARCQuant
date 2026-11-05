# 当前模型、数据集和方法评测状态

更新时间：2026-11-05 21:03 CST。本次重新核查 DeepSeek 原始结果与运行状态；视频部分保留此前记录，未重新评分。

两种 baseline 的交付进度与队列说明见 [TEAL/WINA 实验交付表](BASELINE_PROGRESS.md)。
DeepSeek 已有 8 项完整生成式基线成绩；ARC＋TEAL/WINA 的完整生成式成绩仍未完成。当前 GPU 工作是 greedy 离线预算搜索，尚未进入全量答题。

新增普通 2:4 路径：`structured-plan` / `structured-run` 已接 DeepSeek 的 HiF4 W4A4
运行时和四个文本 benchmark 入口，四项 CPU dry-run 通过；真实权重的单卡投影测试
6/6 输出有限，55,296 个四元组结构检查无违规。当前仍无该路径的整模型正式分数，
share index 和硬件加速也未实现。详情见[结构化稀疏指南](STRUCTURED_SPARSITY.md)。
下表历史 TEAL/WINA 分数属于非结构化方法对照，不能表示 2:4 或层覆盖验收已完成。

展示以[已核查结果文档](RESULTS_PRESENTATION.md)及[彩色 Excel](completed_benchmarks.xlsx)为准。
2026-11-05 晚间重新核对 25 项历史完整文本记录，主表只展示 8 项符合约定协议的成绩（BF16/HiF4 × 四项 benchmark），数值与上版一致。
NVFP4/MXFP4、GPQA 标签似然、LiveCodeBench 128K 续写和视频 Tiny 留在审计快照。可下载 [TSV](completed_benchmarks.tsv) 或 [CSV](completed_benchmarks.csv)。

## 1. 模型和 benchmark 的对应关系

| 模型 | 主要 benchmark | 完整规模 | tiny/smoke 规模 | 当前实际状态 |
| --- | --- | ---: | ---: | --- |
| DeepSeek V4 Flash | GPQA Diamond | 198 题 | smoke 8 题 | BF16/HiF4 生成式基线已完成；ARC/TEAL/WINA 只有旧标签似然诊断，不能填生成式主表 |
| DeepSeek V4 Flash | MATH-500 | 500 题 | smoke 8 题 | BF16/HiF4 基线已完成；旧无 ARC WINA 停在 127/500；当前 greedy+ARC 尚在离线搜索 |
| DeepSeek V4 Flash | LongBench v2 | 503 条 | smoke 8 条 | BF16/HiF4 基线已完成；greedy+ARC 专用入口尚待补齐，无完整方法成绩 |
| DeepSeek V4 Flash | LiveCodeBench v6 | 175 题 | smoke 2 题 | BF16/HiF4 64K 基线已完成；greedy+ARC 专用入口尚待补齐，无完整方法成绩 |
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

### DeepSeek 的历史 ARCQuant 标签似然诊断（非生成式成绩）

`runs/deepseek_formal/` 中已经完成同一份 GPQA Diamond 198 题的五组结果：

| 变体 | 实际路径 | 数量 |
| --- | --- | ---: |
| BF16 | BF16 对照 | 198 |
| HiF4 | HiF4 W4A4 | 198 |
| ARC | HiF4 W4A4 + ARC 残差通道 | 198 |
| TEAL | HiF4 W4A4 + ARC + 50% TEAL 输入掩码 | 198 |
| WINA | HiF4 W4A4 + ARC + 50% WINA 输入掩码 | 198 |

这些是四个答案标签的条件似然诊断，并非主表要求的生成式 Pass@1。
虽然覆盖 198 题，也不能作为完整目标方法结果。该入口使用 HiF4 值重建和 dense PyTorch GEMM，能验证精度路径，不能直接
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

- **DeepSeek × GPQA/MATH：尚无完整生成式方法成绩。** greedy+ARC 入口已实现，仍需完成全模型策略与校准，再按固定协议答完全部题目。
- **DeepSeek × LongBench/LiveCodeBench：尚未满足。** 需补齐 greedy+ARC 专用入口，再做完整评测；基础 HiF4 或旧 uniform 入口不能代替。
- **MiniMax H3 × VBench：尚未满足。** 已有 HiFP4 W4A4 基线，但没有 ARC/TEAL/WINA。
- **MAGI-2 × VBench：尚未满足。** 目前没有完整的 FP4+方法结果；BF16 Tiny 已评分，
  HiFP4 W4A4 任务仍未形成完整评分表。

另外，DeepSeek 旧自定义入口的 GPQA 是答案标签条件似然协议，而 `base_eval` 的
DeepSeek GPQA 是生成式 Pass@1；两者题目数相同，但指标协议不同，不能直接把
百分数放在同一列比较。

## 5. 最新进度与剩余工作

- 旧无 ARC WINA MATH 已停止，保存 127/500 条，不能报告为全量准确率。
- 旧 uniform ARC 队列在校准保存阶段失败（BF16 索引越界）；主仓库已修复，但该旧队列的 11 个答题设置均未启动。
- 当前 `runs/native_greedy_campaign_20261105_185006/full/` 正在做正式 greedy 搜索：128×512 校准、20 段误差样本。21:03 快照为 WINA 3/43 层、TEAL 2/43 层，均覆盖 50%–90% 五档，策略文件 `complete=false`。
- 双机交接控制器仍在等 MAGI 当前批次释放 GPU，没有 `handoff_commit.json`，尚不能计作已双机并行。
- 当前搜索队列不会自动答题。后续要校验完整策略与校准，运行 ARC 无稀疏对照、两种方法五档的 MATH/GPQA，并补齐 LongBench/LiveCodeBench greedy+ARC 入口。

“评测中”是按用户约定填写的缺失成绩占位符，不代表每个配置都已在作答。

## 6. 主表稀疏度与配置

主表使用 TEAL/WINA **greedy、50%/60%/70%/80%/90%、HiF4 W4A4＋ARC 补偿**。
五档是按校准实际执行的逻辑线性 MAC 数加权的预算，各投影配额不同；不是实测全局激活零率，
也不是 2:4。WINA 未启用 transform。实现和原生算法差异见 [greedy 说明](DEEPSEEK_GREEDY.md)。

## 7. 如何刷新已有表格

在仓库根目录使用装有 `openpyxl` 的 Python，读取现有产物即可更新，无需 GPU：

```bash
python -m benchmarks.present_results --snapshot-only \
  --sweep-status runs/deepseek_arc_campaign_20261105/status.json \
  --greedy-search runs/native_greedy_campaign_20261105_185006/full
```

双机交接后用实际的新搜索目录替换 `--greedy-search`；该参数可重复。每个来源都保留路径与快照时间。
需要重新核查原始答题结果时，将 `--snapshot-only` 换成 `--base-eval /path/to/base_eval`；
新完成的 MATH 结果通过 `--math-summary /path/to/summary.json` 加入，只有完整题集、协议、
ARC 实际调用以及 greedy 策略和各 TP rank 证据全部通过才会填分数。
原始运行文件不随仓库上传，迁移时需另行复制；仅浏览现有成绩可直接使用已提交的快照和工作簿。
