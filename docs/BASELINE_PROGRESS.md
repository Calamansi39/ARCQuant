# TEAL / WINA 实验交付表

> **2026-11-05 晚间更新：** 下文保留 11 月 4 日的交付计划与历史诊断。当前 DeepSeek 已转入 greedy+ARC 离线搜索，尚无完整生成式方法成绩；旧 uniform ARC 队列校准保存失败。最新已核查状态见 [当前评测状态](CURRENT_EVAL_STATUS.md)，主表见 [结果表](RESULTS_PRESENTATION.md)。

目标：把 TEAL、WINA 接到指定模型的 HiF4 W4A4 推理路径，在相同评测协议下补齐成绩。
有空卡就运行；没有空卡就进入后台等待。适配问题先修好再扩大评测。

## 当前缺哪些结果

截至 2026-11-04，下表记录的是完成情况；没有正式成绩的位置保持待完成。

| 模型 | benchmark | 规模 | TEAL | WINA |
| --- | --- | ---: | --- | --- |
| DeepSeek V4 Flash | GPQA Diamond | 198 题 | 已有条件似然成绩 43.94% | 已有条件似然成绩 40.91% |
| DeepSeek V4 Flash | MATH-500 | 500 题 | 接线已修正，待 GPU 验证 | 接线已修正，待 GPU 验证 |
| DeepSeek V4 Flash | LongBench v2 | 503 条 | 待验证和全量评测 | 待验证和全量评测 |
| DeepSeek V4 Flash | LiveCodeBench v6 | 175 题 | 待验证和全量评测 | 待验证和全量评测 |
| DeepSeek V4 Flash | Terminal-Bench | 86 个任务 | 待接入代理和评分 | 待接入代理和评分 |
| MiniMax H3 | VBench Tiny | 740 条视频 | 待完成模型适配 | 已有入口，待生成和评分验证 |
| MAGI-2 preview | VBench Tiny | 740 条视频 | 待完成模型适配 | 已有部分入口，待补齐 refiner 并验证 |

GPQA 这两项已有成绩来自 `HiF4 + ARC + 方法` 的条件似然协议；生成式 GPQA
尚无这两种方法的正式成绩。不能与另一套生成式基线直接比较。
其他已有的 BF16/FP4 成绩见[彩色 Excel](completed_benchmarks.xlsx)。

## 本次修正与等待任务

修正 DeepSeek 的三处接线问题：各层使用自己的 grouped 输出参数；不在加载权重前
计算权重重要性；TEAL 专家投影使用自己的前缀规则。grouped 输出改为与原生 WINA
一致的单位权重重要性，并统一先稀疏、后量化。旧的退化结果保留作诊断。
这些是已验证的代码修正，整模型精度是否恢复需要实际生成确认。

新增 `benchmarks/queue_sparse_baselines.py`：

1. 自动等待两张空卡，沿用本仓库的 GPU 文件锁，与其他合作队列协调。
2. TEAL 和 WINA 分别运行 1 题，再运行 8 题；两种方法独立推进。
3. 指定 `--full` 后，通过小样本检查的方法自动运行 MATH-500 的 500 题。
4. 小样本缺题、重复题号或非截断回答缺少可提取的最终答案时暂停该方法。长度截断
   按错误保留，不会因单个截断阻止全量运行；不会按准确率挑选或删除结果。
   一个方法失败不会取消另一个方法。
5. `status.json` 记录等待、运行、完成和待修复状态；逐步日志可定位失败原因。

在其他机器上准备好[迁移指南](MIGRATION_EVALUATION.md)中的环境和资产后，用评测环境的
Python 启动（替换路径）：

```bash
python benchmarks/queue_sparse_baselines.py \
  --base-eval-root /data/base_eval \
  --source-model /data/models/deepseek \
  --converted-model /data/models/deepseek-mp2 \
  --calibration /data/calibration/deepseek \
  --out-dir runs/baseline_queue_run1 --full
```

加 `--dry-run` 可做无 GPU 预检。用 `setsid` 或服务器作业系统启动即可在断开终端后
继续等待。每次使用新的输出目录；修复前后代码指纹不同，不能混用旧结果续跑。
队列负责实验，完成后的正式表仍需核对评测协议再更新。

需要在同两张卡上连续运行 50/60/70/80/90% 网格时，使用
[连续双卡实验队列](CONTINUOUS_GPU_QUEUE.md)。它在整批实验期间保留本地协作锁，
但不是调度器资源预约，也不能阻止其他用户使用 GPU。

## 汇报约定

先报告四项：**已完成、正在跑、排队中、待解决**。代码细节保存在仓库和工作记录中。
“有接口”“小样本通过”“全量已评分”是三个不同阶段；只有最后一种能填写正式成绩。
