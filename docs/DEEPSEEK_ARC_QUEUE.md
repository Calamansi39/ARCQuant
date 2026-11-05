# DeepSeek：ARCQuant 量化＋稀疏队列

2026-11-05 起，`run_deepseek_sparse_math.py`、`queue_sparse_baselines.py` 和
`queue_sparse_sweep.py` 默认启用 **HiF4 W4A4 + ARC 补偿**。
显式传 `--quantization hif4` 才是基础 HiF4 消融实验。

## 现在实际计算什么

一次投影依次执行：**TEAL/WINA 激活掩码 → 按校准顺序重排通道 → HiF4
权重和激活量化 → 对选定通道的激活量化误差再量化 → 拼接补偿通道并计算输出**。
补偿操作复用 `model/deepseek_experiment.py` 的 ARC 投影实现；数值测试直接与
仓库 `fake_reorder_quantize_x/w` 对照。普通注意力、不同长度请求的批量解码、
分组 `wo_a`、路由专家和共享专家都要经过对应路径。

ARC 是量化补偿，不是稀疏方法。`--sparse-method none --sparsity 0` 是 ARC
无稀疏对照；`teal` / `wina` 才增加激活稀疏。本页队列使用统一投影预算 uniform。
当前目标表使用另外的 [greedy 搜索与评测入口](DEEPSEEK_GREEDY.md)，
按投影分配预算；两条入口的结果不能混填。DeepSeek WINA 均未启用 optional transform。
当前计算仍是 HiF4 数值重构＋dense GEMM；这是精度评测，不代表硬件 FP4 或稀疏加速。

## 一条命令启动 uniform 对照安排

使用独立输出目录和一个尚不存在的新校准目录。先准备仓库记录的 `base_eval`、
模型原权重、TP2 转换权重及 WikiText-2 train。将下面路径替换为目标机器的路径：

```bash
python benchmarks/queue_sparse_sweep.py \
  --quantization arc --gpu-ids 0,1 \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --calibration /path/to/new-formal-calibration \
  --calibration-samples 128 --calibration-length 512 \
  --out-dir runs/deepseek_arc_campaign
```

顺序为：等两张空卡 → WikiText-2 正式校准 → ARC 无稀疏对照 → TEAL/WINA
各 50%、60%、70%、80%、90%。每个设置先跑 1 题、8 题验证，随后跑全部
500 题 MATH。校准统计只生成一次，在所有设置间共用；不使用评测题做校准。
本地 GPU 文件锁跨整个流程保持，不能阻止其他用户的非协作作业。

已有合格校准时，省略 `--calibration-data`，并将 `--calibration` 指向它。
全量 ARC 队列拒绝少于 128 段或每段少于 512 token 的校准。
旧 `calibration80` 实际为 **1×64 token**，只能用于接线验证。
`--no-dense-reference` 可跳过已经完成、且校准相同的 ARC 对照；
不要用旧无 ARC 的 WINA 50% 作业来跳过新 ARC 配置。

WikiText-2 固定版本：`Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`，
文件 `wikitext-2-raw-v1/train-00000-of-00001.parquet`，SHA256
`e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`。
本机有该文件缓存，无需重新下载。

## 如何确认确实执行了 ARC

每个结果目录记录：

- `protocol.json`：包含 ARC 开关、方法、稀疏率、校准与实现哈希；在保存和续跑校验前写入。
- `arcquant_sparse_runtime.json`：同一配置身份；输出目录名也含 `arc`/`hif4` 与配置哈希。
- `arcquant_runtime_rank0.json`、`rank1.json`：每张卡的实际 ARC 调用数、补偿调用数、
  注意力路径、专家覆盖、掩码统计及有限 logits 检查数。
- `arcquant_sources/`：实际使用的本仓库代码快照；`performance.json` 保存完整运行统计。

队列验证所有 TP rank 的执行证据。8 题阶段还要求走到不同请求长度的 ragged
解码分支；只有文字标记 ARC、缺少补偿调用、出现非有限 logits、或没有实际
稀疏掩码都会阻止该设置进入全量阶段。错误答案和达到生成长度上限仍计入结果，
不靠筛选容易题放行。

本页队列覆盖 MATH；同一入口支持 `--benchmark gpqa`。LongBench/LiveCodeBench
的旧 extended 入口仍是基础 HiF4＋稀疏，尚未改成这条 ARC 队列，不能混报。
结果主表已经为 ARC 设置预留单元格，旧无 ARC 的分数不能填进 ARC 行。

## 本机验证

2026-11-05，ARC 无稀疏、ARC＋WINA 50%、ARC＋TEAL 50% 均通过真实 TP2 GPU
验证：每组 2 条提示、2 步批量解码，所有 rank 检测到 ARC 残差和两种注意力路径，
logits 有限。36 项定向测试通过。精简原始证据见
[验证记录](DEEPSEEK_ARC_VALIDATION.json)。这些验证不计作准确率成绩。

## 部署核查（2026-11-05 21:00 CST）

历史 uniform 队列位于 `runs/deepseek_arc_campaign_20261105/`，原 PID `2608124`，
目标 GPU `0,1`。它使用该目录 `source/` 的固定提交 `7410808` 工作副本；
后续主仓库修复不会更新这个旧副本。任务目录的 `launch.json` 保存完整命令，
`status.json` 是实时状态，`launcher.log` 为队列日志。

该队列已于 18:51 报 `blocked / calibration_stage=failed`：128 次校准前向后，
保存直方图时发生 BF16 索引越界，未生成可用的正式校准。其 11 个评测设置均未启动，
不能再将它报告为“校准中，之后自动答题”。主仓库中的索引问题已修复。

当前工作转为 `runs/native_greedy_campaign_20261105_185006/` 中的正式 greedy 搜索，
并已部署[双机交接](DEEPSEEK_MULTI_HOST.md)。搜索队列只做 pilot 和完整策略搜索，
不会自动启动 benchmark。完整策略与校准产物通过校验后，还须启动 MATH/GPQA 全量答题，
并补齐 LongBench/LiveCodeBench 的 greedy+ARC 专用入口。最新快照见结果工作簿的“说明与来源”。

旧的无 ARC sweep 已停止；旧 WINA 50% 的 127/500 条 MATH 输出保留在
`runs/baseline_queue_20261104_130054/`，不续接到 ARC 输出目录。
