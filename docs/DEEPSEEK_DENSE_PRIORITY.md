# DS dense 优先评测与续跑

2026-11-20：本轮优先补 **HiF4 W4A4 + ARC、无激活稀疏**。普通 HiF4 的全量基线已完成，不重跑。

## 最新状态：2026-11-25 10:23 CST

**目前两台 H20 均没有 DS ARC HiF4 全量评测进程，也没有这批任务的等待队列。** 三个 dense worker 均已正常结束，原计划只包含 GPQA 198 题和 MATH 500 题；两项完整结果已经核验。

- GPQA 此前安排在 H20-2 GPU 0、1，198/198 完成；MATH 此前安排在 H20-2 GPU 2、3，500/500 完成。这是历史运行位置，当前不占这些卡。
- ARC 版 LongBench、LiveCodeBench 64K / 128K 续写尚未启动，没有加入评测队列。现有扩展入口仍需完成 ARC 接入和协议核验，然后才能安排运行。
- 结果表按此前约定将无数据格统一写作“评测中”。这个标记只表示缺失结果，不能作为任务已启动或已排队的依据。普通 HiF4 的已完成成绩也不能用作 ARC 版成绩。

实际状态应结合本页的带日期记录、运行目录的 `plan.json` / worker 状态以及服务器进程判断。完整分数继续只维护在 [统一结果目录](../reports/verified/README.md)。

## 2026-11-24 00:49 CST 核查（历史记录）

两台 H20 的 16 张卡均有计算进程。H20-1 GPU 0–7、H20-2 GPU 0、1、4、5、6、7 为其他用户任务；我们的 MAGI 继续使用 H20-2 GPU 2、3，目前没有可直接启动 DS 的空闲双卡。

昨天恢复的 DS 30% 小探针已于 11 月 23 日 17:40 全部完成：TEAL/WINA 各 GPQA 8 题、MATH 16 题，合计 48 条答案，四项通过原有协议、策略、ARC 运行证据和题目覆盖校验。监督器正常结束，不能再把它写作等待空卡。该范围仍与 32/100 题主表不同。

MAGI HiF4 TEAL 已完成 57/70 模块，当前第 58 个模块 `refiner.block.layers.17` 保存到第 25 轮。控制器心跳正常，后续 HiF4 WINA 搜索、40% 配置提取/检查、视频生成和评分的依赖仍在队列中；当前没有新的稀疏 VBench 分数。生成前的磁盘检查阈值为 128 GiB，本次共享存储约余 169 GiB。

本次核查未发现空卡，继续现有 MAGI 作业；没有重复启动已完成的 DS 评测。

## 2026-11-23 10:11 CST 核查（历史记录）

GPQA 198/198、MATH 500/500 已全量完成，逐题覆盖、评分、两个 rank 的 ARC 运行证据和 132 道复用答案的哈希绑定均已再次核验。成绩只维护在 [统一结果目录](../reports/verified/README.md)。这两项当前没有 GPU 进程；LongBench/LiveCodeBench 的 ARC 路径仍待接入核验。

DS 30% 的 TEAL/WINA 搜索均已完成，43 层策略通过加载校验。TEAL GPQA 的 8 道答案也已完整保存。原监督器错误地在任务根目录读取 `summary.json`，实际评测器将它写在下一级 `arcquant-*` 身份目录，因此第一项完成后未能衔接下一项。现已改用已有的嵌套结果核验逻辑，并验证策略身份、题目覆盖及 ARC 证据后跳过完成项。

H20-2 的续跑监督器 PID 为 `1258370`，处于 `waiting_for_tp2_evaluation`：等待两张真正空闲的卡，随后补 TEAL MATH 16 题、WINA GPQA 8 题、WINA MATH 16 题。保留原来的模型、校准、策略、生成上限与答案，已完成的 8 题不重跑。该 8/16 题小探针不能直接混入 32/100 题主表。

目前我们只在 H20-2 GPU 2、3 上运行 MAGI；H20-1 的八张卡和 H20-2 其余六张卡均有其他作业。MAGI 的 BF16 TEAL/WINA 搜索各完成 70/70 模块；HiF4 TEAL 完成 48/70，正在第 49 个模块搜索；HiF4 WINA 尚未开始。当前搜索到 60% 的轨迹用于提取 40% 配置，后续四组 Tiny 视频生成和评分尚未开始。这是普通 HiF4 路径，没有 ARC 适配证据。

MAGI 当前使用 11 月 22 日部署的续跑控制器。每卡约 137 GiB 的驱动显存占用中，约 114 GiB 来自 `magi_memory_guard.py` 保留的 PyTorch 缓存，张量实际占用约 21 GiB；不能据此声称模型本身需要 137 GiB。两卡采样利用率为 100%，检查点从第 22 轮推进到第 24 轮。此前自动恢复控制器曾在 `capture-bf16-0` 失败，旧交接文件的 `magi_resumed` 只说明启动过，当前状态以上述新控制器和检查点为准。

## 本批范围

先完成 GPQA Diamond 198 题、MATH-500 500 题。生成上限仍为 8192 token，batch size 20，seed 20261011，提示词、评分器、模型、ARC 校准和运行实现沿用已有 0% Tiny。

已核对并导入 GPQA 32 题、MATH 100 题，只需新增生成 166、400 题。原 Tiny 目录不改动；新目录的 `reused_prefix.json` 逐题记录原记录哈希和导入后的哈希。导入只改变协议选择范围对应的哈希，答案、评分、seed、生成耗时均保持原样。原生评测器随后重新检查记录校验和及评分，只调度未完成题目。

本轮是质量评测；导入答案的时间不能混算成这次全量运行的吞吐或加速比。LongBench、LiveCodeBench 的普通 HiF4 成绩已完成；它们现有扩展启动器尚未具备此 ARC 路径的运行证据，**没有把普通 HiF4 当成 ARC 重新排队**，后续先补 ARC 接入核验。

## 11 月 20 日启动时的调度（历史记录）

- H20-2 GPU 0、1：GPQA dense；GPU 2、3：MATH dense。
- H20-2 GPU 4：原 DS 30% 搜索继续。
- MAGI 搜索在新检查点保存并核验后让卡。35 个完整模块和第 35/36 层的部分搜索进度保留；dense 两项完成后按原命令自动恢复。恢复会重放 dense 前缀以重建激活，但不会重新搜索已保存的候选步骤。
- H20-1 仍由其他任务占满，部署的 worker 只做 CPU 等待；共享任务锁防止两台机器重复评测同一任务。

交接只操作确认身份的 MAGI 控制进程及其搜索进程组，没有修改 MAGI 数值源码、检查点或其他用户进程。校验 greedy 历史必须使用原 MAGI Python 环境：Python 3.11 与 3.12 的浮点求和差异会导致逐位比较假报警。交接工具会自动切换到原控制器的解释器。

运行目录：`runs/deepseek_arc0_full_20261120/`。

- `plan.json`：固定命令和范围；`ready.json`：CPU 协议/复用核验通过。
- `worker-*.json`：心跳、实际 GPU、子进程和失败原因。
- `magi-handoff.json`：让卡前后检查点的路径、哈希、步骤数，以及恢复后的控制器身份。
- `none-0-*/arcquant-*/`：原生协议、逐题答案、摘要和每个 rank 的 ARC 运行证据。
- 结果只更新到 [统一目录](../reports/verified/README.md)，未完成项不显示中途准确率。

## 在其他机器上运行

先按资产指南准备模型、TP2 转换版、正式 ARC 校准、外部评测器以及原有 Tiny 输出。路径用目标机器自己的路径：

```bash
python tools/prepare_deepseek_dense.py \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration /path/to/formal/calibration \
  --tiny-root /path/to/deepseek_arc0_tiny_20261118 \
  --out-dir runs/ds_dense_full

# 用 tmux 或服务器调度系统持久运行。多机器用不同 worker 名称，共享 plan 和输出。
python tools/queue_deepseek_dense.py \
  --plan runs/ds_dense_full/plan.json --worker server-a --gpu-candidates 0,1,2,3

# CPU 报告器：只更新同一份报告。
python -m pip install -r requirements-report.txt
python tools/watch_verified_dense.py --run-dir runs/ds_dense_full --notes ../codexx1.md
```

无旧 Tiny 输出时省略 `--tiny-root`，从完整题库开始。迁移后旧输出的模型路径、包版本或代码身份不一致时，工具会拒绝导入；需要先核验差异，不能删掉身份字段强行复用。再次启动相同 plan 时，已完成题目自动跳过，结果目录不会清空。

当前 job 失败会保留全部输出并记录失败状态。MAGI 自动恢复以 dense 全部通过完整性检查为条件；若 dense 失败，先检查 worker 日志，按同一 plan 续跑，避免并发重启 MAGI 与 dense。
