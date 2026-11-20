# DS dense 优先评测与续跑

2026-11-20：本轮优先补 **HiF4 W4A4 + ARC、无激活稀疏**。普通 HiF4 的全量基线已完成，不重跑。

## 本批范围

先完成 GPQA Diamond 198 题、MATH-500 500 题。生成上限仍为 8192 token，batch size 20，seed 20261011，提示词、评分器、模型、ARC 校准和运行实现沿用已有 0% Tiny。

已核对并导入 GPQA 32 题、MATH 100 题，只需新增生成 166、400 题。原 Tiny 目录不改动；新目录的 `reused_prefix.json` 逐题记录原记录哈希和导入后的哈希。导入只改变协议选择范围对应的哈希，答案、评分、seed、生成耗时均保持原样。原生评测器随后重新检查记录校验和及评分，只调度未完成题目。

本轮是质量评测；导入答案的时间不能混算成这次全量运行的吞吐或加速比。LongBench、LiveCodeBench 的普通 HiF4 成绩已完成；它们现有扩展启动器尚未具备此 ARC 路径的运行证据，**没有把普通 HiF4 当成 ARC 重新排队**，后续先补 ARC 接入核验。

## 当前调度

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
