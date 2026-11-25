# DS ARC + HiF4：LongBench / LiveCodeBench 接续队列

> **2026-11-25 本轮更新：** DS 40% 全量已经提到最高优先级，旧 no-ARC MAGI 队列已在检查点暂停并退役。当前调度看 [评测优先级](EVALUATION_PRIORITY.md)，MAGI 新 ARC 代码和验证边界看 [MAGI ARC](MAGI_ARC.md)。下文为此前实现与运行记录，不再作为当前派发顺序。

## 这次安排是什么

2026-11-25 已在 **H20-2 的 GPU 2、3** 部署常驻交接监督器。当前 MAGI 的 HiF4 WINA 搜索继续计算，完整完成 70 个模块后，按下表顺序使用同一对卡。DS 为 **无激活稀疏、0%、HiF4 W4A4 + ARC** 的全量对照。

| 顺序 | 工作 | 范围 |
| --- | --- | --- |
| 当前 | MAGI HiF4 WINA 搜索 | 保留并完成现有搜索，之后生成视频暂缓 |
| 1 | DS LongBench v2 | 完整 503 题；输入 131072、输出 8192 token；chat |
| 2 | DS LiveCodeBench v6 test6 | 完整 175 题；每题 1 次；输出 65536 token；thinking/max；batch 16 |
| 3 | DS LiveCodeBench 128K 续写 | 已自然结束的答案保留；只续写 64K 到限题，最多再生成 65536 token；batch 8 |
| 最后 | 恢复 MAGI 原控制器 | 提取/检查 40% 配置，接着生成视频、评分 |

GPQA 198 题和 MATH 500 题的 ARC dense 全量结果已经完成，不在这批队列里重跑。128K 是 **64K 答案的续写**，不能描述为一次独立的 128K 采样；原采样 RNG 状态不存在，续写采用原基线的固定新 seed `2026101901`。

## 去哪里看

运行目录：`runs/deepseek_arc0_extended_20261125/`。

- `plan.json`：完整命令、顺序、固定源码哈希、原搜索进程身份。
- `ready.json`、`*-preflight.log`：CPU 检查记录。GPU 实跑验证仍待当前搜索结束。
- `handoff.json`：实时心跳、交接阶段、已完成搜索模块数、DS 子任务和失败原因。部署时监督器 PID `1653532`，原搜索 PID `879569`。
- `longbench/`、`lcb64/`、`lcb128/`：分开的逐题答案、生成配置、评分和 ARC 执行证据。
- `memory_guard/`：新 DS worker 的实际张量、可复用缓存、空闲显存。
- 最终成绩自动回写 [同一份 Excel / CSV / TSV](../reports/verified/README.md)。完整题目覆盖、评分和两个 rank 的 ARC 证据通过后才发布。

`waiting_for_current_search_natural_completion` 表示正在等待 **现有搜索完整结束**，不是 DS 已开始生成答案。MAGI 原 `status.json` 的控制器心跳此时不会更新，因为控制器只暂停了派发；判断搜索是否正常，应看新 `handoff.json`、搜索日志和检查点。

## 如何保留卡

1. 只向核对过 PID、启动时间和命令的 **MAGI 控制器** 发送 STOP；它的 torchrun 和两个计算进程继续运行，不中断搜索。
2. 原 MAGI 的 GPU 锁一直由控制器持有，再增加 ARC 队列锁，避免我们自己的其他队列挑走这两张卡。
3. 当前 MAGI 保留原有可复用 CUDA 缓存。DS 也在进程内保留 PyTorch 空闲缓存，留约 4 GiB 给 CUDA/NCCL 等库；这部分缓存不等于模型张量需求，可以供本进程后续分配使用。
4. 一项 DS 结束后暂不退出进程，保留模型与缓存；CPU 评分、续写准备期间继续保留。下一项完成导入和身份检查后，通知前一项释放，再加载自己的模型。
5. 搜索完整性检查或 DS 运行失败时保留结果，记录失败并恢复 MAGI 控制器。独立看护进程防止监督器意外退出后控制器一直暂停；若 DS 子任务尚在计算，看护会等它结束再恢复。

这是尽量减少空档的措施。**普通文件锁和缓存不能阻止不遵守锁的外部程序或管理员抢用 GPU。** MAGI → DS 的第一次交接、模型换载和最后恢复 MAGI 仍有 CUDA 上下文交接窗口；需要绝对排他时须由服务器调度系统或管理员提供资源预留。

## 代码实际做了什么

- 新入口 `benchmarks/run_deepseek_arc_extended.py` 接入原来的 LongBench 长输入修复、LCB 连续补位批处理，以及自然续写流程。
- 复用已有 ARC 投影实现和正式 TP2 校准：通道重排、HiF4 量化、选定通道的激活量化残差补偿；普通注意力和批量 ragged 注意力的 grouped `wo_a` 都通过 ARC。
- `model/deepseek_arc_long.py` 将长输入按最多 4096 行计算，保持完整通道分组和校准选择，限制中间浮点张量。当前此扩展入口只接受 dense 对照，不能用来冒充已完成的 TEAL/WINA 长输入适配。
- 生成前将 ARC 源码和校准哈希写进原协议。使用普通 HiF4 答案续跑、替换校准或改变实现都会被拒绝。
- 续写准备会冻结外部评测器、ARC 源码和本次 ARC 64K 答案；前缀重建沿用原生协议，核验提示词哈希和文本回编码。切换前缀时同时清理 ARC 权重缓存。
- 每个 rank 留下投影类型覆盖、ARC 残差调用、普通/批量注意力调用和 logits 有限性检查记录。这仍然是 **HiF4 数值重建 + dense GEMM 的质量评测**，不能作为原生 FP4 内核加速结果。
- LongBench 使用与现有基线一致的 `longbench_explicit_answer_v2` 评分；LCB 使用相同 public/private 执行测试。

## 已验证和未验证

已通过两项 CPU 离线检查：503 / 175 题均可加载，模型 TP2 转换和 ARC 校准身份相符。针对性测试覆盖分块与原 ARC 数值计算的一致性、实际外部注意力函数的两条 ARC 接入、不同 ARC 身份拒绝混用，以及暂停父进程后子搜索仍自然结束。

本次部署时没有空闲 GPU；**新增长输入 ARC 的显存峰值、吞吐和最终得分尚未经 GPU 实跑验证**。队列内保留有限 logits 检查和失败记录，不会把 CPU 检查当成完整评测通过。失败后使用原目录续跑，不能删除原始答案或协议以绕过校验。

## 在其他机器上使用

先按 [迁移评测说明](MIGRATION_EVALUATION.md) 准备固定版本的外部 `base_eval`、模型、TP2 转换、正式校准和离线数据，并安装 `requirements-report.txt`。这里的交接工具适用于 Linux、同一台主机上的既有 MAGI 控制器和两张 H20；进程 PID 必须在目标机器重新获取。

单独启动 LongBench（不涉及任何已有进程）：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmarks/run_deepseek_arc_extended.py \
  --stage longbench-v2 --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration /path/to/formal/calibration \
  --out-dir runs/ds_arc_extended/longbench --quant hifp4 --mode w4a4
```

若需要本次相同的整套交接，先用 DS 的 Python 环境运行 `tools/prepare_deepseek_extended.py --help` 配置模型路径、控制器目录/PID、GPU 编号；该工具只做 CPU 预检和生成 plan。随后使用 **MAGI 原 Python 环境（本机为 3.12）** 运行：

```bash
python tools/handoff_deepseek_extended.py --plan /path/to/run/plan.json
```

请用 tmux 或服务管理器保持它常驻。已有活跃 `handoff.json` 时先检查进程，不要重复部署。待跑源码哈希已经固定；需要改代码时，先核查并更新未启动的计划，不能直接让旧 plan 跳过验证。
