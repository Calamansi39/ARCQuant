# 当前评测优先级

更新：2026-11-25。**DS 40% 稀疏 + HiF4 W4A4 + ARC 全量优先于 MAGI。** 旧“等 MAGI 70 层结束，再跑 DS dense，再恢复 no-ARC MAGI”的队列已撤换。

| 优先级 | 工作 | 范围与目的 | 状态 |
| --- | --- | --- | --- |
| 1 | DS 40%，TEAL/WINA，全量 GPQA、MATH500 | 各方法 198+500 题；确认 Tiny 选点能推广到全量 | 两台 H20 并行执行 |
| 2 | DS 40%，TEAL/WINA，全量 LongBench、LCB | 各方法 503+175 题；覆盖长上下文和代码 | TEAL LongBench 已启动，其余共享队列自动领取 |
| 3 | DS ARC dense 缺失对照 | LongBench 503、LCB 175；GPQA/MATH 已完成，不重跑 | 排在尚未领取的 40% 任务之后 |
| 4 | MAGI ARC 的 GPU 数值验证 | 官方模块、真实权重矩阵、卸载恢复 | 已列队；前面的 DS 任务全部完成后可执行 |
| 5 | MAGI ARC 校准、单视频、40% 策略与 Tiny | dense/TEAL/WINA；固定 740 视频、16 维 | 代码与依赖计划已准备；需真实 GPU 验证及足够输出盘空间 |

**暂不增加**：DS 五档全量、MAGI no-ARC 后续全套生成、额外 FP4 格式遍历、128K 续写新批次。现有结果保留。LCB 此轮主结果用固定 64K；历史 128K 续写单列，不和主结果混算。

## 当前机器和任务

本轮先核查时，H20-1 的 8 卡均被别人使用，H20-2 仅 2、3 是我们的 MAGI。随后 H20-1 全部释放，已部署共享任务队列，分成四个双卡 worker。

| 主机 | GPU | 已启动工作 |
| --- | --- | --- |
| H20-1 / 10.0.2.5 | 0、1 | DS TEAL 40% MATH500 全量 |
| H20-1 / 10.0.2.5 | 2、3 | DS WINA 40% GPQA 全量 |
| H20-1 / 10.0.2.5 | 4、5 | DS TEAL 40% LongBench 全量 |
| H20-1 / 10.0.2.5 | 6、7 | DS WINA 40% MATH500 全量 |
| H20-2 / 10.0.2.54 | 2、3 | DS TEAL 40% GPQA 全量 |

这是启动快照；载入权重期间利用率低，不能据此当成空卡。每次查看以 GPU 进程与下面的状态文件为准。文件锁仅约束遵守锁的队列，不能承诺绝对阻止外部作业抢卡。

## 旧成果怎么保留

- MAGI 原 HiF4 WINA 搜索：保存至 preview 第 9 个索引 block 的第 11 轮，前 9/70 个 block 完整。每个已保存决策用原 Python 3.12 重放验证后才让卡；没有删除文件。
- DS 两方法的 GPQA/MATH 复用 264 条同配置 Tiny 答案：GPQA 各 32，MATH 各 100。只改目标协议校验和，原答案、seed、评分保留；其余 1,132 条需要生成。
- 新全量未完成前继续显示“评测中”，不会把 Tiny 分数填进全量表。

## 队列文件和自动发布

运行目录：`runs/deepseek_arc40_full_20261125/`。

- `dispatch-plan.json`：跨主机共同的 GPU 工作列表，包含实际命令。
- `dispatch/h20-1-gpu*.json`：H20-1 各 worker 状态；`-v2` 是修复 torchrun 参数解析后的接续 worker。
- `handoff.json`：H20-2 当前 TEAL GPQA，完成后转入同一共享派发器。
- `dispatch/*.lock` / `*.done.json`：每项独占领取和完成证明；单个任务失败会保留答案及错误，不自动反复重跑。
- `plan.json` 的 `report_jobs`：统一表需要核验的原始结果路径；不会因分配到不同机器而丢列。
- 完成后检查覆盖数、ARC 两 rank 执行、greedy 策略与评分，再自动更新 `reports/verified` 的 CSV/TSV/彩色 Excel。并发发布有互斥锁。

H20-1 初次启动 LongBench 时，torchrun 把评测器的 `--n` 误判成自己的缩写参数；在实际推理之前失败。已用 `--` 分隔修复，接续目录保持原协议，无答案损失。

MAGI ARC 下一阶段的机器计划在 `runs/magi_arc_20261125/plan.json`；方法说明见 [MAGI ARC](MAGI_ARC.md)。该计划未自动放行大量视频生成，原因是 GPU 验证未完成且共享盘空间不足。
