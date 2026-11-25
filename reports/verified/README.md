# 统一实验结果

核查时间：2026-11-20T06:30:45.439718+00:00

**本目录是唯一结果发布入口。[彩色 Excel](实验结果.xlsx)、CSV 和 TSV 使用同一份核验数据，同步更新。**

| 评测范围 | CSV | TSV |
| --- | --- | --- |
| DS 稀疏度 Tiny 横评 | [DS_Tiny.csv](DS_Tiny.csv) | [DS_Tiny.tsv](DS_Tiny.tsv) |
| DS 全量 dense 对照 | [DS_全量dense.csv](DS_全量dense.csv) | [DS_全量dense.tsv](DS_全量dense.tsv) |
| MAGI Tiny | [MAGI_Tiny.csv](MAGI_Tiny.csv) | [MAGI_Tiny.tsv](MAGI_Tiny.tsv) |

## DS Tiny

GPQA 固定 32 题、MATH 固定 100 题，生成上限 8192 token。0% 的 BF16 和普通 HiF4 从已完成全量逐题记录中提取；题目、提示词、seed 和当前评分规则已逐项核对，没有重新推理。

这些是固定测试集前缀上的初筛结果，不能替代全量结论；“评测中”表示该范围尚无结果，不表示有进程正在运行。

未截断率是自然 EOS 比例，不等于有效回答比例；90% 稀疏的提前 EOS 虽然不截断，准确率仍为零。

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | GPQA | GPQA 未截断率 | MATH | MATH 未截断率 | LongBench | LiveCodeBench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS | 无 | 0% | BF16 | 84.4% | 96.9% | 100.0% | 100.0% | 评测中 | 评测中 |
| DS | 无 | 0% | HiF4 W4A4 | 75.0% | 87.5% | 100.0% | 100.0% | 评测中 | 评测中 |
| DS | 无 | 0% | HiF4 W4A4 + ARC | 87.5% | 96.9% | 100.0% | 100.0% | 评测中 | 评测中 |
| DS | TEAL | 40% | HiF4 W4A4 + ARC | 62.5% | 65.6% | 95.0% | 96.0% | 评测中 | 评测中 |
| DS | WINA | 40% | HiF4 W4A4 + ARC | 53.1% | 68.8% | 99.0% | 99.0% | 评测中 | 评测中 |
| DS | TEAL | 50% | HiF4 W4A4 + ARC | 46.9% | 56.3% | 94.0% | 94.0% | 评测中 | 评测中 |
| DS | WINA | 50% | HiF4 W4A4 + ARC | 37.5% | 46.9% | 93.0% | 93.0% | 评测中 | 评测中 |
| DS | TEAL | 60% | HiF4 W4A4 + ARC | 9.4% | 12.5% | 71.0% | 72.0% | 评测中 | 评测中 |
| DS | WINA | 60% | HiF4 W4A4 + ARC | 3.1% | 3.1% | 57.0% | 57.0% | 评测中 | 评测中 |
| DS | TEAL | 70% | HiF4 W4A4 + ARC | 0.0% | 3.1% | 0.0% | 0.0% | 评测中 | 评测中 |
| DS | WINA | 70% | HiF4 W4A4 + ARC | 0.0% | 0.0% | 0.0% | 0.0% | 评测中 | 评测中 |
| DS | TEAL | 80% | HiF4 W4A4 + ARC | 0.0% | 15.6% | 0.0% | 17.0% | 评测中 | 评测中 |
| DS | WINA | 80% | HiF4 W4A4 + ARC | 0.0% | 3.1% | 0.0% | 3.0% | 评测中 | 评测中 |
| DS | TEAL | 90% | HiF4 W4A4 + ARC | 0.0% | 100.0% | 0.0% | 100.0% | 评测中 | 评测中 |
| DS | WINA | 90% | HiF4 W4A4 + ARC | 0.0% | 100.0% | 0.0% | 100.0% | 评测中 | 评测中 |

## DS 已完成全量 dense

GPQA 198、MATH 500、LongBench 503；LiveCodeBench 是 release-v6 的 test6 分片 175 题。64K 为独立生成，128K 为到限题续写，两者分别列出。普通 HiF4 不含 ARC。

**此表是成绩表，不是实时运行队列。ARC 行的 LongBench / LiveCodeBench 缺失格继续沿用“评测中”占位；它们的实际启动状态请看 [DS 调度记录](../../docs/DEEPSEEK_DENSE_PRIORITY.md)。**

ARC dense 续跑进度更新：2026-11-23T02:10:20.589394+00:00。已复用同协议 Tiny 的 132 道答案；未完成项不填中途准确率。[调度与续跑说明](../../docs/DEEPSEEK_DENSE_PRIORITY.md)。

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | GPQA | MATH | LongBench | LiveCodeBench 64K | LiveCodeBench 128K续写 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS | 无 | 0% | BF16 | 86.87% | 98.60% | 50.30% | 84.57% | 94.29% |
| DS | 无 | 0% | HiF4 W4A4 | 78.79% | 98.60% | 47.51% | 73.14% | 93.14% |
| DS | 无 | 0% | HiF4 W4A4 + ARC | 85.35% | 98.00% | 评测中 | 评测中 | 评测中 |

## MAGI 已完成 Tiny dense

108 个唯一 prompt、740 个视频、16 维。两组都是稠密模型；HiF4 采用 ARCQuant 来源的量化格式，但**没有 ARC 重排/残差补偿**。先前将 79.1241% 标作 ARC 成绩不准确，现已更正。

| 模型 | 稀疏方法 | 稀疏度 | 量化方式 | Total | Quality | Semantic | 视频数 | 评分维度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAGI-2 | 无 | 0% | BF16 | 80.0008% | 84.8416% | 60.6373% | 740 | 16 |
| MAGI-2 | 无 | 0% | HiF4 W4A4 | 79.1241% | 83.9607% | 59.7779% | 740 | 16 |

## Dense 还缺什么

| 模型 | 对照 | 范围 | 状态 | 后续 |
| --- | --- | --- | --- | --- |
| DS | HiF4 W4A4 + ARC，0% | GPQA 198 / MATH 500 / LongBench 503 / LiveCodeBench v6 test6 175 | gpqa 198/198；math500 500/500；LongBench/LiveCodeBench 待适配核验 | GPQA/MATH 全量已完成；下一步核验 LongBench/LiveCodeBench 的 ARC 接入 |
| MAGI-2 | HiF4 W4A4 + ARC，0% | VBench Tiny 740 视频 | 尚无 ARC 对照；79.1241% 是普通 HiF4，无 ARC | 先完成 ARC 适配；与方法使用同一量化基座 |
| MiniMax-H3 | BF16 / HiF4 W4A4，0% | VBench Tiny | 本机只有历史评分文件，缺生成视频/配置；BF16 静态覆盖仅 46/50 | 从 A800 同步原始证据、补齐 BF16 覆盖，再确认是否需要重跑 |
| MiniMax-H3 | HiF4 W4A4 + ARC，0% | VBench Tiny 740 视频 | 尚无可核验的 ARC 对照 | 完成适配后补齐 |
| DS（可选） | dense | Terminal-Bench 86 任务 | 尚无结果，代理和容器评测协议尚待接入 | 仅在保留此 benchmark 时安排 |

DS 的 BF16/普通 HiF4 全量基线、三种 dense 的同题 Tiny、MAGI BF16/普通 HiF4 Tiny 已有结果，无需重复运行。视频官方全集不属于当前首轮 Tiny 选点的必做项。

## 核验与更新

- `audit.json` 保存精确计数、题目 ID、来源路径、文件 SHA256 和纳入/排除原因。
- MAGI 生成身份、740 个视频的记录与文件、16 个评分结果、评分输入绑定、官方加权总分均已核验。是否重算视频字节哈希记录在 `video_bytes_verified`。
- 原始逐题输出、视频、模型、日志与搜索策略保留在原运行目录；删除的是过时展示表。
- MiniMax 历史评分缺少本机生成证据，BF16 静态覆盖不足，不进入可信主表。30% 的 8/16 题小探针与主表的 32/100 题范围不同，不混报。
- GPQA 使用生成式得分；旧标签条件似然成绩不混入。当前没有 acc_norm 指标可替换这些生成式得分。

在仓库根目录刷新（不使用 GPU）：

```bash
python -m pip install -r requirements-report.txt
python tools/build_verified_results.py --base-eval /path/to/base_eval --verify-video-bytes
# 其他机器只查看/重建已核验快照：
python tools/build_verified_results.py --snapshot-only
```
