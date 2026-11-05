# 稀疏率要求核查

更新时间：2026-11-04。

> **范围纠正：** 下文保留的是此前根据开源 TEAL/WINA 代码进行的元素稀疏率核查，
> 不能当成结构化实验或项目验收规范。当前项目要求的正式扫描网格是 50%、60%、70%、80%、90%；
> 2:4 模式、层覆盖率、
> share index 和总精度损失必须另外核查。原 `formal_candidate` 配置名仅保留为兼容
> 别名，默认扫描现已改称 `baseline_sweep`。普通 2:4 的独立实现与证据见
> [结构化稀疏指南](STRUCTURED_SPARSITY.md)。

本页回答一个容易混淆的问题：现有实验是不是只要求 50% 稀疏，还是还应包含更高的稀疏率。

## 历史核查结论（只适用于原生元素稀疏率对照）

此前能确认的最高可信结论是：原生 WINA/TEAL 代码曾包含 65% 诊断点，而 TEAL 公开主结果主要集中在 40%–50%。现在的项目实验规范已经明确为 **50%、60%、70%、80%、90%**，因此 65% 只保留为历史诊断点，不进入正式五档表格；每一档都必须按同一协议独立运行。

## 证据

| 来源 | 原文或代码事实 | 应如何理解 |
| --- | --- | --- |
| 原生 WINA `scripts/wina/llama_3_8b.bash`、`scripts/teal/llama_3_8b.bash` | 循环测试 `0.25 0.4 0.5 0.65` | 65% 是原生复现实验的正式测试点之一，且不只用于 WINA |
| 原生 WINA `wina/greedyopt.py` | `--target_sparsity` 默认值为 `0.66`，帮助文本写的是 `Target effective sparsity` | WINA 的 greedy 分配默认以约 66% 的有效模型预算为目标；它不是“保留 66%” |
| 原生 TEAL README | 主要结果写为 `40-50% model-wide activation sparsity`，速度示例为 40% 和 50% | TEAL 公开主结果的证据集中在 40%–50%，不能从 greedy 的内部参数推断最终要求为 65% |
| 原生 TEAL `scripts/greedyopt.bash` | `--target_sparsity 0.9` | 这是逐层 greedy 查找分配表时的搜索目标，不是最终统一模型必须达到的 90% |
| ARCQuant 当前结果和文档 | DeepSeek × GPQA 的 TEAL/WINA 均记录为 `sparsity=0.5` | 50% 是当前已经实际跑完的点；它不是已被证明的最终实验要求 |

## “65% 稀疏”到底表示什么

本项目的 `sparsity` 表示**置零比例**：

```text
sparsity = 0.65
keep = 1 - sparsity = 0.35
```

对 WINA，输入维度为 `d` 时每个 token 保留约 `floor(0.35*d)` 个输入通道。对 TEAL，目标值通过 signed activation histogram 的分位点计算阈值，实际零值比例会受到激活分布、prefill/decode 阶段和本来就为零的输入影响，因此应同时记录目标稀疏率和运行时观测到的零值比例。

## 当前仓库状态

- CLI 和 DeepSeek 生成入口接受任意 `[0, 1)` 的稀疏率，代码没有把 50% 写死为唯一合法值。
- `configs/sparsity_profiles.json` 已登记原生 WINA、原生 TEAL、正式候选和高稀疏四组配置；`python arc.py sparse-plan` 可直接生成带方法和稀疏率标签的 DeepSeek smoke 命令。
- DeepSeek generation runtime 现在会记录目标稀疏率、可掩码元素数和每个投影的实际零值比例，便于检查四档目标是否真的落在对应输入路径上。
- 现有 DeepSeek × GPQA 正式结果只有 50%，对应配置和结果文件中的 `sparsity` 都是 `0.5`。
- WINA 50% 在 DeepSeek MATH-500 生成中已经出现单题和 20 题退化；因此 60%–80% 任务必须先通过 1 题和 8 题闸门。
- 目前没有 60%、70%、80% 或 90% 的正式结果，不能把 50% 表格复制到新一栏。

## 历史扫描建议（非结构化对照，不代表验收矩阵）

在生成链路通过 1 题和 8 题 smoke 检查后，按下表补实验：

| 方法 | 已有结果 | 正式补测网格 |
| --- | --- | --- |
| WINA | 50%（保留现有结果） | 50%、60%、70%、80%、90% |
| TEAL | 50%（与原生公开主结果对齐） | 50%、60%、70%、80%、90% |

每个新点都必须在元数据、结果路径和 CSV/TSV 中单独写出 `0.50`、`0.60`、`0.70`、`0.80` 或 `0.90`，并记录实际 mask ratio；不能用目标值替代观测值，也不能把 greedy 的内部目标当成最终模型稀疏率。

## 原生代码位置

本机保存的参考快照：

```text
/mnt/kpfs_qyang/zhaolianchao/work/wina-source  04ac8794aa4ae345d09414733ceb76470d3b6a3e
/mnt/kpfs_qyang/zhaolianchao/work/teal-source  fb7373c93ac3594817c9ee64d4e08b47430a1822
```

WINA 的测试点见 `scripts/wina/*.bash`，greedy 默认见 `wina/greedyopt.py`；TEAL 的公开稀疏范围见根目录 `README.md` 和 `scripts/ppl_test.bash`。
