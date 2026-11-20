# 三个目标模型：做到哪一步

更新于 2026-11-20。分数、题数和 dense 缺项只维护在 [统一结果目录](../reports/verified/README.md)；此页解释适配阶段。

## DS（DeepSeek V4 Flash）

- 原版权重、TP2 转换版及 GPQA、MATH、LongBench、LiveCodeBench 数据已接入。
- BF16、普通 HiF4 的四项完整生成式基线已完成。LiveCodeBench 的范围是 release-v6 的 test6 分片 175 题；128K 续写单列，不能冒充独立生成结果。
- ARC HiF4 W4A4 的 0% Tiny 对照，以及 TEAL/WINA 40%–90% 的 GPQA 32 题、MATH 100 题均已完成。BF16/普通 HiF4 的相同 Tiny 题目已从全量结果提取。
- ARC dense 全量、选定稀疏度的全量结果仍待补齐；30% 当前是搜索/小探针阶段。Tiny 选点结果不能填入全量成绩。
- 方法运行入口见 [DeepSeek 实验](DEEPSEEK_EXPERIMENT.md)、[greedy](DEEPSEEK_GREEDY.md)、[Tiny 调度](DEEPSEEK_TINY_SWEEP.md)。单卡搜索见 [单卡双 rank 说明](DEEPSEEK_SINGLE_GPU_SEARCH.md)，生成评测仍使用 TP2。

## MAGI-2 preview

- preview、refiner、文本编码器、VAE 与 VBench 数据已接入；已有完整 BF16、普通 HiF4 Tiny：每组 108 个唯一 prompt、740 个视频、16 维评分。
- **上述 HiF4 不含 ARC 通道重排和残差补偿。** 已对照生成时记录的源码哈希与量化实现核实，不能写成 ARC dense 成绩。
- 稀疏适配与搜索代码见 [MAGI 适配](MAGI_ADAPTATION.md)；搜索进度或单视频检查不等于完成 VBench。尚无可核验的 ARC dense 或正式稀疏 Tiny 成绩。
- 下一步先补 ARC 运行覆盖和 0% 对照，再让稀疏方法使用相同量化基座。迁移要求见 [H20 交接](MAGI_H20_HANDOFF.md)。

## MiniMax H3

- 目标是文生视频，VBench Tiny 与 MAGI 使用相同提示词清单；部署说明见 [A800 交接](MINIMAX_A800_HANDOFF.md)。
- 本机仅有历史评分文件，缺原始生成视频、配置等证据；BF16 静态视频覆盖只有 46/50。暂不把这些历史分数放入可信主表。
- 应先从 A800 同步生成证据，补齐 BF16 覆盖并重新核验；已有合格部分可以复用，无需直接重跑全部。ARC dense 和稀疏正式成绩仍待验证。

## 文件准备与评测完成的区别

`assets/` 是本机模型和数据的入口；模型权重、视频、原始输出不上传 GitHub。换机器按 [迁移指南](MIGRATION_EVALUATION.md) 准备资产，再检查：

```bash
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark terminal-bench --benchmark vbench
```

文件检查通过只代表前置资产就绪，不能替代真实生成、评分和方法运行证据。Terminal-Bench 86 任务尚需接入代理与容器协议，属于可选后续。

这些模型的投影、MoE、视频去噪流程各不相同，不能直接沿用 Llama 的投影表。已有 DS ARC/greedy 实验走专用入口；普通 HiF4 数值路径与 ARC 补偿路径必须明确区分。当前质量结果也不等于使用真实 FP4/稀疏内核后的加速结果。
