# MAGI 的 ARC 适配：本轮实现和使用方法

更新：2026-11-25。**代码已实现，18 项针对性 CPU 测试通过；MAGI 整模型、真实视频的 GPU 验证尚未完成，不能把它报告成已有 ARC benchmark 成绩。**

## 先回答最重要的问题

- DS 之前已经用了 ARC。MAGI 之前只有普通 HiF4，缺的是通道重排与量化残差补偿。
- 本轮把 MAGI 的 ARC 计算、校准、搜索、视频生成接到同一条仓库内入口：`benchmarks/run_magi_arc.py`。
- 统一 CLI 同样支持：`python arc.py magi-arc --help`；分布式时可用 `torchrun ... -- arc.py magi-arc ...`。
- 旧 MAGI 成绩和旧 greedy 仍然是 no-ARC，保留原身份。现有 BF16/HiF4 Tiny 分数不修改。
- 这是数值精度实验：HiF4 数值重构后调用稠密 GEMM，不表示 H20 已经运行原生 FP4 或稀疏加速内核。

## ARC 究竟多做了什么

可以把一次线性层理解为“输入数字乘权重”。直接量化会把数字舍入，例如 1.1 变成 1.0，差掉的 0.1 就是量化误差。

1. **离线统计**：观察真实模型输入，记录每个通道的最大绝对值；按 ARC 原有 signed `x > max(x)/8` 规则估计需要保留多少补偿通道，向上对齐到 64。
2. **离线准备权重**：按统计顺序重排原始权重，再做 HiF4；为选中的通道另存一份量化权重。不能先普通 HiF4，再对已经舍入的权重假装做 ARC。
3. **在线执行**：TEAL/WINA 先把选中的激活置零；随后重排、HiF4 量化，把选中通道的“原值减量化值”再量化，作为附加输入列一起做矩阵乘法。

例如原输入是 128 个通道，某专家选中 64 个补偿通道，那么它的 GEMM 输入是 128+64 列。不同专家可以选择不同数量；批处理以零填充对齐，填充不会参与稀疏统计。**40% 指稀疏策略预算，并不表示补偿后的总 GEMM 成本也一定减少 40%。**

## 实际接了哪些地方

| 部分 | 实现 |
| --- | --- |
| preview/refiner 的 QKV、输出、前馈投影 | 分组输入保持原 modality 分发顺序，接 ARC 重排与残差 |
| routed MoE 的 gate/up/down | 保留原打包、FP32 GEMM、SwiGLU7、BF16 contribution 和确定性合并；每批专家限制临时显存 |
| refiner | 保持“矩阵乘法先舍入 BF16，再加 bias”的原顺序 |
| WINA | SVD 变换先于 ARC；通道统计对应变换后的坐标；权重范数从 ARC 主权重映射回选择用坐标 |
| final heads、attention gates 等未纳入稀疏的投影 | 沿用原 HiF4 处理；不误改为 BF16，也不冒充覆盖了 ARC |
| router、mHC、embedding/adapter、文本编码器、VAE | 沿用原运行器边界；没有增加稀疏选择 |

当前 MAGI 策略沿用此前的模型扩展：TEAL 按激活幅度做逐 token 精确 TopK，WINA 使用变换与权重范数。它**不是文本 TEAL 的直方图阈值路径，也不施加 2:4**。输出中的 2:4 统计只是观察分组零分布；ARC 附加通道成本另记，不能据此宣称实现结构化加速。

主要源码：[`arithmetic.py`](../integrations/magi_arc/arithmetic.py)、[`runtime.py`](../integrations/magi_arc/runtime.py)、[`moe.py`](../integrations/magi_arc/moe.py)。原 MAGI 扩展的迁入来源逐文件哈希在 [`UPSTREAM.json`](../integrations/magi_arc/UPSTREAM.json)。没有改动正在产生旧结果的 `base_eval/sparse_eval/magi`。

## 正确的离线和在线顺序

```text
已核验的普通 HiF4 输入轨迹
  → stats：生成 ARC 通道排列、补偿列数
  → capture：用真正 ARC 模型重新生成 10 条校准轨迹
  → search：70 个 block 在 ARC 下搜索配额
  → extract：提取实际访问过的 40% 预算
  → generate：先 1 个真实视频检查，再生成固定 Tiny 的 740 个视频
  → eval_vbench：16 维评分
```

普通 HiF4 轨迹只用作第一步的真实校准输入。ARC capture/search 会重新计算，不复用旧 no-ARC greedy 决策。策略、通道文件、模型文件清单、运行代码、两 rank 的实际执行均有身份检查。

**可以节约的部分**：dense 与 TEAL 使用原始通道坐标，共享一次 ARC 通道统计；WINA 另做统计。WINA 的原始权重 SVD 缓存可以按权重哈希复用，当前约 39 GiB，无需为 ARC 复制一份。greedy 只搜到当前选定的 40%，暂不展开五档。

## 使用入口

先准备依赖：现有 MAGI Python 3.12 环境、Hopper、官方 MAGI 源码、完整权重、VBench 数据和固定评测运行器。路径均由参数传入；不依赖 `/mnt/kpfs_qyang/LBC`。

```bash
python tools/prepare_magi_arc.py \
  --base-eval /path/to/base_eval \
  --model /path/to/MAGI-2-preview \
  --repo /path/to/MAGI-official-source \
  --teacher-root /path/to/completed-no-arc-calibration \
  --output /path/to/new-magi-arc-run
```

这会检查模型头信息、官方源码版本、普通 HiF4 输入轨迹身份，生成按依赖排列的 `plan.json`，不自动启动 740 视频任务。各条 command 可直接执行。`--base-eval-root` 指的是评测器代码目录，不是模型目录。

单个阶段的通用形式：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -- \
  benchmarks/run_magi_arc.py \
  --base-eval-root /path/to/base_eval \
  --model /path/to/MAGI-2-preview --repo /path/to/MAGI-official-source \
  --method teal --root /path/to/new-run/teal \
  --arc-calibration /path/to/new-run/none/channels \
  --asset-dir /path/to/existing-svd-cache \
  --stage capture --quant hifp4 --mode w4a4 \
  --protocol tiny16 --num-inference-steps 50 --refiner-num-inference-steps 5
```

先完成 plan 中的 `none-stats` 才能执行上面的 ARC capture。WINA 使用自己的 `wina/channels`。不能把普通 HiF4 的分布/策略目录作为 `--arc-calibration`。

## 验证与当前边界

CPU 检查已覆盖：ARC 原公式逐元素一致性、0/64/128 个补偿列、不同专家宽度、非连续权重、输入分块、padding、稀疏置零不被补偿恢复、分组投影、refiner 舍入、未适配 final head 保留 HiF4，以及 DS 长输入稀疏位置与旧答案严格复用。

GPU 验证程序是 `python -m integrations.magi_arc.validate --help`，已加入 DS 优先队列末尾。它检查官方 preview/refiner 模块、MoE、卸载恢复，并使用真实 checkpoint 矩阵对齐 ARC oracle；**通过这些检查仍不等于 VBench 分数**。随后需要真实单视频、完整校准和正式 Tiny。

当前磁盘约剩 150 GiB。参考已有数据，两套 ARC capture 约 138 GiB，三套 740 视频约 90 GiB，加统计及余量按约 **250 GiB** 准备（复用现有 SVD）。因此大批量生成尚未自动启动；应另选有空间的输出盘。不能删除旧视频或校准以偷偷腾空间。

## 迁移到另一台机器

新方法代码从 GitHub 克隆。旧运行器中还有未提交的必要实现，单靠旧 Git bundle 可能缺文件；本轮增加了精确文件快照导出：

```bash
python tools/export_magi_arc_runtime.py --base-eval /path/to/base_eval \
  --output /path/to/MAGI_ARC_runtime.tar.gz
tar -xzf /path/to/MAGI_ARC_runtime.tar.gz -C /path/to/runtime
```

包里是代码及模型/提示词清单，没有权重、环境、视频或凭据。`runtime_manifest.json` 保存文件哈希。使用 `/path/to/runtime/base_eval` 作为运行器目录；权重、官方模型源码、VBench、Python 环境按 [迁移指南](MIGRATION_EVALUATION.md) 准备。恢复旧轨迹还要求原 Python/数值代码与输入身份一致；仅查看或在新机器重新评测不要求沿用旧绝对路径。
