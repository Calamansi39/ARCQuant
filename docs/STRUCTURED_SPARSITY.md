# 普通 2:4：实现、调用与证据

## 先分清两个比例

- **每组四取二**：每个 token 的输入沿最后一个通道维度分组，每 4 个位置按绝对值保留 2 个。
- **层覆盖率**：有多少个线性投影采用上述掩码。`--layer-coverage 0.8` 表示请求覆盖当前统计范围至少 80% 的投影，不是置零 80% 的元素。

这是一条新的 `magnitude-structured24` 基础对照，复用仓库 `x_mask.py` 中按幅值四取二的
基本规则，但不使用其可训练门控和 dense/sparse 软混合，也不称为 TEAL 或 WINA。
本路径在量化前施加掩码，不添加 ARC 残差通道。它采用外部固定评测器的 HiF4 权重/
激活量化与 dense GEMM；没有实现稀疏硬件加速或 share index。

## 已接入的范围

- DeepSeek V4 Flash：Attention 和 FFN 的 `ds.Linear` 投影，包含 MoE 专家。
- grouped `wo_a` 通过正在执行的 Attention 实例确定归属，覆盖普通 prefill 和 ragged decode。
- 所选投影处理所有 prefill/decode 行，不保留一个未掩码的前缀。
- 模型路由器、embedding、head、HC plumbing 不在当前统计范围；报告明确列出范围。
- MiniMax H3、MAGI-2 尚未接入此结构化路径。

分母当前是**唯一命名的线性投影数量**，不是 Transformer block 数量，也不是权重或
FLOPs 占比。MoE 专家分别计数；同名 TP 副本只计一次，两个 rank 的 inventory 先合并。
Attention/FFN 分项覆盖率另列。宽度不被 4 整除的投影留在分母且不能选入，覆盖要求
无法满足时直接报错。组的边界要求各 TP 输入分片宽度被 4 整除。

`--scope attention|ffn|all` 决定选择范围。只选择 Attention 时，报告同时保留 Attention
范围覆盖率和全部线性投影覆盖率，不能把局部 100% 报成全模型 100%。最终验收采用
哪种“层”的定义，需要另行约定。

## 从准备到运行

在匹配的 Python/PyTorch 环境中运行，设置本机路径：

```bash
export ARCQUANT_BASE_EVAL_ENV=/path/to/arcquant-env
export ARCQUANT_BASE_EVAL_ROOT=/path/to/base_eval
export ARCQUANT_DEEPSEEK_MODEL=/path/to/DeepSeek-V4-Flash-0731
export ARCQUANT_DEEPSEEK_TP2=/path/to/DeepSeek-V4-Flash-0731-mp2

python arc.py structured-plan --benchmark math500 \
  --base-eval-root "$ARCQUANT_BASE_EVAL_ROOT" \
  --source-model "$ARCQUANT_DEEPSEEK_MODEL" \
  --converted-model "$ARCQUANT_DEEPSEEK_TP2" \
  --layer-coverage 0.8 --scope all --limit 1 \
  --max-new-tokens 128 --smoke-steps 4 --gpu-ids 0,1 \
  --out-dir validation/structured24
```

- `structured-plan` 只显示命令，不加载 GPU 模型。
- 改成 `structured-run --dry-run` 会调用原生评测器检查题库、tokenizer 和协议，不加载模型权重到 GPU。
- 去掉 `--dry-run` 真正运行；加 `--wait-for-gpus` 自动等待两张空卡并选择卡号。
  等待模式不使用固定的 `--gpu-ids`，默认要求每卡空闲显存 ≥138000 MiB、利用率 ≤5%、无 compute 进程。
  换硬件需重新确认显存需求，可用 `--min-free-mib` 和 `--poll-seconds` 配置。
  同账号的本机队列使用文件锁避免相互重复选卡，其他用户仍可能在检查后启动任务。
- `--smoke-steps 4` 只运行有限 decode 步，原生 finite-logits 检查仍生效，未完成输出不评分。
  输出必须位于本仓库或 base_eval 的 `validation/` 中。此选项仅用于 MATH/GPQA。
- 去掉 `--smoke-steps` 才会生成到 EOS/长度上限；128 tokens 只是调试预算，不能作为正式质量协议。
- `--benchmark` 可换成 `gpqa-diamond`、`longbench-v2`、`livecodebench`。
  GPQA 使用原生 thinking **生成式**协议，与历史四选项似然结果分开。
  GPQA 请去掉示例中的 `--max-new-tokens 128`：默认使用 8192，原生协议为最终答案
  预留 1024 tokens，不能用 128 的总预算；可保留 `--smoke-steps 4` 来限制调试步数。
  LongBench/LiveCodeBench 在此入口按 batch-size=1 运行，不支持 bounded smoke steps。

CLI 默认使用已准备的离线 Hugging Face 缓存；缺失数据时先运行资产准备。
确实需要在线下载时可显式设置 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`。

覆盖率为 0 时是相同运行时的 W4A4 dense 对照。正式对比还需同平台、同协议的原始
位宽基线；本工具不会把 dense W4A4 自动当成原始位宽基线。

## 选择哪些层

没有空闲 GPU 时，可以先读取 TP 权重文件的头信息生成完整投影清单和选层文件；
此操作不加载权重张量，也不执行模型：

```bash
python benchmarks/inspect_structured_checkpoint.py \
  --converted-model "$ARCQUANT_DEEPSEEK_TP2" --layer-coverage 0.8 \
  --output validation/structured24_inventory.json \
  --write-selection validation/structured24_selection.json
```

这是 checkpoint 布局核查，运行时还会从实际模型实例重新构建并核对选择。

本机完整 TP2 清单合并后有 33,534 个目标投影：Attention 381、FFN 33,153。
分项请求 0.8 时选择 Attention 305 个、FFN 26,523 个，共 26,828 个。
这些是配置数量，尚未证明每个投影都在真实输入中被执行或质量达标。

默认在 Attention、FFN 内分别按投影名 SHA256 的固定次序选
`ceil(coverage × area_count)` 个投影，避免大量 MoE 专家掩盖 Attention 覆盖不足，保证可复现。
**没有根据敏感度优化层选择**。因此这里的 .8 只是配置能力，不表示精度已经达标。
可以用 `--selection selection.json` 提供显式投影名列表，例如：

```json
["layers.0.attn.wq_a", "layers.0.ffn.shared_experts.w1"]
```

显式列表仍需在每个所选范围内达到请求的覆盖率，未知名字、重复名字、范围外名字和不兼容宽度都会报错。
正式列表应根据完整 inventory 和离线敏感度分析生成，不能直接套用上述示例。

## 报告如何看

每次运行目录包含源代码与配置指纹，改变掩码代码/覆盖率/选层清单会分开输出，防止
断点续跑混合不同方法。保留原生 protocol、输出与评分文件，并新增：

- `structured_config.json`：方法、覆盖率、选层规则、代码哈希。
- `structured_rank0.json` / `structured_rank1.json`：各 rank 的全局投影清单、本地实际执行次数、元素计数和失败状态。

合并各 rank 的观察结果：

```bash
python benchmarks/structured_report.py \
  validation/structured24/structured24-<fingerprint>/structured_rank0.json \
  validation/structured24/structured24-<fingerprint>/structured_rank1.json \
  --output validation/structured24/audit.json
```

报告区分**配置覆盖**和**实际观察到的覆盖**。未路由到的专家不算已验证；缺失 rank
会列出。检查掩码每组恰好保留 2 个位置，以及量化后每组非零值不超过 2 个。
量化可能引入额外零，因此 `mask_zero_fraction` 应为 0.5，而 `post_quant_zero_fraction`
可能更高。统计用于诊断，会产生额外同步开销，不适合作为性能测速实现。

即使结构检查通过，`acceptance_status` 仍为 `not_evaluated`：它没有检查完整任务质量、
相对原始位宽的损失、共享索引、吞吐或能效。

## 2026-11-04 实测

在一张空闲 H20 上加载官方 TP2 的三个真实投影：FP8 `wq_a`、BF16 grouped `wo_a`、
packed FP4 专家 `w1`，分别使用长度 1 和 8 的**合成激活**。

| 检查 | 结果 |
| --- | --- |
| 真实权重＋HiF4 W4A4 投影计算 | 6/6 输出有限，无 NaN/Inf |
| 掩码结构 | 检查 55,296 个四元组，违规 0 |
| 量化后结构 | 违规 0 |
| 主动掩码置零比例 | 50.00% |
| 量化后观测零比例 | 约 50.074% |
| 峰值 PyTorch allocated 显存 | 约 0.211 GiB |
| 相对同输入 HiF4 dense 输出的 L2 差异 | 约 0.363–0.385，不是任务精度损失 |

原始本地报告：`validation/structured24_projection_probe.json`。复现工具：
`benchmarks/probe_structured_projection.py --help`，只需加载三个权重，无需整模型。
这验证了数值接线和结构约束，没有验证真实激活分布、整模型生成质量或性能收益。

四个目标文本 benchmark 的原生 CPU dry-run 已通过：MATH-500、GPQA Diamond、
LongBench v2、LiveCodeBench。GPQA 采用 8192 的原生预算，短测试用 bounded smoke；
其余三个 dry-run 使用 128 的调试预算。dry-run 不运行模型，不能视作 benchmark 完成。
