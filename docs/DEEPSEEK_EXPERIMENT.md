# DeepSeek：HiF4、ARC 与 TEAL/WINA 实验入口

> **方法完整性：** 当前 TEAL/WINA 使用统一投影预算（uniform），没有 greedy
> 分配策略；WINA 未启用 transform。GPQA 标签似然入口包含 ARC，生成式稀疏入口
> 尚不包含 ARC。现有命令不能代表完整原生复现，详见 [预算核查](DEEPSEEK_SPARSE_BUDGET_AUDIT.md)。

当前机器已经跑完一组 198 题 GPQA Diamond 全量对照，结果表和边界说明见
[DeepSeek 完整结果](DEEPSEEK_FORMAL_RESULTS.md)。

`arc.py experiment` 是 DeepSeek V4 Flash 的校准与 GPQA 标签似然诊断入口。它把同一份
官方 TP2 权重、同一份校准数据和同一份 benchmark 题目固定下来，分别运行：

- `bf16`：原始 BF16 计算对照；
- `hif4`：HiF4 W4A4 数值路径，不使用 ARC 残差；
- `arc`：HiF4 W4A4 加 ARC 通道重排和残差通道；
- `teal`：在 ARC 前先按 TEAL 有符号激活直方图掩码；
- `wina`：在 ARC 前先按 WINA 的 `|x| × 权重输入列 L2` 做逐 token top-k。

当前实现使用外部 DeepSeek 官方 reference model 的 MLA、HC、MoE 和 TP 通信，
并把 HiF4、ARC、选通和评分元数据写进结果 JSON。矩阵乘法仍是 dense PyTorch
GEMM，所以这些结果用于精度对照；不能据此声称端到端加速。

## 运行顺序

先用 WikiText-2 train 做一次离线校准。校准包括每个真实投影输入的 ARC 顺序、
残差通道数、TEAL signed histogram 和每个 TP rank 的校准签名。校准不读取 GPQA。

```bash
source external/local_env.sh
python arc.py experiment --stage calibrate \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration runs/deepseek_formal/calibration \
  --calibration-data /path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --samples 128 --calibration-length 512 \
  --output runs/deepseek_formal/calibration_manifest_copy.json \
  --gpu-ids 0,1
```

然后用同一校准目录分别运行 benchmark。下面的命令使用 GPQA Diamond 的
198 题；`--limit 8` 只适合启动检查，不能作为正式成绩。

```bash
python arc.py experiment --stage evaluate --variant hif4 \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration runs/deepseek_formal/calibration \
  --data /path/to/gpqa_diamond.csv --limit -1 \
  --max-length 4096 --output runs/deepseek_formal/gpqa_hif4.json \
  --gpu-ids 0,1

python arc.py experiment --stage evaluate --variant arc --sparsity 0.5 \
  ...同样的路径参数... \
  --output runs/deepseek_formal/gpqa_arc.json --gpu-ids 0,1

python arc.py experiment --stage evaluate --variant teal --sparsity 0.5 \
  ...同样的路径参数... \
  --output runs/deepseek_formal/gpqa_teal.json --gpu-ids 0,1

python arc.py experiment --stage evaluate --variant wina --sparsity 0.5 \
  ...同样的路径参数... \
  --output runs/deepseek_formal/gpqa_wina.json --gpu-ids 0,1
```

### 50%/60%/70%/80%/90% 统一计划

当前 50% 结果保持不变。修复生成链路后，可以用下面的计划器生成同一 benchmark
下的五个正式稀疏度 WINA/TEAL 命令：

```bash
python arc.py sparse-plan --benchmark math500 \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --calibration runs/deepseek_formal/calibration \
  --limit 1 --max-new-tokens 128 --gpu-ids 0,1
```

计划器的配置来自 `configs/sparsity_profiles.json`。每一个运行目录都带有方法和
稀疏率，例如 `wina-0.9`；不能用 50% 的输出替代任何其他稀疏度的结果。完整评测前仍应
按 1 题、8 题、20 题逐级确认 EOS、有限 logits、答案格式和评分状态。

## 结果解释

当前 GPQA 入口复现的是 lm-eval `gpqa_diamond_zeroshot` 的题目模板，但本仓库
的实现只对答案标签 `(A)`、`(B)`、`(C)`、`(D)` 的条件 log-likelihood 打分，
并没有对四个完整选项文本打分，也没有让模型先生成 thinking。这是一个内部
方法诊断协议。它和历史报告中允许 thinking 的生成式 Pass@1 不是同一个指标，
不能直接比较百分数；正式方法结果应使用同一生成式协议重新评测。
结果 JSON 会保存题目哈希、选项置换、每题分数、数据集 SHA256、权重签名、
代码签名、TP 显存峰值和投影覆盖率。

### 重要实现边界

- 一个 `--sparsity` 统一应用到所有纳入掩码的投影，TEAL 的不同阈值不代表已经做了
  不同预算分配；该入口没有装载 greedy 策略的接口。采集直方图不等于完成预算搜索。
- ARC 的顺序和残差通道来自校准输入的 per-channel absmax；残差数按已有
  ARCQuant 的 `max/8` 规则取整到 64 通道。
- TEAL 使用 signed histogram 的 `0.5 + sparsity/2` 阈值；prefill 只掩码后半段，
  decode 掩码当前 token。
- WINA 对 MLA 的 `wkv` 和 MoE 的 `w1` 使用真实列范数；其他投影使用单位范数，
  与仓库已记录的 WINA Llama 接线保持一致。TP row-parallel 投影会先跨 rank
  合并 score，再切回本 rank。
- MoE 路由后的 token 行会保留原始 prefill 位置，因此不会把路由顺序误当成
  序列顺序；未被路由到的专家通过同一校准输入补齐。
- WINA 的 `--transform` 是 Llama/ARCQuant 旧模型路径的坐标变换；DeepSeek MLA、压缩 KV 和 grouped `wo_a` 的坐标不同，当前 DeepSeek 结果不启用这个开关，不能把 Llama transform 的结果套到 DeepSeek 上。
- 没有稀疏 fused kernel。当前 DeepSeek 结果用于方法精度和覆盖率核查；要报告速度，需运行 `benchmarks/sparse_latency.py` 的单投影诊断，并在目标 H20 上接入经过数值回归的稀疏 kernel。

## 校准规模和结果状态

仓库默认正式校准为 128 个 WikiText-2 片段、每段 512 token。TEAL 直方图对每个层/投影只保留确定性的有界样本（默认 262,144 个值），避免把整批激活长期堆在 CPU 内存；manifest 会记录样本数、长度、数据哈希、权重签名和代码哈希。旧的 `calibration80` 目录仍是 smoke 产物，不能覆盖正式校准目录。

DeepSeek 已完成的整模型方法协议是 GPQA Diamond 198 题的四选一条件似然：BF16、HiF4、ARC、TEAL、WINA 都走相同题目和 TP2 权重。MATH-500（500 题）、LongBench-v2（503 题）、LiveCodeBench v6（175 题）目前在仓库中登记了数据版本和外部生成式基线入口；要得到 ARCQuant+TEAL/WINA 的正式表格，必须把同一个稀疏 runtime 接入它们各自的生成解码器后再运行，不能复用 GPQA 的选择题代码。

LongBench-v2 和 LiveCodeBench 已提供 WINA/TEAL 生成入口。它通过本仓库的
`benchmarks/run_deepseek_sparse_extended.py` 注入外部调度器，结果文件会带
`arcquant-<method>-<sparsity>` 标签并把稀疏 runtime 哈希写入协议。普通
`linear`、MoE 路由后的专家投影和 grouped `wo_a` 输入都会经过对应方法；输出中的
`runtime.sparse_masked_elements` 可用于核查每个投影的实际覆盖率。

```bash
torchrun --nproc_per_node=2 benchmarks/run_deepseek_sparse_extended.py \
  --base-eval-root /path/to/base_eval --sparse-method wina --sparsity 0.5 \
  --benchmark longbench-v2 --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 --quant hifp4 --mode w4a4 \
  --limit 8 --out-dir runs/deepseek_sparse_longbench_smoke
```

TEAL 需要与 GPQA 校准使用同一个 TP rank 校准目录：

```bash
torchrun --nproc_per_node=2 benchmarks/run_deepseek_sparse_extended.py \
  --base-eval-root /path/to/base_eval --sparse-method teal --sparsity 0.5 \
  --calibration runs/deepseek_formal/calibration \
  --benchmark longbench-v2 --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 --quant hifp4 --mode w4a4 \
  --limit 8 --out-dir runs/deepseek_sparse_longbench_teal_smoke
```

MATH-500 使用同一 native thinking 与 Math-Verify 协议，入口是
`benchmarks/run_deepseek_sparse_math.py`：

```bash
torchrun --nproc_per_node=2 benchmarks/run_deepseek_sparse_math.py \
  --base-eval-root /path/to/base_eval --sparse-method wina --sparsity 0.5 \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --quant hifp4 --mode w4a4 --limit 8 \
  --out-dir validation/deepseek_math_wina_smoke
```

把 `--limit 8` 改为去掉 `--limit` 才是 500 题完整任务；正式任务应使用新的
`runs/` 目录并保留 `protocol.json`、`outputs.jsonl`、`summary.json`。Terminal-Bench
仍需要终端代理和执行沙箱，仓库只登记数据版本与题数，不把独立 `base-eval`
成绩冒充 ARCQuant 稀疏结果。
