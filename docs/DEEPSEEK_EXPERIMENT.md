# DeepSeek：HiF4 + ARC + TEAL/WINA 正式评测

当前机器已经跑完一组 198 题 GPQA Diamond 全量对照，结果表和边界说明见
[DeepSeek 完整结果](DEEPSEEK_FORMAL_RESULTS.md)。

`arc.py experiment` 是 DeepSeek V4 Flash 的完整实验入口。它把同一份
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
  --samples 128 --calibration-length 256 \
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

## 结果解释

当前 GPQA 入口复现的是 lm-eval `gpqa_diamond_zeroshot` 的题目模板和选项
似然评分：每个选项独立计算条件 log-likelihood，最高者作为答案。它和历史
报告中原生 thinking 的生成式 Pass@1 不是同一个指标，不能直接比较百分数。
结果 JSON 会保存题目哈希、选项置换、每题分数、数据集 SHA256、权重签名、
代码签名、TP 显存峰值和投影覆盖率。

### 重要实现边界

- ARC 的顺序和残差通道来自校准输入的 per-channel absmax；残差数按已有
  ARCQuant 的 `max/8` 规则取整到 64 通道。
- TEAL 使用 signed histogram 的 `0.5 + sparsity/2` 阈值；prefill 只掩码后半段，
  decode 掩码当前 token。
- WINA 对 MLA 的 `wkv` 和 MoE 的 `w1` 使用真实列范数；其他投影使用单位范数，
  与仓库已记录的 WINA Llama 接线保持一致。TP row-parallel 投影会先跨 rank
  合并 score，再切回本 rank。
- MoE 路由后的 token 行会保留原始 prefill 位置，因此不会把路由顺序误当成
  序列顺序；未被路由到的专家通过同一校准输入补齐。
- 目前没有移植 WINA 的可选 `--transform` 全模型权重变换，也没有稀疏 fused
  kernel。若论文需要完整 WINA transform 或真实 latency，必须单独增加实现和
  速度实验，不能复用本页的精度数字。
