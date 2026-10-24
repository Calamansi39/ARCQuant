# 实验入口与参数

首次使用请先读 [从零开始](QUICKSTART.md)。所有实验从仓库根目录运行 `python arc.py ...`；`configs/experiments.json` 保存可用模型、benchmark、数据源和状态。模型权重及数据本身不进 Git。

## 日常只需四个动作

```bash
python arc.py list                         # 看登记与适配状态
python arc.py fetch --model llama-3.1-8b  # 按清单下载一种资产
python arc.py plan ...                     # 核对命令和路径，不用 GPU
python arc.py run ...                      # 真正运行并记录到 runs/
```

对于 ARCQuant + Llama，先运行 `python arc.py prepare --model llama-3.1-8b` 生成 `saved/` 中的量化校准缓存。TEAL 再运行 `python arc.py calibrate --model llama-3.1-8b --method teal --sparsity 0.5` 生成直方图。WINA 的 `uniform` 分配不用额外离线数据；`greedy` 需要 `calibrate --allocation greedy`。相同方法的 `plan/run` 默认在资产目录自动查找这些产物。

```bash
python arc.py run --model llama-3.1-8b --method none --quant-type HiF4 --benchmark math500 --profile smoke
python arc.py run --model llama-3.1-8b --method teal --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke
python arc.py run --model llama-3.1-8b --method wina --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke
```

`--sparsity` 是目标稀疏率，取 `[0,1)`；`--allocation uniform|greedy` 决定七个投影用相同率还是离线逐投影分配；`--disable-prefill-sparsity` 只在解码时掩码。`--model-path` 可覆盖权重目录，`--histogram-path` / `--strategy` 可覆盖离线产物路径。`--quant-type HiF4` 指 W4A4 数值路径，不代表已生成压缩权重或启用了稀疏内核。`--method xmask --x-mask-ckpt FILE` 走学姐原有的另一条 2:4 路径，当前不与 TEAL/WINA 同时启用。

## 当前适配范围

| 模型 | ARCQuant 自有量化 | TEAL/WINA | 外部生成基线 |
| --- | --- | --- | --- |
| Llama 3.1 8B | 原有适配代码 | 已接入 Llama 投影，需真实模型回归 | 无 |
| Qwen 2.5 7B、Mixtral 8x7B | 原有适配代码 | 未接入 | 无 |
| DeepSeek V4 Flash | 未接入 | 未接入 | GPQA、MATH-500、LongBench v2、LiveCodeBench |
| MiniMax H3、MAGI-2 | 未接入 | 未接入 | VBench |

Llama 的 `math500` 和 `gpqa-diamond` 使用 `lm-evaluation-harness` 任务；DeepSeek 的同名任务通过 `base-eval` 的生成式协议评测，不能直接并表。`terminal-bench` 目前只登记数据，尚无代理/容器评测入口。`model/eval_longbench.py` 是旧 LongBench，不是 LongBench v2。模型或 benchmark 状态可由 `python arc.py list` 查看，不能把“已登记”理解为“已跑通”。

外部基线的启动及多环境依赖见 [资产与基线](ASSETS_AND_BASELINES.md)。方法代码来源、离线/在线结构、WINA 可选变换尚未接入的原因见 [实现核查](METHOD_PROVENANCE.md)。需要卡时估计时看 [GPU 预算](GPU_BUDGET.md)。
