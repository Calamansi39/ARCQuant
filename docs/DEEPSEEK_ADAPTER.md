# DeepSeek V4 Flash 投影适配：整模型 GPQA 与生成入口

本页记录已经写进仓库并通过 CPU 检查的代码，以及仍然需要 GPU 复核的部分。DeepSeek 的专用 `arc.py experiment` 已可在 TP2 reference model 上运行 HiF4、ARC、TEAL、WINA 的 GPQA 198 题；通用 `arc.py run --backend arcquant` 仍然只面向 Llama/Qwen/Mixtral，因此不会把 DeepSeek 当成旧模型路径加载。

## 代码做了什么

`model/deepseek_adapter.py` 以 DeepSeek V4 Flash 0731 官方权重命名为准，识别注意力主投影、压缩器与索引器投影，以及共享/路由专家的 `w1/w2/w3`。路由器、归一化和超连接权重不进入这条路径。官方主干索引中全部 33,534 个这些投影的权重键均能识别。

官方权重不全是普通浮点矩阵：注意力等位置有带缩放网格的 FP8，路由专家使用两个数放进一个字节的 FP4。适配器先把**当前单个矩阵**还原成数值，再执行：

```text
投影输入 → 可选 TEAL 阈值 / WINA top-k → ARCQuant 通道重排
         → HiF4 权重及激活的数值量化 → 增强残差通道 → 投影输出
```

增强残差通道需要该投影自己的重排索引和通道数。代码要求传入目标模型实测的校准结果；缺失或错误的索引会报错，不会临时套用旧模型的缓存。`wo_a` 在官方代码中直接参与分组 `einsum`，普通线性层钩子无法覆盖；适配器提供了单独的 `grouped_wo_a` 投影实现。

路由专家收到的二维输入只包含被路由选中的 token，行号不再等于原序列位置。GPQA 整模型接线会由路由代码传入 `sparse_rows`，生成式入口也会在 MoE dispatch 时恢复这些行的位置，因此 prefill 和 decode 规则不会混用。生成式 LongBench/LiveCodeBench 通过 `benchmarks/run_deepseek_sparse_extended.py` 接入 WINA 或 TEAL；`wo_a` 的 grouped contraction 会在官方 quantize-activation 边界执行同样的逐组掩码，运行时会把每个投影的掩码数量写入结果。

TEAL 使用校准得到的直方图阈值；WINA 根据当前输入绝对值与列范数选 top-k。DeepSeek 的层结构与原论文 Llama 不同，因此目前**没有**把 Llama 的 `h1/h2` 阈值文件或 K/gate 权重选择方案冒充 DeepSeek 的配置。调用者必须指定每个投影的 TEAL 直方图映射，或指定哪些 DeepSeek 投影使用 WINA 的列范数。

## 做过哪些检查

- `python arc.py assets ...`：三个目标模型和登记的六类 benchmark 均报告 `ready`。这些主要是本机已有文件的软链接；检查索引、分片头和数据条数，未对全部权重逐字节校验。
- 官方 DeepSeek 0731 权重索引：33,534 个主干投影权重键全部符合适配器的命名规则。
- 从官方分片实际读取一个 FP8 注意力矩阵和一个打包 FP4 路由专家矩阵的切片。两者展开后的数值与固定外部评测器的解码函数逐元素完全相同；再用这些真实切片完成 WINA → ARCQuant HiF4 投影的 CPU 前向，输出有限值。
- `tests/test_deepseek_adapter.py`：检查官方命名、FP8/FP4 解码、TEAL/WINA 掩码顺序、残差通道、`wo_a` 分组路径和缺失校准时拒绝运行。下面的命令可在装有 PyTorch 的环境重跑：

```bash
python -m unittest tests.test_deepseek_adapter -v
```

## 距离完整 DeepSeek 评测还差什么

1. 重新用正式规模（默认 128×512）校准并在目标机器复核每个投影的稀疏覆盖率；旧的 `calibration80` 只能作为 smoke 产物。
2. MATH-500 已有同一 native thinking + Math-Verify 的 WINA/TEAL 启动器；先完成单题 smoke，再按 500 题协议续跑。Terminal-Bench 还要接入终端代理和沙箱。
3. LongBench/LiveCodeBench 的短上下文 WINA smoke 已通过；正式任务仍需按完整上下文和题数运行，并记录显存、耗时与结果覆盖率。
4. 在显存足够时依次做小样本 GPU 前向、结果正确性、完整任务精度与耗时检查；当前实现仍是 dense GEMM，稀疏统计不能直接解释成端到端加速。

因此当前 DeepSeek 已有可复现的 GPQA 整模型方法结果和生成式 WINA 入口，但 MATH/Terminal-Bench 以及视频模型仍不是 ARCQuant 稀疏正式结果。
