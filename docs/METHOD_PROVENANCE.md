# TEAL / WINA：原生规则、接线与可信范围

本页对照了 TEAL 官方仓库 [`FasterDecoding/TEAL`](https://github.com/FasterDecoding/TEAL) 的 `fb7373c93ac3594817c9ee64d4e08b47430a1822`，以及 WINA 的 [`Calamansi39/wina`](https://github.com/Calamansi39/wina) 提交 `04ac8794aa4ae345d09414733ceb76470d3b6a3e`（fork 自 [`microsoft/wina`](https://github.com/microsoft/wina)）。当前代码以 ARCQuant 的 Llama 量化层为宿主，实现两篇方法中可独立使用的原生选通规则。它们不等于论文全部实验设置。

## 方法怎么选输入

| 项目 | TEAL | WINA |
| --- | --- | --- |
| 每个输入的分数 | 本身的绝对值，与离线直方图阈值比较 | `abs(激活) × 对应权重输入列的 L2 范数` |
| 选多少 | 离线 `h1/h2` 直方图的 `0.5+s/2` 分位数给出阈值；实际零率会随当前数据变化 | 每个 token 对投影输入通道取 top-k，`k=int(通道数×(1-s))` |
| Llama 投影 | Q/K/V 共用注意力 `h1`；O 用注意力 `h2`；gate/up 共用 MLP `h1`；down 用 MLP `h2` | 官方 Llama 接线只对 K 和 gate 用实际列范数；Q/V/O/up/down 的范数设为 1 |
| 预填充 / 解码 | 默认只处理预填充后半段；解码处理当前 token | 默认处理预填充末尾约 99%；解码处理当前 token |
| 参考文件 | TEAL `utils/utils.py`、`teal/self_attn.py`、`teal/mlp.py` | WINA `utils/utils.py`、`wina/self_attn.py`、`wina/mlp.py` |

例如输入 `[1,2,3,4]`：TEAL 阈值为 `2.5` 时留下 `[0,0,3,4]`；WINA 保留两个通道、第一列范数为 `10` 且其它为 `1` 时，分数 `[10,2,3,4]`，留下 `[1,0,0,4]`。两者都是按当前激活选输入通道，**不保证 2:4 结构**。仓库已有 `xmask` 是另一条 2:4 路径。

## 本仓库离线和在线分别做什么

1. `arc.py prepare` 是 ARCQuant 原有工作：读取模型和校准数据，生成通道重排索引、残差通道数、激活统计到 `saved/`。`none`、`teal`、`wina` 走同一套 HiF4 准备过程。
2. `arc.py calibrate --method teal` 逐层读取校准文本，收集**有符号**的注意力与 MLP `h1/h2` 激活，生成每层的 `histograms.pt`。这一步沿用 TEAL 的直方图布局和阈值定义。`wina --allocation uniform` 不需要这一步；加载层时从原权重计算 K/gate 列范数即可。
3. 两种方法均可用 `--allocation greedy`。校准器对每层 Q/K/V/O/gate/up/down 逐一试增加稀疏率，选使该层输出与未稀疏输出差别最小的投影，直到按七个权重矩阵参数量加权的预算达到目标值。每次候选消耗相同的权重预算：以 Q 投影的参数量为单位，参考默认步长为 `0.025`，其它矩阵的稀疏率增量按参数量反比换算。误差使用参考仓库的层输出后段 token 范数。生成的 `strategy.json` 记录每层七个稀疏率、校准设置和模型配置哈希；`run` 会核对。这里是基于参考 greedy 思路的重新实现，样本数与步长可由 CLI 控制，不应把它描述为逐字节复制官方 CSV 曲线。
4. `arc.py run` 在线运行时，先对每个投影输入应用 TEAL 阈值或 WINA top-k，再走 ARCQuant 的通道重排、HiF4 权重/激活数值模拟及投影计算。`model/sparse_utils.py` 实现掩码，`model/qLlamaLayer.py` 把七个掩码接到对应输入；`model/calibrate_sparse.py` 实现离线产物。

TEAL 的早期 `model/sparse_utils.py` 前半部分来自本仓库曾有的 `feature/arc-teal` 分支，已对照官方规则核查；WINA 在本次整合按原仓库的 Llama top-k 接线重新编写。没有直接复制 WINA 仓库的大段源码。

## 未覆盖的论文配置和验证

- WINA 官方复现脚本使用 `--greedy --transform`。当前已经有 `greedy` 分配，但**没有**移植其可选的全模型张量变换。这个变换涉及 LayerNorm 融合、残差旋转和投影权重变换，必须与 ARCQuant 重排及量化缓存一起校准；不能只改一个分数开关。因此当前结果是 WINA 原生 top-k 规则与可选 greedy 分配在**未变换模型**上的组合，不能称为官方完整复现。
- TEAL/WINA 原仓库没有验证与 ARCQuant 增强残差通道的 HiF4 W4A4 组合。`HiF4` 路径目前仍主要是密集矩阵乘法的数值模拟；出现零值不意味着实际稀疏内核已跳过乘法，也不能据此宣称加速。
- 锁定 `transformers==4.46.3` 的小型 Llama CPU 检查已覆盖离线四组激活、官方层前向一致性、greedy 预算、TEAL 文件读取、WINA 零稀疏率下的量化输出一致性。还没有在真实 8B 模型上完成精度、逐层误差和端到端延迟回归。迁移后应先用 `smoke` 跑通，再用相同模型/评测协议分别测 `none`、`teal`、`wina`。
- 当前仅 Llama 有 TEAL/WINA 的 ARCQuant 投影接线。Qwen、Mixtral 保留原量化路径，DeepSeek 和视频模型只登记了资产及外部基线；`base-eval` 不实现 ARCQuant 稀疏方法。
