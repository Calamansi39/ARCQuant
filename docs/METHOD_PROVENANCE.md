# TEAL / WINA：原生规则、接线与可信范围

> **2026-11-05 晚间更新：** DeepSeek 已接入原生规则的 greedy 预算搜索和 MATH/GPQA
> 生成式 ARC 入口；正式 43 层策略仍在搜索，完整方法成绩尚未完成。WINA transform 未执行。
> DeepSeek 的具体投影映射、原生算法保留项和工程差异见 [greedy 说明](DEEPSEEK_GREEDY.md)。
> 下文 Llama 的 greedy/transform 是独立旧模型路径，不能当作 DeepSeek 已完成的证据。

本页对照了 TEAL 官方仓库 [`FasterDecoding/TEAL`](https://github.com/FasterDecoding/TEAL) 的 `fb7373c93ac3594817c9ee64d4e08b47430a1822`，以及 WINA 的 [`Calamansi39/wina`](https://github.com/Calamansi39/wina) 提交 `04ac8794aa4ae345d09414733ceb76470d3b6a3e`（fork 自 [`microsoft/wina`](https://github.com/microsoft/wina)）。下文说明以 ARCQuant 的 Llama 量化层为宿主的旧接线，实现两篇方法中可独立使用的原生选通规则。DeepSeek 使用单独的 MLA/HC/MoE adapter；两者都不等于论文全部实验设置。

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
3. **旧 Llama adapter** 中两种方法均可用 `--allocation greedy`。DeepSeek 使用另外的专用入口，见页首链接。校准器对每层 Q/K/V/O/gate/up/down 逐一试增加稀疏率，选使该层输出与未稀疏输出差别最小的投影，直到按七个权重矩阵参数量加权的预算达到目标值。每次候选消耗相同的权重预算：以 Q 投影的参数量为单位，参考默认步长为 `0.025`，其它矩阵的稀疏率增量按参数量反比换算。误差使用参考仓库的层输出后段 token 范数。生成的 `strategy.json` 记录每层七个稀疏率、校准设置和模型配置哈希；`run` 会核对。这是基于参考 greedy 思路的重实现；本地 WINA 搜索直接使用 top-k，原版固定源码用 threshold 搜索、top-k 评测，不能把两条流程称为完全相同。
4. `arc.py run` 在线运行时，先对每个投影输入应用 TEAL 阈值或 WINA top-k，再走 ARCQuant 的通道重排、HiF4 权重/激活数值模拟及投影计算。`model/sparse_utils.py` 实现掩码，`model/qLlamaLayer.py` 把七个掩码接到对应输入；`model/calibrate_sparse.py` 实现离线产物。

本次服务器上的逐行核查使用了保存的源码快照：

```text
/mnt/kpfs_qyang/zhaolianchao/work/teal-source  fb7373c93ac3594817c9ee64d4e08b47430a1822
/mnt/kpfs_qyang/zhaolianchao/work/wina-source  04ac8794aa4ae345d09414733ceb76470d3b6a3e
```

核查结论：ARCQuant 的 `ThresholdSparsifier` 与 TEAL 的 `SparsifyFn` 在有符号阈值、`0.5+s/2` 分位数、h1/h2 投影共享、prefill 尾段和 decode 选通上等价；`WeightInformedTopKSparsifier` 与 WINA 的 Llama 接线在 K/gate 使用输入列 L2 范数、其它投影使用单位范数、`abs(x)*norm` 后逐 token top-k 上等价。ARCQuant 只增加了显式的 decode-only 开关、统计计数和 HiF4 输入接口。

WINA 的 `--transform` 现在由 `model/wina_transform.py` 提供：LayerNorm/RMSNorm 融合、首层 K 的 SVD 起始方向、逐层 gate SVD、投影左右旋转以及两个残差坐标矩阵都已接入 `QLlamaDecoderLayer`。这是可选路径；必须先用 `reorder_indices.py --wina_transform` 生成带 `_wina_transform` 后缀的 ARC 缓存，再用 `arc.py run --method wina --wina-transform` 评测，不能混用变换前缓存。

## 未覆盖的论文配置和验证

- WINA 官方复现脚本使用 `--greedy --transform`。变换代码已移植，但它仍需要在目标机器上重新生成对应 ARC 缓存；未带 `_wina_transform` 后缀的旧缓存不能用于变换模型。只有同时记录 transform manifest、缓存哈希和评测结果，才可称为完整 WINA+ARCQuant 实验。
- TEAL/WINA 原仓库没有验证与 ARCQuant 增强残差通道的 HiF4 W4A4 组合。`HiF4` 路径目前仍主要是密集矩阵乘法的数值模拟；出现零值不意味着实际稀疏内核已跳过乘法，也不能据此宣称加速。
- 锁定 `transformers==4.46.3` 的小型 Llama CPU 检查已覆盖离线四组激活、官方层前向一致性、greedy 预算、TEAL 文件读取、WINA 零稀疏率下的量化输出一致性。还没有在真实 8B 模型上完成精度、逐层误差和端到端延迟回归。迁移后应先用 `smoke` 跑通，再用相同模型/评测协议分别测 `none`、`teal`、`wina`。
- 旧通用 greedy/transform 路径限制在 Llama adapter。DeepSeek 专用 greedy 与生成式 ARC 补偿已实现；当前 128×512 校准、20 段误差样本的正式策略搜索尚未完成，不能将其描述为已有完整可用校准或 benchmark 成绩。DeepSeek transform 尚未接入。视频模型的适配进展另见各模型交接文档。
