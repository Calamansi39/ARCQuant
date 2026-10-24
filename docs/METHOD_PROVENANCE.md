# TEAL / WINA 实现来源与可信范围

本页核对的是 `ARCQuant` 当前 `--method teal`、`--method wina` 的 **Llama 激活稀疏实现**。它们是在 ARCQuant 的量化层前加稀疏掩码的实验性组合；方法名表示采用了对应的选取规则，**不表示完整复现原论文的实验配置、速度或精度**。

## 代码从哪里来

| 当前选项 | 直接代码来源 | 对照的原仓库 | 当前实现 |
| --- | --- | --- | --- |
| `teal` | 本仓库 `feature/arc-teal` 分支的 `model/sparse_utils.py`，提交 `c90485f`；当前文件前 126 行与该分支相同 | [FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL)，核对提交 `fb7373c93ac3594817c9ee64d4e08b47430a1822` | 把原方法的直方图阈值规则接入 ARCQuant 的 Llama 投影 |
| `wina` | 本次整合重新编写在 `model/sparse_utils.py`；参考 [Calamansi39/wina](https://github.com/Calamansi39/wina) 提交 `04ac8794aa4ae345d09414733ceb76470d3b6a3e`，它是 [microsoft/wina](https://github.com/microsoft/wina) 的 fork | WINA 的 `utils/utils.py`、`wina/self_attn.py`、`wina/mlp.py` 与 `wina/model.py` | **均匀稀疏率的 top-k 变体**，不是原仓库复现实验脚本的完整配置 |

TEAL 官方实现对每层注意力输入和 MLP 输入分别收集 `h1`、`h2` 直方图。目标稀疏率为 `s` 时，用直方图的 `0.5 + s/2` 分位数作阈值，把绝对值低于阈值的激活置零。Q/K/V 共用注意力 `h1`，O 使用注意力 `h2`；gate/up 共用 MLP `h1`，down 使用 MLP `h2`。当前代码对应 [稀疏规则](../model/sparse_utils.py) 与 [Llama 投影接线](../model/qLlamaLayer.py)。TEAL 官方 `utils/utils.py` 的默认实现只稀疏预填充序列后半段，并在解码时处理当前 token；当前代码沿用这一选择。

WINA 官方代码用 `|激活值| × 权重输入列的 L2 范数` 给每个输入通道打分，再按 token 保留 top-k。其发布的 Llama 接线对 K、MLP gate 使用实际权重列范数，对 Q/V/O/up/down 使用 1；当前代码沿用这项具体设置。预填充阶段处理最后约 99% 的 token，解码阶段处理当前 token。`--disable-prefill-sparsity` 可关闭预填充稀疏。WINA 官方 README 明确允许跳过 greedy 分配，使用均匀稀疏率；当前实现属于这个变体。

## 哪些部分还不是原仓库的完整复现

- TEAL 官方可通过 `teal/grab_acts.py` 生成直方图，并支持逐投影的 greedy 稀疏率分配。当前 `arc.py prepare` **只生成 ARCQuant 重排与量化缓存**；它不会生成 TEAL 直方图。`teal` 运行必须另外提供与同一模型匹配的直方图，当前仅支持统一稀疏率。
- WINA 官方的复现实验脚本默认运行 `--greedy --transform`。当前 ARCQuant 既没有接入逐投影 greedy 稀疏率分配，也没有接入权重变换；因此当前结果不能称为 WINA 原论文实验的复现。
- 在 ARCQuant 中采用“先稀疏激活，再重排及 HiF4 量化”是接入方案。TEAL 的 ARCQuant 旧分支也采用这一顺序；TEAL 和 WINA 原仓库没有验证这个 **ARCQuant W4A4 组合**。已有 CPU 测试只覆盖小张量掩码和接口，尚无真实 Llama 模型的精度、逐层误差或延迟对照。
- 当前 `HiF4` 路径仍用密集矩阵乘法进行数值模拟。把激活置零并不意味着使用了 TEAL 的稀疏内核，也不能据此声称推理加速或节省显存。
- TEAL/WINA 的掩码是按激活幅值选通道，**不强制 2:4 结构**。本仓库的 `xmask` 才是另一条 2:4 掩码路径；它来自 ARCQuant 的既有工作，不能写成 TEAL 或 WINA 的原生实现。
- 当前两种方法仅接入旧 Llama 包装层；DeepSeek、MiniMax、MAGI 等新架构未接入。`base-eval` 后端也没有 TEAL/WINA。

因此，请把 CLI 中的 `teal` 和 `wina` 看作**有原仓库规则依据的 Llama + ARCQuant 原型**。在用于论文或组会结果前，需要固定同一模型、校准数据与评测协议，分别跑 `none` / `teal` / `wina`，再测实测零率、任务精度和端到端延迟。当前仓库的测试通过不等于这些指标已经通过验证。
