# DeepSeek V4 Flash：当前完整 GPQA 结果

本页记录仓库当前实现跑出的第一组完整对照结果。模型是
`DeepSeek-V4-Flash-0731` 官方 TP2 权重，数据是本仓库固定的 GPQA Diamond
198 题，校准数据是 WikiText-2 train 的同一份离线校准目录。

| 变体 | 量化/稀疏配置 | 正确题数 | 准确率 | 相对 BF16 |
| --- | --- | ---: | ---: | ---: |
| BF16 | BF16 基线 | 99 / 198 | 50.00% | — |
| HiF4 | HiF4 W4A4 | 92 / 198 | 46.46% | -3.54 个百分点 |
| ARC | HiF4 W4A4 + ARC | 102 / 198 | 51.52% | +1.52 个百分点 |
| TEAL | HiF4 W4A4 + ARC + TEAL，稀疏率 50% | 87 / 198 | 43.94% | -6.06 个百分点 |
| WINA | HiF4 W4A4 + ARC + WINA，稀疏率 50% | 81 / 198 | 40.91% | -9.09 个百分点 |

## 这组数字说明什么

- 当前“正式结果未完成”不是因为没有 W4A4。HiF4 W4A4、ARC、TEAL 和 WINA
  已经在同一个 DeepSeek 整模型入口中完成了 198 题统一评测。
- ARC 在这组 GPQA 选项似然协议下恢复了 HiF4 的精度，并略高于 BF16 基线；
  这说明 ARC 的校准、MLA、MoE 和 TP row-parallel 接线已经能工作。
- TEAL/WINA 的 50% 输入掩码会带来明显精度损失。这是需要继续调稀疏率、校准
  数据和方法细节的实验结论，不应把结果解释成“稀疏加速已经验证”。

## 评测协议和边界

评测使用 `gpqa_diamond_zeroshot` 风格的四选项条件似然：模型分别计算四个
选项的 log-likelihood，最高者作为答案。它不是 thinking 生成式 Pass@1，不能
直接和采用生成式协议的历史数字比较。

当前实现的矩阵乘法是 dense PyTorch GEMM，结果用于精度对照；没有据此报告真实
端到端稀疏速度。当前 WINA 接线使用输入列重要性进行 top-k 掩码，尚未加入 WINA
论文中可选的全模型 `transform` 变换，因此这组数字应标为 **WINA mask-only**。

原始结果文件只保存在运行机器的 `runs/`（该目录被 Git 忽略），每个 JSON 包含
每题预测、题目哈希、数据集哈希、权重和代码签名、覆盖率及显存记录。复现实验
命令和参数见 [DeepSeek 实验说明](DEEPSEEK_EXPERIMENT.md)。

## 当前覆盖范围

这页是 DeepSeek × GPQA Diamond 的第一组完整结果，不代表 MiniMax H3、MAGI-2
或 MATH-500、LongBench、LiveCodeBench、Terminal-Bench、VBench 已经完成。下一步
应在相同 CLI 和元数据协议下逐项接入这些模型与 benchmark，并先补齐 WINA
transform 和真正的 sparse kernel，再报告速度指标。
