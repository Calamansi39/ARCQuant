# ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs

<h5 align="center">

[![arXiv](https://img.shields.io/badge/ARCQuant-2601.07475-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2601.07475)
 <br>

</h5>

![arcquant](kernels/ARCQuant_Visualize.png)

**ARCQuant** is a high-performance quantization framework designed to resolve the conflict between accuracy and inference efficiency in low-bit LLMs.

While fine-grained quantization (e.g., Block-wise/NVFP4) effectively isolates quantization noise, activation outliers still degrade performance in critical channels. Traditional mixed-precision methods address this by splitting computations into separate branches, which introduces significant kernel launch overhead and memory fragmentation.

ARCQuant takes a different approach. Instead of treating outliers separately, we leverage the structural sparsity of quantization errors in fine-grained settings. We capture the quantization residuals of these critical channels and fuse them back into the computation as **Augmented Residual Channels (ARC)**.

To do: 

- [x] Release arxiv version of [ARCQuant](https://arxiv.org/abs/2601.07475).
- [x] Release code for reproducing results.
- [x] Release CUDA kernels on NVFP4.
- [x] Release calibration and preprocessing scripts.
- [x] Support [vLLM](https://github.com/vllm-project/vllm) integration.
- [ ] **Model Support**: Add support for more model families:
    - [ ] Qwen3
    - [x] Mixtral
    - [ ] Wan2.2


## 1. Installation
```bash
conda create -n arcquant python=3.10 -y
conda activate arcquant
```
Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is in your environment.
```bash
git clone --recurse-submodules https://github.com/actypedef/ARCQuant.git
cd ARCQuant
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Building Kernels
```bash
sudo apt-get update
sudo apt-get install python3-dev
```
```bash
conda install pybind11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
cd kernels/
bash remake.sh
```

This might take a few minutes.

### 2.2 Preprocessing

Reorder_indices, select_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 128 --seqlen 2048 --act_sort_metric max
```
Results are saved in ./saved/

### 2.3 Accuracy Evaluation

ARCQuant supports multiple formats, including NVFP4, MXFP4, HiF4, and INT4. You can modify the `quant_type` parameter if needed.
```bash
bash evaluate.sh /PATH/TO/YOUR/MODEL/
```

## 3. Efficiency Evaluation

FlashInfer:
```bash
cd third-party/flashinfer
python -m pip install -v .
```
We will release our vLLM evaluation very soon.

## TEAL / WINA 统一实验入口（中文）

本仓库的整合版本以学姐维护的 ARCQuant 代码为基准，同时合并你自己仓库已有的预处理提交。它保留原有的低比特量化、`x_mask` 和联合校准代码，加入 Llama 模型的 TEAL 与 WINA 激活稀疏选项。**量化**是把权重和激活表示成低精度数值；**激活稀疏**是把一部分运行时数值置零。两者可以按“稀疏化 → ARCQuant 量化 → 评测”的顺序组合。在这里选择 `HiF4` 就是 W4A4 数值模拟路径，不代表已经生成压缩后的 4-bit 权重文件，也不自动代表推理更快。

从仓库根目录开始：

```bash
# 用已有的 ARCQuant 预处理脚本生成 saved/ 中的重排与校准文件
python arc.py prepare --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B

# 只查看实验计划；不会加载权重或占用 GPU
python arc.py plan --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B \
  --method wina --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke

# 真正运行；实验记录写入 runs/
python arc.py run --model llama-3.1-8b --model-path /path/to/Llama-3.1-8B \
  --method wina --sparsity 0.5 --quant-type HiF4 --benchmark math500 --profile smoke
```

`--method none` 运行 ARCQuant 原有量化路径；`--method teal` 还要提供 `--histogram-path /path/to/histograms`；`--method xmask` 调用学姐已有的 2:4 掩码路径，需要 `--x-mask-ckpt`。目前 TEAL/WINA 和统一 CLI 的 xmask 只接入 **Llama** 适配器，TEAL/WINA 不能与 `x_mask` 同时启用。`python arc.py list` 可查看模型与评测的真实接入状态。GPQA Diamond 和 MATH-500 已映射到当前 `lm-evaluation-harness` 的任务；DeepSeek V4 Flash、MiniMax H3、MAGI-2 及生成式评测仍需对应模型或评分适配器，登记路径不等于可以运行。

更完整的命令、实现说明、权重与 benchmark 配置见 [实验指南](docs/EXPERIMENTS.md)；组会卡时粗估见 [GPU 预算](docs/GPU_BUDGET.md)。

## 4. Citation
```
@article{meng2026arcquant,
  title={ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs},
  author={Meng, Haoqian and Luo, Yilun and Zhao, Yafei and Liu, Wenyuan and Zhang, Peng and Ma, Xindian},
  journal={arXiv preprint arXiv:2601.07475},
  year={2026}
}
```

## 5. Acknowledagement
Our code is built on the following repos, thank you for your contributions to community:
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

