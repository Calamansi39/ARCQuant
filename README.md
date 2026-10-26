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
conda create -n arcquant python=3.11 -y
conda activate arcquant
```
For the original native NVFP4 kernels, use the matching [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) toolchain. For the HiF4 PyTorch evaluation path, install a PyTorch build compatible with your GPU driver; see the [Chinese quickstart](docs/QUICKSTART.md).
```bash
git clone --recurse-submodules https://github.com/Calamansi39/ARCQuant.git
cd ARCQuant
# Install the PyTorch build matching your CUDA driver first; see docs/QUICKSTART.md.
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

## 中文快速指南：方法整合与迁移

**当前实验目标是 DeepSeek V4 Flash、MiniMax H3 和 MAGI-2。** 先读 [目标模型进度](docs/TARGET_MODELS.md)：它逐项区分权重、已有基线结果、ARCQuant HiF4 和 TEAL/WINA 的完成状态。模型与 benchmark 的来源保存在 [实验清单](configs/experiments.json)，[从零开始的指南](docs/QUICKSTART.md) 说明资产准备和 CLI 用法。

```bash
python arc.py list
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark vbench
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark gpqa-diamond \
  --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
```

`assets` 只检查文件，`plan` 只生成启动计划；两者都不运行 GPU。当前三个目标模型通过固定外部评测器提供 BF16/FP4 **基线**入口；ARCQuant 残差通道与 TEAL/WINA 尚未接到它们。仓库内已有的 Llama 稀疏代码是方法原型，不能作为目标模型完成的证据。方法来源见 [实现核查](docs/METHOD_PROVENANCE.md)，迁移依赖见 [资产指南](docs/ASSETS_AND_BASELINES.md)，卡时估计见 [GPU 预算](docs/GPU_BUDGET.md)。

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

