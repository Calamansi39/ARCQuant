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

**当前实验目标是 DeepSeek V4 Flash、MiniMax H3 和 MAGI-2。** 换机器先按 [模型评测迁移指南](docs/MIGRATION_EVALUATION.md) 准备文件和环境，再用 `arc.py doctor` 查看逐项状态。[目标模型进度](docs/TARGET_MODELS.md) 区分权重、已有基线结果、ARCQuant HiF4 和 TEAL/WINA 的完成状态。DeepSeek 的投影级代码与未完成部分见 [适配进度](docs/DEEPSEEK_ADAPTER.md)。来源与固定版本保存在 [实验清单](configs/experiments.json)。

外行读法：`provision` 准备模型权重、题库、评分器和源码；`doctor` 逐项检查能否启动；`plan` 展示将执行的命令；`run` 才真正占用 GPU。大文件在 Git 忽略的 `assets/`，本机专用环境路径在 `external/local_env.sh`，新机器可按 [`configs/runtime.example.sh`](configs/runtime.example.sh) 填写。TEAL/WINA 是先离线确定稀疏规则、再在线稀疏投影输入；三个新目标目前只有**独立量化基线**可经统一 CLI 启动，目标模型的 ARCQuant + 稀疏整模型方法还在适配中。

```bash
python arc.py list
python arc.py assets --model deepseek-v4-flash --model minimax-h3 --model magi-2-preview \
  --benchmark gpqa-diamond --benchmark math500 --benchmark longbench-v2 \
  --benchmark livecodebench --benchmark vbench
python arc.py provision --model deepseek-v4-flash --benchmark gpqa-diamond
python arc.py doctor --model deepseek-v4-flash --benchmark gpqa-diamond --gpu-ids 0,1 --native-dry-run
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark gpqa-diamond \
  --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
```

想先看到真实模型里的激活分布，不必等待完整 benchmark，可以用两张卡跑一
个短 prompt 的诊断探针。它只保存统计量，不保存完整激活张量：

```bash
python arc.py probe --model deepseek-v4-flash \
  --base-eval-root /path/to/base_eval \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --gpu-ids 0,1 --output runs/deepseek_activation_probe.json \
  --quant hifp4 --max-length 64
```

在确认统计正常后，可直接让探针用同一批激活估计并应用一次稀疏掩码，查看
logits 变化：加上 `--method teal` 或 `--method wina --sparsity 0.5`。

探针会报告各投影的激活范围、零值比例，以及 TEAL/WINA 在几个目标稀疏率
下的实际保留率和输入能量损失。它用于校准和 debug，不能替代正式精度结果；
完整的实测记录见 [DeepSeek smoke run](docs/DEEPSEEK_SMOKE.md)。

`assets` 只检查文件，`plan` 只生成启动计划；两者都不运行 GPU。当前三个目标模型通过固定外部评测器提供 BF16/FP4 **基线**入口。DeepSeek 已有读取官方 FP8/FP4 权重并执行稀疏加 ARCQuant HiF4 的投影级代码，但目标模型的校准与整模型前向尚未接线；MiniMax H3 与 MAGI-2 尚无此类适配。仓库内已有的 Llama 稀疏代码不能作为目标模型完成的证据。方法来源见 [实现核查](docs/METHOD_PROVENANCE.md)，迁移依赖见 [资产指南](docs/ASSETS_AND_BASELINES.md)，卡时估计见 [GPU 预算](docs/GPU_BUDGET.md)。

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

