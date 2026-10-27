# 换机器后怎样准备模型评测

这份说明面向刚接触本仓库的同学。先记住三个不同的状态：

1. **文件准备好**：模型、数据、转换分片、评测器源码在本机能找到。
2. **基线可运行**：指定的 Python 环境、评分权重和空闲 GPU 也齐全；可以评 BF16 或普通 W4A4 HiF4 基线。
3. **我们的方法可运行**：还需 ARCQuant 增强残差通道加 TEAL/WINA 的目标模型校准和完整前向接线。当前三个目标模型都**未达到第 3 步**，不能把基线分数称作我们的方法分数。

仓库保存来源、固定版本、入口、检查规则和方法代码。几百 GB 的权重、视频和运行结果不放进 GitHub。换机器后需要重新下载，或把 `assets/` 整体迁走；这台机器的 `assets/` 多数是指向别处的软链接，**不能只复制链接本身**。

## 第一步：克隆并指定资产位置

```bash
git clone git@github.com:Calamansi39/ARCQuant.git
cd ARCQuant
# 建议用容量足够的本地或共享磁盘；不设置时默认用仓库内 assets/
export ARCQUANT_ASSET_ROOT=/large-disk/arcquant-assets
# 官方 Hugging Face 在当前网络不可达时，可选用可访问的镜像
# export HF_ENDPOINT=https://hf-mirror.com
python arc.py list
```

控制 CLI 的 `list`、`provision`、`doctor` 用 Python 标准库即可；`provision --execute` 需要下载模型时，控制 CLI 的 Python 还需安装 `huggingface-hub`，DeepSeek 权重转换还需要评测环境的 PyTorch 与 `safetensors`。原仓库 Llama 路径需另装根目录 `requirements.txt`，它固定 `transformers==4.46.3`；**不要把它和目标模型生成环境混装**。`provision` 准备文件与链接，Python/CUDA 环境需按第三步安装。

目标模型及数据来源固定在 [`configs/experiments.json`](../configs/experiments.json)。三个目标模型的关键配置/索引 SHA256 已与本机资产及登记的公开提交比对；五个文件型 benchmark 还记录了数据文件 SHA256。DeepSeek 固定到 `7872f01...`。巨大权重分片只检查索引、存在性、文件头和少数独立文件大小，**未逐字节校验全部权重**。部分模型需要 Hugging Face 账号取得访问许可。模型名指当前实验登记版本，换版本必须重新核对权重结构、转换和评测协议。

## 第二步：按模型准备文件

先看计划，再执行。`provision` 会依次检查并准备模型、数据、固定版本评测器、必要的两卡转换分片，以及视频评分权重；已有且通过检查的文件会跳过。DeepSeek 转换会额外检查源文件 SHA256、两卡分片大小和完成标记。视频评分权重执行时按清单逐文件验证哈希。

```bash
# DeepSeek + GPQA Diamond；把 benchmark 改为 math500、longbench-v2、livecodebench 也一样
python arc.py provision --model deepseek-v4-flash --benchmark gpqa-diamond
python arc.py provision --model deepseek-v4-flash --benchmark gpqa-diamond --execute

# MiniMax H3 / MAGI-2：VBench 提示词、21 个模型/BERT 文件和 ViCLIP 分词文件
python arc.py provision --model minimax-h3 --benchmark vbench --execute
python arc.py provision --model magi-2-preview --benchmark vbench --execute
```

**评测器访问限制：** 当前登记的 `xiaodiandian-AI/base_eval` GitHub 地址在这台机器没有验证通过。本机已有固定提交的可用副本，但仓库的公开克隆不能带走它。新机器需要该仓库的读取权限，或获得授权的完整 Git checkout，然后运行：

```bash
python arc.py provision --model deepseek-v4-flash --benchmark gpqa-diamond \
  --backend-source /path/to/authorized/base_eval --execute
```

`--backend-source` 也可传有权限的 Git URL；它不会改变清单要求的提交号。**未取得评测器权限时，这项依赖无法仅靠克隆 ARCQuant 自动补齐。** MAGI-2 的官方运行源码是另一项公开仓库，`provision` 会按清单固定提交检出。

只下载 VBench 评分资产也可用 `python arc.py fetch --scorer vbench`。它包含 21 个模型/BERT 文件和 ViCLIP 分词文件；下载后按清单中的 SHA256 或 Git blob 哈希逐一核验，缺失的分词文件也会提前补齐，避免评分环境导入时向 GitHub 临时下载。`fetch --benchmark vbench` 只准备官方提示词和评分源码。

## 第三步：给三个目标准备独立运行环境

不同模型不能假定共用一个 Python。下面的变量指向新机器上**真实存在的**环境；保留 `venv/bin/python` 原路径，不要用 `readlink -f` 改成底层解释器，否则会失去虚拟环境的包。

可复制 [`configs/runtime.example.sh`](../configs/runtime.example.sh) 为 Git 忽略的 `external/local_env.sh`，填写本机路径，然后在仓库根目录运行 `source external/local_env.sh`。本机已放好一份对应现有环境的 `external/local_env.sh`；它不会上传到 GitHub。若默认使用仓库内 `assets/`，删除模板里的 `ARCQUANT_ASSET_ROOT` 一行即可。

| 目标 | 必需环境与来源 | 运行资源预检下限 |
| --- | --- | --- |
| DeepSeek V4 Flash | `ARCQUANT_BASE_EVAL_ENV=/path/to/deepseek-env`；安装固定评测器的 `requirements-deepseek-v4.txt`，并按 GPU 驱动安装兼容 PyTorch/Triton | 两张卡，每张空闲至少 132 GiB（135168 MiB） |
| MiniMax H3 | `ARCQUANT_H3_PYTHON=/path/to/h3-env/bin/python`；`ARCQUANT_VBENCH_PYTHON=/path/to/vbench-env/bin/python`；按固定评测器的 `requirements-minimax-h3.txt`、`requirements-vbench-a800.txt` 及 `scripts/setup_minimax_h3.sh` 准备 | 至少一张卡空闲 79000 MiB；评分还需 16000 MiB |
| MAGI-2 | `ARCQUANT_MAGI2_PYTHON=/path/to/magi2-env/bin/python`；同一 VBench 评分环境；官方 Hopper 依赖见 `scripts/setup_magi2.sh`；安装 FFmpeg 并使 `ffprobe` 可执行 | 两张卡，每张空闲至少 142000 MiB；大量系统内存 |

这些显存值是本仓库预检阈值，不是质量或速度保证。视频生成是多阶段、长步数计算；`tiny` 也需要每种精度约 740 个视频，不等于一分钟冒烟测试。完整耗时估算见 [GPU 预算](GPU_BUDGET.md)。

MAGI-2 的 `ffprobe` 若不在 `PATH`，可设置 `ARCQUANT_FFMPEG_BIN=/path/to/ffmpeg/bin`。在 `ARCQUANT_BASE_EVAL_ENV` 已指定时，CLI 也会从该环境传递 CPU 预检解释器。DeepSeek 的 LongBench/LiveCodeBench 启动器使用仓库内的兼容包装，核对外部脚本哈希后仅在内存里替换旧的 conda 激活语句，不改评测协议。不同 CUDA 架构需要重建视频评分扩展；只复制虚拟环境目录不够。

## 第四步：运行“医生”检查

`doctor` 只读，不占 GPU。它输出 `ready`、`blocked` 或 `unchecked`，每个阻塞项给出修复提示。默认会检查模型/数据文件、评测器固定提交、DeepSeek 转换签名、Python 模块、固定评测器的视频环境导入、视频评分资产大小和空闲显存。视频分数发布前加 `--hash-scorer`，逐文件验完全部 9.4 GB 评分权重及 ViCLIP 分词文件；跨机器迁移前加 `--check-remote` 验证评测器远端是否有权限。

加 `--native-dry-run` 可进一步用固定评测器做 CPU-only 计划检查：DeepSeek 核对真实权重布局和两卡转换版；MiniMax H3/MAGI-2 核对模型布局及 VBench tiny 的 108 条提示词、740 条视频计划。这个选项仍不加载整模型权重或占用 GPU。`--skip-gpu` 时最终状态会是 `unchecked`，退出码为 2，表示仍需在开跑前检查空闲卡。

```bash
python arc.py doctor --model deepseek-v4-flash --benchmark gpqa-diamond --gpu-ids 0,1
python arc.py doctor --model minimax-h3 --benchmark vbench --hash-scorer
python arc.py doctor --model magi-2-preview --benchmark vbench --gpu-ids 0,1 --hash-scorer

# 换机器后先做更深的 CPU 检查
python arc.py doctor --model minimax-h3 --benchmark vbench --native-dry-run --skip-gpu

# 新机器暂时没有空闲 GPU 时，检查其余项目；报告会保留 GPU=unchecked
python arc.py doctor --model deepseek-v4-flash --benchmark gpqa-diamond --skip-gpu
```

`doctor --backend arcquant --method teal/wina` 对三个目标仍会明确阻止；这表示目标方法接线尚未完成，不是权重缺失。三个目标的具体进度见 [目标模型状态](TARGET_MODELS.md)，DeepSeek 投影级代码与测试见 [适配说明](DEEPSEEK_ADAPTER.md)。

## 第五步：先跑同协议基线

用 `plan` 先看将要执行的命令，再用 `run` 真正占用 GPU。结果和元数据写入被 Git 忽略的 `runs/` 或外部评测器的结果目录。精度比较要固定模型、数据版本、采样参数和评分协议。

```bash
# 文本：少量题目先验启动，再评完整集
python arc.py plan --backend base-eval --model deepseek-v4-flash \
  --benchmark gpqa-diamond --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
python arc.py run --backend base-eval --model deepseek-v4-flash \
  --benchmark gpqa-diamond --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1

# 视频：tiny 是 16 维子集，不可当官方全集成绩
python arc.py plan --backend base-eval --model minimax-h3 \
  --benchmark vbench --profile tiny --method none --quant-type HiF4 --gpu-ids auto
```

已登记的 Terminal-Bench 包含 86 个任务，但终端代理、容器及评分协议尚未整合，`doctor` 会阻止把它当作当前可运行评测。LongBench v2 和 LiveCodeBench 走各自的生成式协议；外部 `base-eval` 的 HiF4 是普通 W4A4 假量化，**不包含 ARCQuant 增强残差通道，也不包含 TEAL/WINA**。这三个目标方法仍需要模型级校准、推理接线和 GPU 回归，不能通过切换 `--method` 绕过。
