# 模型、数据和外部基线

## 迁移时复制什么

在新机器只需克隆 Git 仓库，按 [快速指南](QUICKSTART.md) 安装依赖，再用 `python arc.py fetch ...` 下载所需模型和 benchmark。来源、版本、预期数据条数及运行状态统一记录在 [`configs/experiments.json`](../configs/experiments.json)。`assets/`、`saved/`、`runs/` 和 `external/` 被 Git 忽略；GitHub 不存储数十到数百 GB 的权重或视频。

默认 `assets/` 放在仓库内；设置 `ARCQUANT_ASSET_ROOT` 后可放到任何大磁盘。Hugging Face 下载使用 `<资产根>/hf_home/hub` 缓存，`models/` 与 `datasets/` 中建立相对链接，不重复保存一份巨型权重。仓库原有的 ARCQuant 校准缓存保存在 `saved/`；TEAL/WINA 离线产物保存在 `<资产根>/calibration/`。如果直接复制整套资产目录到另一台机器，保留内部相对路径即可。

```bash
python arc.py list
python arc.py fetch --model llama-3.1-8b
python arc.py fetch --benchmark wikitext2
python arc.py fetch --benchmark gpqa-diamond
python arc.py fetch --benchmark math500
python arc.py fetch --benchmark longbench-v2
python arc.py fetch --benchmark livecodebench
python arc.py fetch --benchmark terminal-bench
python arc.py fetch --benchmark vbench
python arc.py assets
```

`fetch` 一次下载一个对象，避免误拉取几个大型模型。MiniMax H3 默认只取文本生成视频（T2VA）所需组件；不包含参考视频生成的大型 `transformer_ref` 等额外权重。部分模型需要 Hugging Face 访问许可和 token。对于已有权重，用 `--model-path` 或 `ARCQUANT_LLAMA_MODEL`、`ARCQUANT_DEEPSEEK_MODEL` 等清单中的环境变量指定即可。`python arc.py assets` 只检查本机文件、索引分片和已登记数据条数；它不等于模型精度或运行环境验证。

## 外部生成评测器

DeepSeek V4 Flash、MiniMax H3、MAGI-2 的当前入口是固定 Git 提交的 [`base_eval`](https://github.com/xiaodiandian-AI/base_eval)。先执行：

```bash
python arc.py fetch --backend base-eval
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark gpqa-diamond --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
```

**访问限制（2026-10-25）：**清单中的 `xiaodiandian-AI/base_eval` 地址，用当前 HTTPS 和 SSH 身份均返回仓库不存在或无权限。新机器必须先获得该仓库读取权限，或取得有授权的完整 Git checkout / bundle；否则 `fetch --backend base-eval` 会失败。已有授权副本时可运行 `python arc.py fetch --backend base-eval --source /path/to/base_eval`，仍会检出清单固定的 SHA。这个外部依赖未上传到本仓库，不能把视频和 DeepSeek 基线描述为仅克隆 ARCQuant 即可运行。

下载会固定到清单中的 SHA，放入 `external/base_eval`，并把 `external/data` 链接到资产根目录，以适配评测器的相邻目录约定。`run` 会检查该 SHA、启动脚本、权重和数据，不会误用另一版评测协议。外部评测器有自己的 Python 和 GPU 依赖，不包含在 ARCQuant 的 `requirements.txt` 中；在新服务器还需按其 `requirements-deepseek-v4.txt`、`requirements-minimax-h3.txt`、`requirements-vbench-a800.txt` 及 MAGI-2 官方运行说明准备环境。视频环境的可执行文件可分别通过 `ARCQUANT_H3_PYTHON`、`ARCQUANT_VBENCH_PYTHON`、`ARCQUANT_MAGI2_PYTHON` 指定。VBench 的评分模型来源、SHA256 和大小也保存为 [`VBench 评分资产清单`](../configs/vbench_download_manifest.json) 与 [`BERT 清单`](../configs/vbench_bert_download_manifest.json)，来源为本机外部评测器的固定清单；可用其 `preload_vbench_assets.py --full` 下载和校验。`fetch --benchmark vbench` 只下载 VBench 源码与提示词。

DeepSeek 两卡推理先进行权重转换。外部评测器提供转换程序；确保当前 shell 已激活它所需的运行环境，然后从本仓库根目录执行：

```bash
python external/base_eval/convert_deepseek_v4.py --world-size 2 \
  --source-model assets/models/deepseek-v4-flash \
  --output assets/models/deepseek-v4-flash-tp2 --plan
# 检查计划后去掉 --plan，执行实际转换
```

若设置了 `ARCQUANT_ASSET_ROOT`，把以上 `assets/` 改为该根目录。运行时用 `--converted-model` 或 `ARCQUANT_DEEPSEEK_TP2` 指向转换结果。两卡默认要求每卡约 132 GiB 可用显存；不同 GPU 要先评估显存和吞吐。VBench `tiny` 虽只覆盖 108 个唯一提示词，按官方维度规则仍生成约 740 条视频/精度；见 [GPU 预算](GPU_BUDGET.md)。

```bash
python arc.py plan --backend base-eval --model deepseek-v4-flash --benchmark livecodebench --profile smoke --method none --quant-type HiF4 --gpu-ids 0,1
python arc.py plan --backend base-eval --model minimax-h3 --benchmark vbench --profile tiny --method none --quant-type HiF4 --gpu-ids auto
python arc.py plan --backend base-eval --model magi-2-preview --benchmark vbench --profile tiny --method none --quant-type HiF4 --gpu-ids 0,1
```

将 `plan` 改为 `run` 后才真正评测。`base-eval` 的 HiF4 是独立 W4A4 基线，不含 ARCQuant 的增强残差通道，更不含 TEAL/WINA。LongBench v2 与 LiveCodeBench 走生成和专用评分协议；GPQA Diamond 与 MATH-500 在该后端也走生成式协议。ARCQuant Llama 的 `math500` / `gpqa-diamond` 入口则使用 lm-eval 任务，两者不可直接当成同一评测协议比较。

## 2026-10-24 本机核对记录

这台服务器曾检查过 DeepSeek V4 Flash 原权重的 48 个分片及两卡转换的两片、MiniMax H3 的 T2VA 所需组件、MAGI-2 preview 的 89 个文件；相应索引分片与已知文件大小匹配。GPQA Diamond 198 题、MATH-500 500 题、LongBench v2 503 题、LiveCodeBench v6 175 题、Terminal-Bench core 0.1.1 的 86 个任务、VBench 的 944 条唯一英文提示词也通过了本机数据检查。这些是**当时的本机记录**，迁移后仍应重新运行 `python arc.py assets`。文件检查不包含数百 GB 权重的逐字节哈希，也不证明新 GPU 上的精度、速度或兼容性。

当前机器已有资产可通过环境变量复用，例如 `ARCQUANT_DEEPSEEK_MODEL=/mnt/kpfs_qyang/djs/models/DeepSeek-V4-Flash-0731`、`ARCQUANT_MINIMAX_MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MiniMax-H3`、`ARCQUANT_MAGI2_MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview`。这些机器专属路径不写入清单默认值，因此在新机器克隆后不会误指向旧服务器。
