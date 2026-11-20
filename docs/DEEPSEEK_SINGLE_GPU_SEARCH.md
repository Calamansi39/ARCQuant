# 只有一张空卡时，先运行 DS greedy 搜索

## 解决什么问题

原队列必须等两张空闲 H20，才能开始 30% 的搜索。2026-11-20 核查时，H20-2 只有 GPU 4 空闲，队列因此一直等待。

新增入口把 **TP2 的两个搜索进程放到同一张物理 GPU**。搜索只把当前层放到 GPU，其余权重保存在 CPU 内存里，所以不需要同时把完整模型塞进一张卡。

- 模型仍使用原有两个 TP 分片，逻辑上的 `world_size=2`。
- 原有 TEAL/WINA greedy、校准文本、ARC HiF4 W4A4、误差函数保持原配置。
- 两个进程之间的求和、WINA 在线检查所需的通道拼接，改为 GPU → CPU → Gloo 集合通信 → GPU，绕开 NCCL 不支持同卡双 rank 的限制。
- 这会增加传输开销。它用来利用零散空卡，不是推理加速实验；不能用它的耗时声称方法加速。
- 真正生成 GPQA/MATH 答案时，仍使用原有双卡 NCCL 评测入口。

不同通信后端可能有浮点舍入差异，不能预先保证搜索轨迹与双卡逐位相同。启动时会实际检验 BF16/FP32/FP64 求和、拼接和 rank 通信；原搜索还会检查非有限值和 0% 稀疏重放误差。新的执行方式及代码哈希记录在 `search/colocated_execution.json`，源码另有快照。

## 启动

在目标机器上激活原有 ARCQuant Python 环境，设置下面四个本机资源路径和运行目录，然后在仓库根目录执行：

```bash
python tools/queue_deepseek_probe.py \
  --run-dir "$RUN_DIR" \
  --base-eval-root "$BASE_EVAL_ROOT" \
  --source-model "$SOURCE_MODEL" \
  --converted-model "$TP2_MODEL" \
  --calibration-data "$CALIBRATION_PARQUET" \
  --sparsity 0.3 \
  --gpqa-limit 8 --math-limit 16
```

队列依次运行：

1. 等一张没有计算进程、可用显存至少 138000 MiB 的卡；
2. 两个进程共用该卡，逐层搜索 TEAL、WINA 的精确 30% 配置；128 条 × 512 token 校准，其中 20 条用于候选误差比较；
3. 校验两种方法覆盖模型全部层，并核对精确目标稀疏度；
4. 释放搜索卡，等两张满足空闲条件的卡；
5. 每种方法评测 GPQA 前 8 题、MATH 前 16 题，生成上限 8192 token，校验题数完整性。

这批 8/16 题是初步探针，不能直接和已有 32/100 题总分作公平比较。比较时应先取已有结果中相同的题目。

默认扫描 GPU 0–7。可通过 `--search-gpu-candidates 4` 限定搜索卡，通过 `--eval-gpu-candidates 4,5` 限定评测卡。实际需要约 170 GB 的 TP2 模型文件、充足的 CPU 内存，以及校准产物的磁盘空间。本次环境有约 1.7 TiB 可用 CPU 内存；每 rank 的 PyTorch 显存上限设为 60 GiB。其他型号 GPU 需要重新核实容量，当前阈值针对本机 143771 MiB 的 H20。

## 查看进度与恢复

- `status.json`：队列阶段、GPU 序号、进程号；等待时也每 20 秒更新心跳。
- `search/status.json`：实际校准/搜索阶段、当前层、候选次数。
- `search.log`：模型加载、通信校验及每步计算日志。
- `search/{teal,wina}/strategy.json`：逐步保存的策略，只有 `complete=true` 才能评测。
- `eval/`：四个方法与 benchmark 组合的原始逐题结果和汇总。

中断后使用相同命令、相同运行目录重启，搜索会按原生检查点恢复；原评测器也会验证已存在结果的协议再续跑。完整代码/配置不一致会拒绝恢复。队列锁防止同一个运行目录被两个新监督进程同时写入。

GPU 锁只协调采用该锁的 ARCQuant 队列，无法阻止其他用户启动任务。队列不会终止或挤占已有计算进程，也不会为了等待第二张卡而启动空转占显存程序。

## 2026-11-20 部署

- H20-2，GPU 4：DS 30% TEAL/WINA 联合逐层搜索。
- 运行目录：`runs/deepseek_30_tiny_probe_20261119/`。
- 新监督程序：`tools/queue_deepseek_probe.py`；替换了只等待双卡的旧 `wait_and_run.py` 进程。
- H20-2 GPU 0–3：原有 MAGI HiF4 TEAL greedy 继续运行。
- H20-1 当前没有空卡。
