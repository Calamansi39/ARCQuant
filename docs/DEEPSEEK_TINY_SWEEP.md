# DeepSeek 五档 tiny 横评

`benchmarks/queue_deepseek_tiny_sweep.py` 是一个跨主机调度器。它对 TEAL、WINA 分别等待完整 native greedy 策略；某一种方法先完成，就先启动该方法的五档任务，然后调用现有 `run_deepseek_greedy_eval.py`，不会另写一套模型推理逻辑。

## 评测范围

本轮固定运行：

- HiF4 W4A4 + ARC；
- TEAL、WINA；
- 50%、60%、70%、80%、90%；
- MATH 前 100 题、GPQA Diamond 前 32 题；
- 生成式答案判分，`max_new_tokens=8192`。

这 20 个任务用于五档筛查，结果标为 `tiny/screening`，不能填写成 MATH-500 或 GPQA Diamond 完整成绩。题目来自现有最终测试池前缀，若要作为独立选点集，应在后续换成与最终测试去重的开发题并披露当前选择偏差。

## 多主机运行

两台机器共享 `runs/deepseek_tiny_sweep_20261106/`。**每对 GPU 运行一个 worker**，同一主机的配对不能重叠，`--host-label` 必须唯一。worker 会：

1. 检查完整 `strategy.json`、校准 manifest、模型/TP/代码哈希；
2. 只从通过检查的 race 或主搜索目录选择一个策略；
3. 通过共享 claim 目录内的文件锁领取任务，跨机器避免重复启动；
4. 发现本机两个空闲 GPU 后以 TP2 启动作答；
5. 每 15 秒更新运行心跳，保存命令、GPU、策略哈希、输出目录和失败状态；
6. 验证 `arcquant-*/summary.json`、逐题覆盖、ARC 身份和两个 rank 的 greedy 执行证据后标记完成，并继续下一项；
7. 在同一档稀疏度中交错安排 WINA、TEAL，再进入下一档；默认失败后额外重试一次，失败次数跨重启保留。

调度器重启时会核对本机 evaluator 的 PID、启动时间和完整命令，然后接管仍在运行的任务。它不会因为本机找不到远程 PID 就重跑远程任务。旧版记录缺少物理主机名，首次接管时需要沿用旧 `--host-label`。远程主机宕机后留下的未完成任务需先在原主机核实进程状态，避免重复计算。

后台运行请使用独立会话，例如在 `tmux` 中运行以下命令。每个 GPU 配对启动一次，修改 `--gpu-candidates` 和 `--host-label`；资产目录按实际机器填写。只启动裸后台 shell 无法保证 SSH/终端退出后调度器仍存活。

```bash
python benchmarks/queue_deepseek_tiny_sweep.py \
  --out-dir runs/deepseek_tiny_sweep_20261106 \
  --strategy-root wina runs/deepseek_method_split_20261105_2040/wina-search \
  --strategy-root teal runs/deepseek_greedy_race_20261106/teal-search \
  --base-eval-root /path/to/base_eval \
  --source-model /path/to/DeepSeek-V4-Flash-0731 \
  --converted-model /path/to/DeepSeek-V4-Flash-0731-mp2 \
  --gpu-candidates 0,1 --host-label h20-1-pair01
```

GPU 锁只约束使用本仓库相同锁目录的程序，不会阻止其他人的作业。任务锁需要共享文件系统支持跨主机 `flock`，当前两台 H20 已实际验证。

非连续 GPU 也可以组成 TP2，例如 h20-2 的 GPU 4、7。单卡空闲不能启动 DeepSeek TP2，worker 会等待另一张卡，不会把单卡记为有效运行。

## 查看状态

```bash
cat runs/deepseek_tiny_sweep_20261106/heartbeat-h20-1-pair01.json
cat runs/deepseek_tiny_sweep_20261106/heartbeat-h20-2-priority01.json
find runs/deepseek_tiny_sweep_20261106/claims -name state.json -print
```

正式升级前应检查每个任务的 `summary.json`、`protocol.json`、逐题 `outputs.jsonl` 及 `state.json`。tiny 成绩只用于筛查；缺题、策略哈希不一致或 ARC 量化身份错误不能作为完成成绩。达到 8192 token 后截断的题按既定评分规则保留，不能删除这些题来抬高分数；比较时应同时检查截断率。

## 2026-11-07 调度调整

用户指定 DeepSeek 最高优先级，MAGI 暂缓。已停止 h20-2 的 MAGI 调度器及其生成进程，保留已完成视频缓存；当前未完成的视频恢复时可能需要重算。释放的 GPU 0、1 和 2、3 已分别运行 TEAL 50% 的 MATH 100 题、GPQA 32 题。DeepSeek 此时共运行七组 TP2：h20-1 四组，h20-2 三组。MAGI 不会被 tiny 队列自动重启。

服务器上的暂缓记录与原启动命令保存在 `runs/deepseek_tiny_sweep_20261106/magi_deferred_20261107.json`。待 DeepSeek 优先任务完成、资源允许时，可用该记录中的 `resume_command` 恢复 MAGI 并复用完成视频缓存。
