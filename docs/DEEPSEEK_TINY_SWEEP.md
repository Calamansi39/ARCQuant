# DeepSeek 五档 tiny 横评

`benchmarks/queue_deepseek_tiny_sweep.py` 是一个跨主机调度器。它等待 TEAL/WINA 的完整 native greedy 策略，然后调用现有 `run_deepseek_greedy_eval.py`，不会另写一套模型推理逻辑。

## 评测范围

本轮固定运行：

- HiF4 W4A4 + ARC；
- TEAL、WINA；
- 50%、60%、70%、80%、90%；
- MATH 前 100 题、GPQA Diamond 前 32 题；
- 生成式答案判分，`max_new_tokens=8192`。

这 20 个任务用于五档筛查，结果标为 `tiny/screening`，不能填写成 MATH-500 或 GPQA Diamond 完整成绩。题目来自现有最终测试池前缀，若要作为独立选点集，应在后续换成与最终测试去重的开发题并披露当前选择偏差。

## 多主机运行

两台机器共享 `runs/deepseek_tiny_sweep_20261106/`。每台运行一个 worker。worker 会：

1. 检查完整 `strategy.json`、校准 manifest、模型/TP/代码哈希；
2. 只从通过检查的 race 或主搜索目录选择一个策略；
3. 通过共享 claim 目录领取尚未运行的任务；
4. 发现本机两个空闲 GPU 后以 TP2 启动作答；
5. 保存命令、GPU、策略哈希、输出目录、耗时和失败状态。

非连续 GPU 也可以组成 TP2，例如 h20-2 的 GPU 4、7。单卡空闲不能启动 DeepSeek TP2，worker 会等待另一张卡，不会把单卡记为有效运行。

## 查看状态

```bash
cat runs/deepseek_tiny_sweep_20261106/heartbeat-h20-1.json
cat runs/deepseek_tiny_sweep_20261106/heartbeat-h20-2.json
find runs/deepseek_tiny_sweep_20261106/claims -name state.json -print
```

正式升级前应检查每个任务的 `summary.json`、`protocol.json`、逐题 `outputs.jsonl` 及 `state.json`。任何截断、缺题、策略哈希不一致或 ARC 量化身份错误都不能填入正式结果表。
