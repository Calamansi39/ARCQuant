# DeepSeek 五档 tiny 横评

`benchmarks/queue_deepseek_tiny_sweep.py` 是一个跨主机调度器。它对 TEAL、WINA 分别等待完整 native greedy 策略；某一种方法先完成，就先启动该方法的五档任务，然后调用现有 `run_deepseek_greedy_eval.py`，不会另写一套模型推理逻辑。

## 本轮收尾与初步选点（2026-11-11）

本轮按已有协议跑完，不在中途改变题目、生成预算、评分或稀疏策略。最新成绩看同一份[彩色 Excel](deepseek_tiny_sparsity_results.xlsx)或[文本表](deepseek_tiny_sparsity_results.md)。00:33 快照为 **19/20 项完成**，仅 WINA 70% MATH 尚在运行，保存 60/100 题；使用 h20-2 GPU 5、6，进程仍在推进。共享 GPU 上还有其他作业，耗时不能由当前累计平均吞吐可靠外推。

**初步选 50% 进入下一阶段。** 在本轮 ARC HiF4 W4A4 + greedy 实现和固定输出预算下，50% 档的 TEAL/WINA 分别得到 MATH 94%/93%、GPQA 46.9%/37.5%；60% 档降到 MATH 71%/57%、GPQA 9.4%/3.1%。70%–90% 的已完成项均为零分。这里选择的是五档中当前最有依据的候选，不代表已证明最优速度与精度折中：本轮没有同协议 ARC 0% 稀疏对照，尚不能计算相对精度保留率；共享 GPU 的时间也不适合直接比较方法加速比。

### 怎样结合截断来理解结果

- **直接停止原因**：`finish_reason=length` 表示达到 8192 个输出 token；它本身不证明模型没有写过正确答案，也不表示输入被截短或发生 OOM。
- **50%**：MATH 的 TEAL/WINA 分别仅 6/100、7/100 截断；抽查包括仍在解题、反复核对而未提交终答的情况。GPQA 分别有 14/32、17/32 截断，科学问答仍对生成预算/终答协议敏感。
- **60%**：MATH 截断升为 28/100、43/100，GPQA 升为 28/32、31/32；同时准确率明显下降，当前不宜仅为提高稀疏度而优先投入全量。
- **70%–80%**：除未结束思考、缺少明确终答外，抽查发现无休止反复计算、重复字符和无关文本。WINA 80% MATH 的 97 条截断中，有 74 条被保守的末尾周期重复检查标记；这支持存在输出退化，但不证明无限延长也永远不会恢复，更不能直接归因于原论文方法。
- **90%**：已完成项没有长度截断但仍全错，模型会提前结束且没有有效终答。因此“没有截断”不等于“模型表现正常”。

11 月 10 日 CPU 审计读取 1280 条已有输出，其中 635 条达到长度上限；仅跳过“length 强制判错”后，用原终答提取器可恢复 3 条正确答案。它们是 TEAL 70% MATH 的 2 条、WINA 60% GPQA 的 1 条，不足以扭转当前选点结论。本轮正式表仍沿用原规则，没有混入这次诊断性重评分。审计在共享盘 `runs/truncation_audit_20261110/audit.json`，原始生成在本轮 runs 目录；两者未纳入 Git。

GPQA 的思考阶段还有 7168 token 的强制关闭边界及约 1024 token 的终答预留；MATH 没有这个强制关闭策略。不同 benchmark 的截断需按各自协议解释。这些观察描述了输出症状，还未严格区分高稀疏、校准分配、ARC 组合和生成协议各自的影响。

收尾后优先核对完整覆盖和逐题停止原因，再确认候选档位；下一阶段才补同协议 ARC 0% 对照，并对选定档进行进一步验证。当前 tiny 题是测试集前缀，选点后的全量成绩也应披露这部分题目曾参与选择。

### 自动刷新结果

在一台能读共享盘的机器上，用 `tmux` 或独立后台会话运行：

```bash
python tools/watch_deepseek_tiny_results.py --notes ../codexx1.md
```

该 CPU 报告器每 60 秒检查一次，复用原有完整性校验；题数、成绩或告警变化时，原位更新四种格式的结果表以及工作记录中带标记的实时区块。它只替换自己的区块，保留历史讲解。文件采用临时文件写完再替换，避免读到半个 Excel。共享文件锁阻止重复报告器，20 项均通过核验后自动退出。用 `--once` 可以只更新一次。

报告器状态：`runs/deepseek_tiny_sweep_20261106/results-watch.json`。调度心跳超过 5 分钟或日志超过 30 分钟不更新时记录告警；报告器本身不占 GPU、不重启任务、不更改评分、不自动推送 Git。原 GPU 队列继续负责运行和有限次数的失败重试。当前 GitHub 表格是提交时的快照，后续完成结果先更新共享盘，再显式提交同步。

## 40% 补充评测（2026-11-11）

由于 60%–90% 的输出稳定性明显下降，新增单独的 40% tiny 横评。40% 使用独立 native greedy 搜索，不能把 50% 策略简单截短。搜索和评测根目录分别为：

```text
runs/deepseek_40_greedy_20261111/{wina-search,teal-search}
runs/deepseek_40_tiny_sweep_20261111/
```

搜索仍使用 128 个 WikiText-2 train 片段、512 token、20 个 search samples、ARC 校准和 TP2。两条搜索在 h20-1 GPU 0–3 上运行；40% 评测队列在 GPU 4–7 上等待策略完成，使用 `--rates 0.4`，策略通过校验后自动启动 WINA/TEAL 各自 GPQA 32 题和 MATH 100 题。40% 的独立表格由 `python tools/build_deepseek_40_results.py` 生成，输出 `docs/deepseek_40_sparsity_results.{xlsx,md,csv,tsv}`。

队列和搜索都使用 `/tmp/arcquant-gpu-locks-$UID/gpu*.lock` 的本地协作锁，防止 ARCQuant 自己的调度器重复选卡。40% 运行期间 h20-1 的 GPU 0–7 另外设置为 NVIDIA `EXCLUSIVE_PROCESS`：驱动会拒绝其他计算上下文进入这些卡；搜索占用 0–3，评测队列预留 4–7。管理员仍可改变计算模式或结束进程，不能把它当成跨用户的绝对抢占权限。`tools/restore_gpu_compute_mode_after_sweep.py` 会在 40% 四项 claim 终止后恢复 `DEFAULT`。日志和 reservation JSON 同时记录实际 GPU、命令和退出状态。

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
