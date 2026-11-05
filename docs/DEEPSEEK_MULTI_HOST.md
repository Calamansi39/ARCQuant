# DeepSeek：把两种方法分到两台机器

## 为什么这样分

当前任务是在校准文本上，为每层搜索 50%、60%、70%、80%、90% 的稀疏预算。
原来同一组两张卡先做 WINA，再做 TEAL；两种方法的搜索互不依赖。

因此可以让两台机器**各用两张卡、各做一种方法**：

| 位置 | GPU | 工作 |
| --- | --- | --- |
| 原 DeepSeek 机器 | 0、1 | WINA，全部五档 |
| h20-2 / `10.0.2.54` | 5、6 | TEAL，全部五档 |

理想情况下，两条耗时相近的任务并行后，搜索墙钟时间接近减半。
这不是已实测的加速倍数：还要计入等卡、加载、各层差异和校准开销。
GPU 总耗时未必减少。后续不同 benchmark / 稀疏度也可以按任务分组并行。

当前模型文件是 TP2 分片，每个任务仍需要两张卡。直接把一个任务的 GPU
数从 2 改成 4 或 8，既不符合现有分片，也不能保证更快。单条 greedy 曲线
的后一步依赖前一步；候选跨组并行需要另行实现和验证。

## 当前资源安排

2026-11-05 核查 h20-2 时，8 张卡都有进程：0–3 跑原 MAGI HiF4 对照，
5–6 是新增 MAGI 辅助 worker，4、7 属于其他服务。用户选择：**5–6 当前
双视频批次完成后交给 DeepSeek**。使用 MAGI 已有的 `STOP_AFTER_BATCH`
门禁，不中断当前视频；原 0–3 卡继续 MAGI。

共享盘已核对，DeepSeek 权重、校准数据、评测器和 Python 环境可以复用，
不需要再复制整个模型。h20-2 CPU 预检通过；真实 GPU 验证在让卡后执行。

## 自动交接怎么保护已有进度

入口为 `benchmarks/split_deepseek_greedy.py`：

1. 保存调度与计算源码副本，记录原队列 PID 和进程启动时间。
2. h20-2 等待 5–6 的协作锁和实际空闲，随后执行 TEAL 的真实一步短验证；
   零稀疏重放必须与参考一致。等待期间原机器继续两种方法的搜索。
3. 验证通过后，把当前断点复制到两个**新的输出目录**。两卡的层边界和
   校准样本必须一致，已完成层的预算、校准文件、代码哈希必须匹配；
   若复制期间原任务跨层，则拒绝这次快照并重试。
4. 断点复制和两侧配置检查全部成功后，核实原队列身份，再停止该队列。
   两个新任务各自恢复一种方法。可能重算复制期间的少量最新步数，
   已保存的完整前缀层和各档预算会保留。
5. 各自完成后重建完整校准文件及其哈希，导出各自的完整策略。
   后续评测分别使用对应任务的策略和校准目录。

原目录和 manifest 不修改；只在新 manifest 中把 `methods` 由两种改为
选中的一种。每个新目录的 `fork_provenance.json` 记录来源、层边界及
复制文件的 SHA256。不能把一个方法的策略配上另一目录的校准文件。
原实现每层参考激活由无稀疏的量化基线传播，因此两方法可以独立续跑；
此工具不适用于前层稀疏策略会改变后层校准输入的其他算法。

## 使用与检查

以下第一步只创建计划和源码副本，不启动远端、不停止原队列：

```bash
python -m benchmarks.split_deepseek_greedy \
  --parent-queue /shared/ARCQuant/runs/original_campaign \
  --out-dir /shared/ARCQuant/runs/method_split \
  --remote-host root@10.0.2.54 --remote-hostname 10-0-2-54 \
  --remote-gpus 5,6
```

确认目标机器共享同样路径、Python 可用，并已安排让卡后，在原机器启动：

```bash
python /shared/ARCQuant/runs/method_split/source/benchmarks/split_deepseek_greedy.py \
  --plan /shared/ARCQuant/runs/method_split/plan.json --role local
```

长任务应由 tmux 或后台进程管理。远端控制进程自动通过已配置的 SSH 启动；
两个角色各有独占锁，不允许重复执行同一计划。

| 文件 | 用途 |
| --- | --- |
| `plan.json` | 主机、卡号、来源队列和固定源码 |
| `local_status.json` / `remote_status.json` | 等卡、验证、复制、运行或失败状态 |
| `remote-pilot.log` | h20-2 真实 GPU 短验证 |
| `handoff_commit.json` | 已完成断点交接、原队列已退出的标志 |
| `wina-search/status.json` / `teal-search/status.json` | 实际层搜索进度 |
| `wina-full.log` / `teal-full.log` | 两种方法的计算日志 |

`ready_for_handoff` 只表示远端验证通过；只有 `handoff_commit.json` 存在，
且两份搜索状态持续更新，才说明已经真正双机并行。远端在交接前失败时，
原队列继续运行；交接后的失败会保留独立断点，需按状态恢复。
本调度只完成预算搜索，不会自动启动 benchmark 全量答题。

验证覆盖一致断点复制、拒绝不同 rank 层边界、缺失校准、源码变化、
不完整前缀、进程 PID 复用和远端验证失败时保持原队列。另对运行中真实
128×512 双卡断点进行了 CPU 复制演练，保留前两层 TEAL，未占用额外 GPU。
