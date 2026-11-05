# 连续双卡实验队列

## 这次解决什么问题

DeepSeek 的评测需要两张 GPU。以前每个方法、稀疏率、验证阶段分别等卡，阶段结束就释放队列锁。现在可以把一批实验交给同一个后台进程：拿到指定两张卡的本地锁后，连续运行整批实验，到结束或退出时才释放锁。

**这能减少本仓库队列之间的抢卡和重复等待，但不能保证独占服务器 GPU。** 文件锁只对同一机器、同一操作系统用户、遵守相同锁协议的启动器有效。其他用户的程序不受该锁限制。每个评测子进程退出后会释放显存，下一个子进程再加载模型；中间存在空档，不是持续占满显存。需要管理员或集群调度器分配两张卡，才能获得资源层面的独占保证。

## 运行的实验

入口：`benchmarks/queue_sparse_sweep.py`。

- 模型：DeepSeek；benchmark：MATH-500；量化：HiF4 W4A4。
- 默认扫描 TEAL、WINA 的 50%、60%、70%、80%、90% 目标激活元素稀疏率。
- 两种方法是非结构化激活稀疏；这条队列没有施加 2:4。
- 每个组合依次进行 1 题检查、8 题检查、500 题全量评测。
- 截断回答按错误保留；缺题、重复题号、非法生成记录等会阻止该组合继续。
- 某个组合失败后记录 `needs_debug`，继续其他组合。不能把失败或检查通过写成全量成绩。
- `--skip wina:0.5` 用于跳过已经在独立进程运行的 WINA 50%，剩余 9 个组合。

这是有限批次，默认只负责上述 MATH-500 网格。GPQA、LongBench、LiveCodeBench、视频实验没有自动包含在内。整批结束、代码发生变化或进程退出后会释放锁，不会无限占卡；没有配置断电重启后的自动恢复。

## 在其他服务器启动

先按 [迁移指南](MIGRATION_EVALUATION.md) 准备模型、TP2 权重、评测环境与校准文件。下面路径都需替换成目标服务器实际路径，使用已经装好依赖的 Python。

```bash
python benchmarks/queue_sparse_sweep.py \
  --base-eval-root /data/base_eval \
  --source-model /data/models/deepseek \
  --converted-model /data/models/deepseek-mp2 \
  --calibration /data/calibration/deepseek \
  --out-dir runs/sparse_preflight_run1 \
  --gpu-ids 0,1 --skip wina:0.5 --dry-run
```

`--dry-run` 只检查本地资产与原生评测入口，不加载整模型、不获取 GPU 锁，也不能证明整模型数值正确。迁移后若没有另行运行 WINA 50%，应去掉 `--skip wina:0.5`，运行完整 10 个组合。

预检通过后，换一个**新的输出目录**，去掉 `--dry-run`，后台运行：

```bash
mkdir -p runs/sparse_sweep_run1
nohup python benchmarks/queue_sparse_sweep.py \
  --base-eval-root /data/base_eval \
  --source-model /data/models/deepseek \
  --converted-model /data/models/deepseek-mp2 \
  --calibration /data/calibration/deepseek \
  --out-dir runs/sparse_sweep_run1 \
  --gpu-ids 0,1 --skip wina:0.5 \
  > runs/sparse_sweep_run1/launcher.out 2>&1 < /dev/null &
echo $! > runs/sparse_sweep_run1/launcher.pid
```

`--gpu-ids` 是 `nvidia-smi` 显示的两个物理编号。默认要求两卡各有至少 `138000 MiB` 可用显存、GPU 利用率不超过 5%、没有计算进程，连续检查通过后才开始。该阈值适用于当前 H20 环境；其他机器可用 `--min-free-mib` 调整，但减少阈值不会降低模型实际显存需求。默认每 2 秒检查一次。

## 查看和停止

```bash
cat runs/sparse_sweep_run1/status.json
tail -n 30 runs/sparse_sweep_run1/launcher.out
# 每个组合的详细阶段状态：
cat runs/sparse_sweep_run1/teal-0.5/status.json
```

顶层 `status.json` 记录主机、PID、GPU 编号、代码指纹及所有组合。`waiting_for_gpu_locks` 表示同协议任务还在用卡；`waiting_for_gpu_memory` 表示锁已取得但卡仍忙；`running` 表示正在执行组合；`completed` 表示本批次完成；`finished_with_failures` 表示至少一个组合待修复。

需要停止时，先用 `ps -p <PID> -o pid,args` 核对 PID 对应本队列，再向**队列主进程**发 `kill -TERM <PID>`。队列会清理它启动的当前评测子进程并释放自己的锁。不要停止其他用户的 GPU 进程，也不要删除正在使用的锁文件。

每次启动使用新输出目录。等待或运行期间不要修改指纹覆盖的算法/队列源码；修改会让后续组合停止，避免不同实现混在同一批次里。文档更新不影响指纹。重启前查清已经完成的组合，用 `--skip` 逐个排除，原目录保留作为审计记录。

## 实现与验证

外层用 `fcntl.flock` 持有 `/tmp/arcquant-gpu-locks-<uid>/gpu<index>.lock`，内层沿用 `queue_sparse_baselines.py` 的检查与评测逻辑，只借用外层的两卡锁，不在每个阶段释放/重新获取。锁文件放在服务器本地 `/tmp`，不能放到共享存储上假装跨机器预约。

单元测试覆盖连续持锁、退出释放、忙卡等待、9 组实验共享一份锁、组合失败继续、CPU 预检不申请锁、源码变化阻止启动；执行：

```bash
python -m unittest discover -s tests -p 'test_sparse_queue.py' -v
python -m unittest discover -s tests -p 'test_sparse_sweep.py' -v
```

通过单元测试和 CPU 预检只证明排队与启动前检查可用。是否完成自动衔接、是否跑完全量，仍应以真实进程和结果文件为准。
