# DeepSeek：TEAL/WINA 预算分配核查

核查日期：2026-11-05。结论以当前源码、产物和运行入口为依据。

**当前 DeepSeek 使用统一稀疏配额（uniform），没有执行或接入数据驱动的 greedy
预算分配。WINA/TEAL 原仓库均包含这一流程；不能把当前简化配置说成完整复现。**

## 1. 三件不同的事

| 工作 | 解决的问题 | 当前 DeepSeek 状态 |
| --- | --- | --- |
| 在线选通 | 给定某投影的稀疏率，删除哪些输入分量？ | 已接入 TEAL 阈值 / WINA 加权 top-k |
| 离线预算分配 | 各投影分别使用多少稀疏率，才能在同一总预算下减小输出误差？ | 未适配、未执行；统一传入一个稀疏率 |
| WINA transform | 在配套坐标变换后运行校准与稀疏推理 | 当前 DeepSeek 未执行；旧模型代码不能当作完成证据 |

“training-free”表示不通过反向传播训练模型，不表示没有数据采集、离线搜索或准备成本。
原仓库支持 uniform 作为一种选择，但 WINA 的复现脚本使用 greedy 和 transform。

## 2. 原仓库的预算流程

本次复读的固定源码：

- TEAL：`FasterDecoding/TEAL`，`fb7373c93ac3594817c9ee64d4e08b47430a1822`。
- WINA：`Calamansi39/wina`，`04ac8794aa4ae345d09414733ceb76470d3b6a3e`。
  另直接读取 `microsoft/wina` 主分支的 `wina/greedyopt.py`，确认与本地文件在
  统一换行后内容一致；TEAL 的 `teal/greedyopt.py` 也与官方主分支一致。
  这只核实这两个文件，不代表 fork 的全部文件都已与 upstream 比较。

WINA README 的 `Sparsity Allocation` 明确要求先采集激活，再调用
`wina/greedyopt.py`。其复现脚本先运行 `grab_acts.py --transform`，再运行
`greedyopt.py --transform`，最后以 `--greedy --transform --mask_by topk` 评测。

具体流程是：

1. 保存校准输入及相应分布；WINA 会采集乘以对应权重列重要性后的激活。
2. 对每个 Transformer 层，用未稀疏的层输出作为目标。
3. 从零稀疏率出发，分别尝试给该层一个投影增加一小份稀疏预算。
4. 重新做层前向，比较候选的输出误差；选择误差最小的候选。
5. 重复到目标有效稀疏率，沿途写入每层 `lookup/layer-*/results.csv`。
6. 评测时，从每层 CSV 找到最接近目标有效稀疏率的一行，将其中的各投影配额装载到模型。

预算按投影大小的相对权重计算：`sum(weight[p] * sparsity[p]) / sum(weight[p])`。
候选稀疏率增量按权重的倒数缩放，使各候选增加相同的加权预算。
输出误差使用序列后段，默认 `last_fraction=0.25`；不是用 benchmark 正确答案选配额。
原加载器默认给每层相同的目标有效稀疏率，在各层内部为投影分配不同配额，
不能将其描述成已经实现任意跨层、跨专家的全局最优分配。

一个仅用于说明的例子：两个大小相同的投影，总目标 50%，可能分成 30% 和 70%。
敏感投影少删一些，耐受投影多删一些；真实投影大小不同时要按权重计算总预算。
这组数字不是 DeepSeek 的实测策略。

### 原版 WINA 的重要细节

固定版本的 `wina/greedyopt.py` 用 `mask_by='threshold'` 构建搜索模型，依赖采集的
加权激活分布；复现评测脚本使用 `mask_by='topk'`。离线阈值搜索和在线 top-k
不是同一次操作。因此不能只实现在线 `abs(x) * norm -> top-k` 就声称复现了整条流程。
原版 greedy 默认搜索到 0.66；要评测本项目 70%/80%/90%，必须生成覆盖这些预算的曲线，
不能把不足覆盖范围的 CSV 通过“最近一行”默默复用为更高目标。

TEAL 官方同样有 `teal/greedyopt.py`，其预算搜索通过直方图阈值改变各投影稀疏率，
再按层输出误差选择候选。统计直方图只是准备输入，尚未完成预算优化。

## 3. 本仓库实际接到了哪里

- `arc.py:calibration_plan` 将现有通用 greedy 校准限制在旧的 Llama adapter；
  `model/calibrate_sparse.py` 也检查 `model_type == 'llama'`。它不是 DeepSeek 的入口。
- 通用 `greedy_allocate_layer` 是参考思路的重实现。其中 WINA 候选直接使用 top-k
  掩码，与上述原版阈值搜索存在差别；不能称为逐项原样复现。
- `benchmarks/deepseek_experiment.py` 只有一个 `--sparsity`，没有 greedy/strategy 入口。
- `benchmarks/run_deepseek_sparse_math.py` 和 `run_deepseek_sparse_extended.py`
  同样把一个标量稀疏率传给生成式 runtime。
- `model/deepseek_generation_runtime.py` 的 WINA 使用同一个 `self.sparse_sparsity`
  决定 top-k；TEAL 用同一目标率给各投影直方图换算阈值。不同阈值不等于不同预算。
- 当前校准目录保存了直方图、ARC 通道顺序和残差通道数，没有 DeepSeek greedy
  预算曲线或策略表。已经核查的 DeepSeek 入口也不读取任何此类策略。

因此当前结果应标为 **TEAL uniform** 或 **WINA uniform / mask-only / transform 关闭**。
2026-11-05 更新：MATH/GPQA 生成式队列已接入 ARC 补偿，见 [ARC 队列说明](DEEPSEEK_ARC_QUEUE.md)。这次接线不包含 greedy 分配或 WINA transform；标签似然诊断与生成式成绩仍须分开报告。

## 4. 耗时和后续实现要求

此前记录的约 75.8 秒来自 `calibration80` 的 1×64 token 统计流程，
**不包含 greedy 预算搜索**。DeepSeek 没跑过这一步，因此目前没有其真实耗时。
也不能由这 75.8 秒简单估出完整预算搜索成本：搜索会对多个候选、多个步长重复前向。

要补齐目标模型上的预算分配，需要：

1. 按 DeepSeek MLA、分组投影和 MoE 结构定义预算单位，明确参数量预算与实际激活计算预算
   的区别；原仓库的 dense 投影权重表不能直接套用。
2. 在独立校准集保存层输入/参考输出和路由覆盖，采集足够代表性的直方图；明确使用
   哪个量化与 ARC 配置作为搜索基座。
3. 实现与原始规则对应的候选搜索、误差度量、五档预算曲线和可恢复产物；若改变搜索
   掩码或成本模型，明确标为适配变体并做对照。
4. 将策略显式接入所有 DeepSeek 评测入口，记录策略哈希、预算定义、校准签名，
   拒绝缺投影、TP 不匹配、范围不足或坐标变换不匹配的策略。
5. 在同一基座、同一协议下比较 uniform 与 greedy，再补完整 benchmark。

这些是本审计时的待实现要求，不是本次审计已完成或已入队的实验。本次修正方法说明，没有修改运行中
的模型配置，也没有将 uniform 运行中途切换为另一方法。

## 后续实现入口（2026-11-05）

审计之后已新增 DeepSeek 专用 greedy 搜索、五档预算导出、断点恢复及 MATH/GPQA
策略加载入口，使用方法和实际验证状态见 [DEEPSEEK_GREEDY.md](DEEPSEEK_GREEDY.md)。
本页保留审计时的历史发现；完整策略是否已生成，以对应运行产物为准。
