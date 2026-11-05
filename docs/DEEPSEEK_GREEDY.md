# DeepSeek：greedy 离线预算搜索与推理

两台机器各做一种方法、从现有断点安全续跑，见 [双机搜索与交接](DEEPSEEK_MULTI_HOST.md)。

实现日期：2026-11-05。支持 DeepSeek-V4-Flash-0731 的官方 TP 权重。
运行状态以 `runs/.../queue_status.json`、`status.json` 和策略文件的 `complete` 为准；
脚本存在、CPU 测试通过或单层 pilot 通过均不等于完整模型的预算已搜索完成。

## 做什么

1. 使用独立 WikiText-2 train 文本，采集实际层输入、激活分布及专家路由覆盖。
2. 生成当前层的 ARC 校准数据和无激活稀疏的 HiF4+ARC 参考输出。
3. 沿用 TEAL/WINA 原生 greedy 的候选比较：对一个投影增加相同加权预算，
   重做真实层前向，选择相对参考输出误差最小的候选，逐步形成预算曲线。
4. WINA 离线用加权绝对激活的经验 CDF/阈值，在线仍用加权 top-k；TEAL 用 signed
   histogram 阈值。误差保留原版 `norm(..., dim=1).mean()`，默认只计算末尾 25% token。
5. 一条曲线保存 50%/60%/70%/80%/90% 五个精确预算点，分别比较同预算 uniform 误差。
6. 将策略装入生成式 MATH/GPQA 入口，策略哈希进入运行标识与评分协议，避免混用 uniform 结果。

没有反向传播。当前 WINA transform 为关闭状态，不将其标成原复现脚本包含 transform 的完整配置。

## 原生算法与 DeepSeek 适配边界

原算法来自 TEAL `fb7373c` 和 WINA `04ac879` 的 `greedyopt.py`；候选选择、反比步长、
层输出误差和 WINA threshold-search/top-k-eval 流程均有明确来源。
原脚本绑定 dense 模型层接口、投影名和硬编码权重表，不能直接加载 DeepSeek MLA/HC/MoE。
本实现保留搜索规则，通过适配器调用真正的 DeepSeek 层前向。

- **分组**：每层注意力投影分别分配；grouped wo_a 共用一个投影配额；路由专家同类
  w1/w2/w3 分别共享配额，共享专家单独分配。不是逐专家独立搜索。
- **WINA 权重重要性映射**：主注意力 `attn.wkv` 和 FFN 的 `w1` 使用已反量化源权重的
  列 L2 范数；query/output、`w2/w3` 以及新增 compressor/indexer 投影使用单位重要性。
  这沿用当前 DeepSeek adapter 的显式映射，新增结构没有原仓库一一对应的配置，
  单位重要性并不表示已证明是这些新投影的最优选择。
- **成本**：默认按校准中实际执行的逻辑线性 MAC 数加权，处理 TP 分片和重复副本。
  这使 dense 情况退化为矩阵大小比例，也避免 MoE 未激活参数数量淹没预算。
  专家补采不增加成本权重；完全未观测组保持 dense，并显式记录。
- **步长**：以当前层 wq_b 的校准成本为单位，默认 `0.025`，其余组按成本倒数缩放。
  目标点最后一步裁剪到精确预算；原代码可能越过目标点，本实现不静默使用最近点替代。
- **校准样本**：默认 128×512 token，其中前 20 段用于候选误差搜索；每投影直方图
  保留最多 16,384 个确定性抽样值。这个有界抽样是工程适配，非保留原版全部激活。
- **批量重放**：默认最多 20 个独立样本一起前向，参考和候选使用相同批次边界，
  再按样本数平均误差；避免 BF16 不同批大小造成的数值差异影响候选排序。
  显存较小时可降低 `--search-batch-size`，每层都检查零稀疏重放误差。
- **基座**：默认 HiF4 W4A4 + ARC 无激活稀疏输出；逐层以该基座的输出向下一层传播。
  原生 BF16 方法与 HiF4+ARC 组合仍须在相同评测协议下验证，不能直接套用原论文分数。
- 每次层重放清空 KV/压缩缓存，MoE token 位置和 TP 集合通信保留；各 rank 使用相同的候选顺序。
- WINA 在线 top-k 对输入分片投影先拼回逻辑通道再选择；其他投影沿用原有 mask。
  每层另外保存 threshold 搜索策略在 top-k 下的误差，避免只验证离线掩码。
- `--layer-streaming` 将完整模型保留在主存，只把正在搜索的一层搬到显卡；队列默认启用。
  需要充足主存：每个 TP 进程保留各自权重及校准层输入。它改变存放位置，不替换真实层计算。
- 这里的预算是校准成本加权的配置预算，不是测得的端到端 FLOP 减少、全局零率或加速倍数。

## 快速启动

使用装有 PyTorch 和 DeepSeek 依赖的评测 Python。大模型/数据路径由参数提供，代码中没有本机绝对路径。

```bash
python arc.py deepseek-greedy \
  --gpu-ids 0,1 --out-dir runs/greedy_campaign \
  --base-eval-root /data/base_eval \
  --source-model /data/models/DeepSeek-V4-Flash-0731 \
  --converted-model /data/models/DeepSeek-V4-Flash-0731-mp2 \
  --calibration-data /data/wikitext-2-raw-v1/train-00000-of-00001.parquet
```

队列等待当前 ARCQuant GPU lease 结束，再核对空卡；不会杀进程或申请空占显存。
先运行 2×128 token、首层、各一步候选搜索的 WINA 和 TEAL pilot，失败即停止；通过后进入完整配置。
进入等待前固定代码快照，后续编辑工作仓库不会改变正在运行的实验。
持有同一对本机协作锁直至该批次结束。这不是集群资源预约，不能约束不使用协作锁的任务。
若只需试跑，增加 `--pilot-only`。

直接运行/恢复搜索：

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/deepseek_greedy_search.py \
  --base-eval-root /data/base_eval --source-model /data/models/DeepSeek-V4-Flash-0731 \
  --converted-model /data/models/DeepSeek-V4-Flash-0731-mp2 \
  --calibration-data /data/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --out-dir runs/greedy_full --methods wina,teal --quantization arc \
  --samples 128 --search-samples 20 --length 512 \
  --targets 0.5,0.6,0.7,0.8,0.9 --step 0.025 --layer-streaming
```

恢复时保持数据、代码、量化和校准参数一致，加 `--resume`。每一步保存预算，
每层保存下一层输入；不需要重做已经完成的搜索层。诊断 `--max-steps` 可在恢复时移除。
`--dry-run` 验证本地资产和签名，不运行 GPU。
若已做正式规模的部分搜索，可给队列加 `--resume-search runs/greedy_full`，并为队列本身
指定新的 `--out-dir`。它取得 GPU 协作锁后继续已有断点；模型、数据和搜索源码必须一致。
可设置 `--gpu-memory-gib 24` 限制每个进程的 PyTorch 显存分配器预算；该值不包含 CUDA
上下文等少量外部分配，也不是 GPU 计算时间预约。与本人的其他任务共卡前需确认实际余量。

## 产物

- `search_manifest.json`：数据、模型、TP、量化与实现哈希；`sources/` 保存本次源码。
- `calibration/layer-N.rankR.pt`：各层 ARC/TEAL/WINA 统计；覆盖文件记录专家补采和最少行数。
- `wina/layer-N.json`、`teal/layer-N.json`：逐步候选选择、误差、各组预算、耗时与断点。
- `wina/layer-N.online.json`：同一层、同一预算下 greedy 与 uniform 的在线 top-k 误差。
- `wina/strategy.json`、`teal/strategy.json`：按五档预算整理的策略。只有全部 43 层齐全才 `complete=true`。
- 全模型结束后生成 `calibration/rankR.pt` 和 `manifest.json`，供同基座推理加载。
- `status.json`：当前层、方法、阶段、迭代和进度。单层 pilot 结果不会被评测入口接受为完整策略。

## 用搜索结果评测

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/run_deepseek_greedy_eval.py \
  --strategy runs/greedy_full/wina/strategy.json --sparse-method wina --sparsity 0.8 \
  --quantization arc --calibration runs/greedy_full/calibration \
  --base-eval-root /data/base_eval --source-model /data/models/DeepSeek-V4-Flash-0731 \
  --converted-model /data/models/DeepSeek-V4-Flash-0731-mp2 \
  --benchmark math500 --quant hifp4 --mode w4a4 --limit 500 \
  --batch-size 20 --max-new-tokens 8192 --out-dir runs/greedy_math80
```

TEAL 改方法和策略路径；GPQA 改 `--benchmark gpqa` 及对应完整题数。
当前专用 greedy 包装入口覆盖 MATH/GPQA。LongBench/LiveCodeBench 的 greedy+ARC
生成调度尚需适配和 GPU 验证，不能从这个入口推断所有 benchmark 已完成接线。

策略必须完整、目标点存在、模型/TP/量化/校准一致、投影覆盖完整；否则拒绝运行。
不同预算文件有不同运行身份，不会把 uniform 成绩改名为 greedy。

## 外行读法：离线和在线各做什么

把一层里的多个投影理解成几个工位。统一稀疏率相当于每个工位都删掉 70% 的输入；
greedy 则实际试算“这次把删减额度分给哪个工位，输出变化最小”，据此分配。
例如最终可能有的投影是 40%、有的是 85%，按计算成本加权后仍是该层 70%。
这些数字仅为解释例子，不是本轮已得到的预算。

**离线**读取独立训练文本，统计分布、反复试算层输出，产出逐投影预算文件；不更新模型参数。
**在线**加载文件直接执行：TEAL 按该投影的已校准阈值筛选激活，WINA 按加权分数做 top-k。
在线不再进行 greedy 搜索。这里没有把两者改成 2:4。

目前保留了原 greedy 的比较和步长规则，DeepSeek 专家分组、按实际路由成本加权、
有界抽样、精确目标截断属于明确记录的适配；误差使用 FP32 计算原来的序列维范数。
WINA transform、全模型 greedy 结果和完整 benchmark 精度应分别验收，不能相互替代。

## 已验证的阶段

真实 DeepSeek TP2、HiF4+ARC、首层 2×128 token 的两种方法各一步搜索已通过。
WINA 和 TEAL 均比较了 11 个投影候选；WINA 同时执行了在线 top-k 对照。
完整证据保存在 [DEEPSEEK_GREEDY_VALIDATION.json](DEEPSEEK_GREEDY_VALIDATION.json)。
这些是运行验证，不是完整 43 层策略，也不是全量 benchmark 成绩。
128×512 校准、20 段误差样本的正式规模测速与完整搜索状态继续记录在运行目录。

正式规模首层复测（128×512 校准、20 段误差样本、batch 20）也已通过：
两种方法的零稀疏重放误差均为 0；WINA 首步约 4.66 秒，TEAL 约 4.10 秒。
首层预算增量为 0.2963%，因此这里只能验证候选搜索，不能报告 50%–90% 的方法精度。
若其余 42 层与首层同速，到 90% 的两方法搜索约为 31.8 小时 / 63.6 GPU 小时；
这仅是基于首步的外推，不含全部校准和加载，也未覆盖后续层结构及候选饱和的变化。
详细原始误差、候选、实现哈希与估算假设均在上述验证 JSON 中。

2026-11-05 18:54（Asia/Shanghai）部署状态：最终两方法 pilot 已通过，
完整 43 层队列已进入 `full`，运行于 GPU 0/1。运行目录为
`runs/native_greedy_campaign_20261105_185006/`；后续以其中的实时状态为准。
正式规模首层每 rank 覆盖 395 个投影，最低观测行数为 660/630，无专家补采。
尚未生成完整模型策略或新增完整 benchmark 准确率。
