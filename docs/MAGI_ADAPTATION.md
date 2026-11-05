# MAGI-2：HiF4 + WINA / TEAL 适配与验证

本实现位于独立开发副本，不修改另一台机器正在使用的 ARCQuant、旧 MAGI
评测器、共享 Python 环境或模型权重。固定 `base_eval=c452d6d...`、官方
MAGI 源码 `f68a0f9...`；旧 HiF4 Tiny 仍由旧评测器 `7a399226...` 继续生成。
两套输出不可混用，已有 BF16 Tiny 不重复运行。

## 正式目标与当前缺口（2026-11-05 用户确认）

最终主表需要 **12 组完整、同协议的 Tiny 结果**：

| 分组 | 量化 | 激活稀疏 | 组数 |
| --- | --- | --- | --- |
| 精度基线 | BF16 | 无 | 1 |
| 量化基线 | ARCQuant HiFP4 W4A4，实际执行 ARC 补偿 | 无 | 1 |
| 方法组 | 同一 ARCQuant HiFP4 配置 | TEAL greedy，50/60/70/80/90% | 5 |
| 方法组 | 同一 ARCQuant HiFP4 配置 | WINA greedy，50/60/70/80/90% | 5 |

当前完成的是基础 HiF4 加 uniform 掩码的架构接线，**没有 ARC 补偿，也没有
greedy 配置学习**。TEAL 直方图采样只是阈值校准，不等于预算学习；WINA
原始权重范数也不是学习后的逐投影策略。旧 HiF4 Tiny 作为无 ARC 消融保留，
不能重命名填入 ARCQuant 基线。当前零稀疏单视频只是接线一致性检查。

正式长作业的前置顺序：

1. 完成 MAGI 的 ARC 通道统计、重排和补偿接线，核对 grouped/routed/refiner
   全路径，先做无稀疏真实单视频，确认每个 rank 实际调用补偿。
2. 固定与评测提示词完全分离的校准集及留出检查集。现有四视频配置仅为
   初始诊断方案，不能未经覆盖性检查便宣称足够完成正式 greedy 学习。
3. 分别实现 TEAL、WINA 的 MAGI greedy 搜索。明确融合投影共享 mask、
   MoE 路由以及时间桶的预算单位、目标误差与加权预算；不能直接照搬
   Llama 投影表，不能用各层统一标量代替学习得到的逐投影配额。
4. 导出覆盖 50/60/70/80/90% 的策略及校准/代码/量化身份，校验实际预算、
   覆盖率、搜索耗时、量化设置一致性。更高目标不得借用较低预算的策略。
5. 策略冻结后通过真实视频、有限值及画面检查，再运行正式 12 组矩阵。
   BF16 完整结果复用；uniform 对照仅作为另标身份的可选消融。

这些步骤尚未完成，不能把现有适配称为正式方案已经实现。代码、策略和
结果按可验证阶段提交；不提交权重、视频、环境或私密配置。

旧 `sparse-campaign-v2` 运行中的生成器不热修改。此次仅修改调度入口，
其既有源码身份门禁会在当前零稀疏检查返回后拒绝继续下一档；若看到
`Campaign implementation changed`，这是计划变更导致的有意停队，不代表
模型数值失败。新入口直接拒绝 `--tiny-after-validation`，直到另行实现
正式 ARC/greedy 队列，不会把十档 uniform 验证自动扩展为 7400 视频。

## 方法定义与边界

这是 **MAGI 视频架构适配**，标识 `magi-projection-shared-mask-v1`，不是
把 WINA 原论文的 Llama 重参数化、greedy 分配或文本层选择原封不动移植。
没有 ARC residual channels，没有 2:4，没有真正 FP4 存储或稀疏 GEMM 加速。

核查原生参考快照：WINA `04ac8794aa4ae345d09414733ceb76470d3b6a3e` 的
`wina/mlp.py`、`wina/attention.py`；TEAL
`fb7373c93ac3594817c9ee64d4e08b47430a1822` 与本仓库
`model/calibrate_sparse.py` / `model/sparse_utils.py` 的 signed histogram 规则。
原生 WINA 的 separate-projection / fused-projection 分支及权重变换假设不同；
本实现不宣称复现了不存在于 MAGI 的 Llama 变换。

| 路径 | 处理方式 |
| --- | --- |
| 普通 `nn.Linear` | 原始 `[out,in]` 权重的输入列 L2 范数；稀疏 pre-hook 在量化 pre-hook 前 |
| preview grouped / custom grouped | 按 modality 切分，分别计算原始输入列范数；保持 gather/split 契约 |
| refiner BaseLinear / NativeMoELinear | 按 dispatcher 分组；保持先 BF16 GEMM 舍入、再加 bias 的顺序 |
| fused QKV / fused gate-up | 一个 fused 输入 mask；范数覆盖融合输出矩阵 |
| routed gate/up | `[expert,in,out]`；拼接 gate/up 的输入行范数，二者共享同一个 mask |
| routed down | 独立原始输入行范数；在 SwiGLU7 后、激活量化前稀疏 |
| router / mHC / embedding / adapter / text encoder / VAE | 保持原生精度、不稀疏；报告排除清单 |

WINA 保留 `floor(width * (1 - sparsity))` 个通道（至少一个），分数为
`abs(x) * original_input_norm`。范数在 `apply_magi2_quant` **之前**分块计算，
缓存为 CPU FP32 张量，身份包含原权重文件元数据、版本、rank、stage、源码。
不复制整份权重，也不在每次 forward 重新计算范数。

专家使用固定评测器的 pack、SwiGLU、BF16 contribution 和升序专家 FP32 merge
内核，只在 GEMM 输入边界增加 mask。padding 不计入零率。稀疏率 0 的 routed
路径直接调用原量化实现，以便做严格基线一致性验证。

## TEAL 校准

`configs/magi_calibration_prompts.json` 固定四条校准提示词。预检与 **全部官方
VBench 提示词**比较，存在交集就拒绝。每条仍使用 10 秒、1080p、50+5 步、
seed=0 的生成协议；逐视频 seed 仍由固定评测器派生。

- 在真实去噪 forward 的稀疏前/激活量化前边界采样，不使用文本模型直方图。
- 按 preview/refiner、模块/投影、modality 和 `forward_index // 10` 分桶。
  这里是 DiT **forward 调用序号**，包含 CFG 调用，不冒充实际 scheduler timestep。
  每个视频重置序号；routed expert 在同层同投影内池化，再合并两个 CP/EP rank。
- 独立 CPU 随机生成器，避免污染模型采样 RNG；每次边界采样最多 512 元素、
  每个 key 最多保留 16384 元素。是有预算的 event-stratified reservoir，
  不是声称扫描了全部激活分布的精确直方图。
- signed histogram 使用 10000 bins、1% 尾部规则；阈值严格使用
  `icdf(0.5 + sparsity / 2)`，在线保留 `abs(x) > threshold`，不偷换为在线 top-k。
- 两个 rank 必须成功完成，且 preview/refiner 都有数据才能合并。
  权重、协议、环境、源码、提示词身份变化即拒绝复用；缺失投影/时间桶即失败，
  不自动改用别的桶或稠密实现。

校准中途失败时不得只复用已生成视频并采集剩余视频的激活：包装器拒绝这种
不完整样本复用。保留失败证据，使用新校准输出；只有完整两 rank 的校准产物
可以直接用于后续 TEAL。普通评测视频可以按固定身份正常接续。

## 统计与闸门

`quantization_report.json` 仅表示安装时覆盖；运行统计在
`sparse_runtime_rank0.json` / `sparse_runtime_rank1.json`，每个视频后更新：

- 覆盖/排除清单、每模块与每 forward 的调用次数、总元素、mask 数；
- 输入自然零、稀疏后实际零、输入与模型输出 NaN/Inf；
- 实际零率的边界是 **稀疏后、激活假量化前**，不是量化后的最终零率；
- 每视频 GPU 同步墙钟时间、峰值 allocated/reserved 显存与提示词 SHA256；
- rank、主机、源码哈希、方法与校准身份；异常或未完成不会标记 `complete=true`。

`video-queue` 开放 MAGI WINA/TEAL smoke。完整 Tiny 额外要求零稀疏视频验证，
并在本档 smoke 通过实际覆盖、零率、有限值检查后才运行完整生成。
零稀疏验证比较相同配置/提示词/seed 的 **解码后整视频 SHA256**。
当前 `magi_campaign.py` 仅为 uniform/no-ARC 诊断入口，不能启动正式 Tiny。
正式 ARC/greedy 队列仍需画面审阅，不能把“没有 NaN”当作画面正常。

Tiny 最终成绩必须有 740 视频、108 去重提示词、固定 manifest、16 个维度、
10 条闪烁提示词的 250 候选覆盖以及 `tiny_scores` 的 total/quality/semantic。
不使用简单维度平均，不把 smoke 分数写入正式表。

## 本机运行

本机为 `10-0-2-54`，8×H20-3e 143771 MiB。0–3 卡是旧 HiF4 任务；
4 卡有其他 VLLM 服务，不干预。当前独立验证候选卡限定为 **5,6**，
7 卡仅用于短小的 GPU 单元测试，使用本机 GPU lease 后释放。

```bash
cd /mnt/kpfs_qyang/LBC/magi_work/10-0-2-54/ARCQuant
source /mnt/kpfs_qyang/LBC/ARCQuant/external/local_env.sh
export ARCQUANT_ASSET_ROOT=/mnt/kpfs_qyang/LBC/ARCQuant/assets
export ARCQUANT_MAGI2_MODEL=/mnt/kpfs_qyang/zhaolianchao/models/MAGI-2-preview
export PATH="$ARCQUANT_FFMPEG_BIN:$PATH"
export PYTHONDONTWRITEBYTECODE=1

python3 benchmarks/magi_campaign.py \
  --baseline /mnt/kpfs_qyang/LBC/magi_runs/10-0-2-54/hif4-smoke-50p5-v2 \
  --out-dir /mnt/kpfs_qyang/LBC/magi_runs/10-0-2-54/uniform-diagnostic-new \
  --gpu-ids 5,6
```

默认只打印计划，增加 `--execute` 才执行。使用 tmux 长期运行；不要在同目录
启动第二个 campaign。它先等待独立基线、零稀疏、50% WINA smoke，再做四视频
校准、50% TEAL 和其它八档 smoke，之后停止，不开放十组 Tiny。
此命令仅说明旧诊断能力；当前优先事项是 ARC 与 greedy，不自动重开诊断网格。
每个新版本用新目录；修改源码会使正在等待的队列拒绝继续，以免混版本。

本轮第一次独立基线因 allocator 未采用旧任务的配置而在第一个 denoise forward
OOM。新队列固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、
`NCCL_ALGO=^NVLS`、`OMP_NUM_THREADS=16`。这不降低显存需求，也不允许
把 96GB H20 假装成可运行的双卡配置。旧失败日志保留于 `hif4-smoke-50p5-v1`。

进度查看：

```bash
cat /mnt/kpfs_qyang/LBC/magi_runs/10-0-2-54/sparse-campaign-v2/campaign_status.json
python3 benchmarks/report_magi2.py \
  --baseline-root /mnt/kpfs_qyang/zhaolianchao/base_eval/results/MAGI-2-preview/vbench_standard \
  --campaign /mnt/kpfs_qyang/LBC/magi_runs/10-0-2-54/sparse-campaign-v2
```

旧 `validated_smokes.json` 只证明 uniform/no-ARC 诊断，不能授权 ARC/greedy
正式扫点。画面正常也不能替代 ARC 实际执行证据和冻结的 greedy 策略。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' "$ARCQUANT_MAGI2_PYTHON" -m unittest \
  tests.test_magi2_sparse_runtime tests.test_magi2_adaptation \
  tests.test_magi_validation tests.test_video_deployment tests.test_gpu_wait -v
```

GPU 测试需先取得空闲卡 lease，再运行 `tests.test_magi2_adaptation`。
覆盖专家方向、共享范数、稀疏/量化顺序、全部 grouped/refiner 契约、空 modality、
gather、BF16 bias 舍入、padding、五档稀疏率、CUDA batched/reference 一致性、
零稀疏 CUDA HiF4 bitwise 一致性、直方图规则、独立 RNG、身份与完整性拒绝。

2026-11-05 的初次全仓库测试：116 项、3 个失败、6 个跳过；失败均为继承自
原仓库的 `test_sparsity_sweep.py` 旧网格期望（仍期望 50/65%，与当前
50/60/70/80/90% 配置不一致），不是本次 MAGI 改动。不要为了本机任务修改
正在承担 DeepSeek 评测的共享配置。

单元测试、CPU dry-run 或“已启动”均不代表真实单视频/740 视频已经完成。
最终状态以运行目录、两 rank 运行证据和完整评分为准。
