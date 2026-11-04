# 已完成 benchmark 结果表

请使用新的[实验结果展示文档](RESULTS_PRESENTATION.md)，可直接分享 GitHub 链接。

- [彩色 Excel 工作簿](completed_benchmarks.xlsx?raw=true)：默认显示阅读说明与三张成绩页，另四张核查附页已隐藏；打开即可显示颜色。
- [CSV](completed_benchmarks.csv) / [TSV](completed_benchmarks.tsv)：15 项符合展示范围的完整文本结果。
- [视频 Tiny 参考 CSV](video_tiny_reference.csv)：3 项参考结果，不计入完整 benchmark 主表。
- [审计快照](results_audit.json)：原始计数、协议、来源路径及 SHA256。

2026-11-04 已重新核对逐题记录和汇总分数。旧 30 行混合表已替换：

1. MATH-500 的 0-shot 分数改用正确的 math500_generative 来源；旧路径属于 4-shot 协议。
2. 按任务指定的 HiF4 W4A4 筛选，保留 BF16 参考：10 项文本基线和 5 项 GPQA 条件似然方法对照分别展示。
3. MiniMax / MAGI 的视频分数仅为 Tiny 附录；MiniMax BF16 闪烁覆盖不足（46/50）。
4. MAGI Tiny 改为报告内聚合总分 80.00；旧 0.6372 是不同指标的维度均值。

NVFP4/MXFP4 是服务器已有的历史结果，本次没有为整理表格重新运行这些模型评测。
这些额外格式已从展示表移除，原始审计快照仍保留 25 项完整文本与 5 项视频参考记录供追溯。

Excel 可直接导入腾讯文档、飞书或 Excel Online。仓库提供可分享阅读的文档和下载文件；
本次没有创建这些平台的在线协作表格。CSV/TSV 不保存颜色。

更新命令与核查边界见[展示文档末尾](RESULTS_PRESENTATION.md#更新方法)。旧 HTML 入口已停用。
