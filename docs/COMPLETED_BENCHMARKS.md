# 已完成 benchmark 结果表

完整长表提供四种格式：

- [彩色 Excel 工作簿](completed_benchmarks.xlsx)：用 Excel、LibreOffice 或 WPS 打开即可看到单元格颜色，首行冻结并支持筛选；
- [彩色 HTML 表格](completed_benchmarks_colored.html)：浏览器打开即可看到颜色、筛选框和图例；
- [TSV 表格](completed_benchmarks.tsv)：适合命令行、Excel 和 pandas；
- [CSV 表格](completed_benchmarks.csv)：适合脚本读取。

CSV/TSV 是纯文本格式，本身不保存颜色。需要在表格软件中直接看到颜色时打开上面的
`.xlsx` 文件；它不依赖插件或额外脚本。HTML 版本用于浏览器查看和快速筛选。

当前表共有 30 行，覆盖服务器已有的 DeepSeek 独立基线、MiniMax H3 的 VBench
Tiny 基线、MAGI-2 的 BF16 Tiny 结果，以及 ARCQuant 自己完成的 DeepSeek GPQA
五组对照。

## 字段说明

- `evaluated_count`：语言 benchmark 的题目/记录数；VBench 使用每种精度生成的视频数；
- `metric` 和 `protocol`：必须一起看，避免把生成式 Pass@1 和条件似然准确率混比；
- `evaluation_stack`：`base_eval` 是独立基线，`ARCQuant custom` 是本仓库的整模型方法入口；
- `source`：相对于服务器资产根目录或仓库根目录的结果文件路径。权重、视频和逐题大文件没有上传到 Git。

## 重要口径

DeepSeek 的 `base_eval` GPQA 是生成式 thinking Pass@1；ARCQuant custom GPQA 是四选项
条件似然准确率。两者都使用 198 题，但不是同一个评测协议。MiniMax 的 VBench Tiny
分数采用报告中的总分（原值乘 100）；MAGI-2 当前记录的是 VBench summary 的
`mean_of_available_dimensions` 原始值，因此不要把它和 MiniMax 总分直接横向比较。

表中所有行都已经有对应的结果文件；未完成的 MAGI-2 HiFP4 W4A4 任务、Terminal-Bench
以及尚未接入 ARC/TEAL/WINA 的 benchmark 没有放进来。
