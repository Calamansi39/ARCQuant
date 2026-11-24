# 工作区和结果入口

## 克隆仓库后先看哪里

| 目的 | 仓库内入口 |
| --- | --- |
| 查看实验成绩 | [CSV / TSV / 彩色 Excel](../reports/verified/README.md) |
| 了解评测范围、哪些结果可汇报 | [结果说明](../reports/verified/README.md) |
| 安装和使用方法 | [中文快速开始](QUICKSTART.md) |
| 查看 DS 当前调度和续跑记录 | [DS 评测说明](DEEPSEEK_DENSE_PRIORITY.md) |
| 理解 TEAL/WINA 搜索和推理 | [greedy 方法说明](DEEPSEEK_GREEDY.md) |
| 迁移 MAGI / MiniMax | [H20 交接](MAGI_H20_HANDOFF.md) / [A800 交接](MINIMAX_A800_HANDOFF.md) |

结果只发布到 `reports/verified/`。Excel 内三页分别是 `DS_Tiny`、`DS_全量dense`、`MAGI_Tiny`；每页另有同名 `.csv` 和 `.tsv`，列名、百分比、缺失数据标记与总表一致，随结果刷新同步生成。CSV 使用 UTF-8 BOM，TSV 使用 UTF-8。原始逐题结果、搜索策略与日志在运行目录中；完整性与来源信息见结果目录中的 `audit.json`。

## 当前服务器的目录

工作区是 `/mnt/kpfs_qyang/LBC`。2026-11-24 整理后的日常入口：

```text
LBC/README.md                  工作区导航
LBC/实验结果/                 快捷入口，指向 ARCQuant/reports/verified/
LBC/ARCQuant/                 主仓库
LBC/codexx1.md                动态工作记录，新记录在前
LBC/项目资料/项目要求/        SOW 合作要求原件
LBC/项目资料/参考附件/        图片附件
LBC/项目资料/服务器迁移/      MiniMax A800 离线包与校验文件
LBC/历史归档/环境迁移模板/    旧环境迁移工具
LBC/历史归档/整理记录/        清理审计清单
```

SOW 的新位置是 `LBC/项目资料/项目要求/面向长序列的大模型稀疏量化与通信加速技术合作项目SOW_v3.docx`。MAGI/A800 交接说明统一使用本仓库 `docs/` 下的文件。另一台共享挂载的机器可以直接读取这些位置。

`magi_work/` 是历史 MAGI 工作副本，包含未提交修改，已完整保留。`magi_runs/`、`ARCQuant/runs/`、校准文件及模型/数据入口保留原位。`cloudcli-deploy/`、`cloudcli-portal/`、`h20-2/` 和隐藏环境目录用于正在运行的服务。

本次清理释放 **307,143,474 字节（约 293 MiB）**：可核验的重复下载分片、与保留压缩包内容一致的安装暂存目录、旧环境迁移压缩包、重复说明和空测试目录。MiniMax 离线包移动后已重新通过 SHA256 校验。保留了存在差异的安装源码暂存目录，原始实验数据与运行依赖路径均未改变。

该整理只改变本机工作区的导航、附件位置与冗余安装文件；在其他机器克隆仓库后，仍从 `README.md` 和 `reports/verified/` 开始阅读。模型和原始实验产物按资产与迁移指南准备。
