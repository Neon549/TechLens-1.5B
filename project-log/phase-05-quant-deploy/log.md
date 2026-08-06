# Phase 05-quant-deploy

## 目标
合并/imatrix量化/CPU部署实测

## 做法

补充 `deployment/llama_cpp/README.md`、`scripts/benchmark_llama_cpp.py` 和 `scripts/cost_latency_report.py`，用于合并权重→GGUF Q4_K_M→CPU llama-server→完整冻结集实测。

## 实际记录
| 日期 | 事项 | 结果/数字 | 备注 |
|---|---|---|---|
| 2026-08-06 | llama.cpp 安装与转换 | b10293，Q4_K_M 1,107,408,576 B | F16 3,447,348,928 B；SHA256 49D376295055481BC842B78E7181F45A53AEAEEF0D95AA2A6F9BF89BD7F466CA |
| 2026-08-06 | 纯 CPU 真实集评估 | n=120；P50 5.720s；P95 8.753s | Intel Core Ultra 9 275HX，24 线程，`-ngl 0`，llama.cpp b10293 |
| 2026-08-06 | 质量实测 | 格式/决策/KDJ 100%；字段全对 26.7%；价位纪律 75.8% | Q4_K_M 质量未达生产替换阈值 |

## 踩坑与学到的

GPU INT8 P50 不代表 CPU GGUF 性能，二者不能混用作成本或延迟结论。实际 Q4_K_M 在真实数据上出现分类与价位纪律退化，量化后必须重新评估，不能只复用 BF16 分数。
