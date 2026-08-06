# Phase 05-quant-deploy

## 目标
合并/imatrix量化/CPU部署实测

## 做法

补充 `deployment/llama_cpp/README.md`、`scripts/benchmark_llama_cpp.py` 和 `scripts/cost_latency_report.py`，用于合并权重→GGUF Q4_K_M→CPU llama-server→完整冻结集实测。

## 实际记录
| 日期 | 事项 | 结果/数字 | 备注 |
|---|---|---|---|
| 2026-08-06 | CPU 部署环境检查 | 未检测到 `llama-server` / GGUF 文件 | 未伪造 CPU 延迟；待安装 llama.cpp 后按文档实测 |

## 踩坑与学到的

GPU INT8 P50 不代表 CPU GGUF 性能，二者不能混用作成本或延迟结论。
