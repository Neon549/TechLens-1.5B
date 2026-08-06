# GGUF CPU 部署

当前仓库已包含合并的 Hugging Face 权重，但本机未检测到 `llama-server` 或 GGUF 文件，因此没有伪造 CPU 延迟结果。以下命令在安装了与模型架构兼容的 llama.cpp 后执行。

```powershell
# 1. 从合并权重导出并量化（路径按本机 llama.cpp 调整）
python D:/tools/llama.cpp/convert_hf_to_gguf.py models/merged/techlens-1.7b --outfile models/gguf/techlens-f16.gguf --outtype f16
D:/tools/llama.cpp/llama-quantize models/gguf/techlens-f16.gguf models/gguf/techlens-q4_k_m.gguf Q4_K_M

# 2. 在纯 CPU 模式启动；-c 应覆盖系统提示词、三段工具结果和输出上限
D:/tools/llama.cpp/llama-server -m models/gguf/techlens-q4_k_m.gguf -ngl 0 -c 4096 --port 8080

# 3. 对完整冻结集测量，不要传 --limit
python scripts/benchmark_llama_cpp.py --eval data/eval/real_test.jsonl --out experiments/m4_gguf_cpu
```

量化前先运行 `python scripts/freeze_real_eval.py` 得到真实行情冻结集。随后使用 `scorecard.json` 生成成本报告：

```powershell
python scripts/cost_latency_report.py --scorecard experiments/m4_gguf_cpu/scorecard.json --cloud-input-tokens 1200 --cloud-output-tokens 150 --cloud-input-usd-per-mtok <rate> --cloud-output-usd-per-mtok <rate> --local-host-usd-per-hour <rate>
```

将命令、CPU 型号、线程数、RAM、GGUF 文件哈希、P50/P95 及所有价格假设写入 `project-log/phase-05-quant-deploy/log.md`。
