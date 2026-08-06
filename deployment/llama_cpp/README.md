# GGUF CPU 部署

已使用 llama.cpp b10293 在 Windows x64 CPU 上验证转换、量化和服务启动。实际 120 条真实行情结果见 `reports/m4_cpu_benchmark.md`。转换时不改动原始模型，而是在暂存副本中移除与当前 Transformers 不兼容的 `extra_special_tokens` 列表；该变化有记录且权重以硬链接复用。

```powershell
# 1. 准备不改原模型的转换暂存目录，导出并量化
python scripts/prepare_gguf_export.py
python tools/llama.cpp-src/convert_hf_to_gguf.py models/gguf/techlens-convert-staging --outfile models/gguf/techlens-f16.gguf --outtype f16
tools/llama.cpp-bin/llama-quantize.exe models/gguf/techlens-f16.gguf models/gguf/techlens-q4_k_m.gguf Q4_K_M

# 2. 在纯 CPU 模式启动；-c 应覆盖系统提示词、三段工具结果和输出上限
tools/llama.cpp-bin/llama-server.exe -m models/gguf/techlens-q4_k_m.gguf -ngl 0 -t 24 -c 4096 --jinja --host 127.0.0.1 --port 8080

# 3. 对完整冻结集测量，不要传 --limit
python scripts/benchmark_llama_cpp.py --eval data/eval/real_test.jsonl --out experiments/m4_gguf_cpu
```

量化前先运行 `python scripts/freeze_real_eval.py` 得到真实行情冻结集。随后使用 `scorecard.json` 生成成本报告：

```powershell
python scripts/cost_latency_report.py --scorecard experiments/m4_gguf_cpu/scorecard.json --cloud-input-tokens 1200 --cloud-output-tokens 150 --cloud-input-usd-per-mtok <rate> --cloud-output-usd-per-mtok <rate> --local-host-usd-per-hour <rate>
```

将命令、CPU 型号、线程数、RAM、GGUF 文件哈希、P50/P95 及所有价格假设写入 `project-log/phase-05-quant-deploy/log.md`。
