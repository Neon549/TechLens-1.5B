"""把实测 scorecard 与显式价格假设写成可审计的成本/延迟报告。"""
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--cloud-input-tokens", type=int, required=True)
    parser.add_argument("--cloud-output-tokens", type=int, required=True)
    parser.add_argument("--cloud-input-usd-per-mtok", type=float, required=True)
    parser.add_argument("--cloud-output-usd-per-mtok", type=float, required=True)
    parser.add_argument("--local-host-usd-per-hour", type=float, required=True)
    parser.add_argument("--out", default="reports/cpu_cost_latency.md")
    args = parser.parse_args()
    card = json.loads(Path(args.scorecard).read_text(encoding="utf-8"))
    p50 = card["speed"]["latency_p50_s"]
    cloud_per_request = (args.cloud_input_tokens * args.cloud_input_usd_per_mtok
                         + args.cloud_output_tokens * args.cloud_output_usd_per_mtok) / 1_000_000
    local_per_request = args.local_host_usd_per_hour * p50 / 3600 if p50 is not None else None
    local_one = f"${local_per_request:.6f}" if local_per_request is not None else "n/a"
    local_total = f"${local_per_request * args.requests:.2f}" if local_per_request is not None else "n/a"
    report = f"""# CPU 部署成本与延迟报告

## 可复现输入

- scorecard: `{args.scorecard}`
- 请求量: {args.requests:,}
- 云端 token 假设: input={args.cloud_input_tokens}, output={args.cloud_output_tokens}
- 云端单价假设: input=${args.cloud_input_usd_per_mtok}/MTok, output=${args.cloud_output_usd_per_mtok}/MTok
- 本地主机成本假设: ${args.local_host_usd_per_hour}/hour

## 实测与估算

| 指标 | 数值 |
|---|---:|
| CPU P50 延迟 | {p50} s |
| CPU P95 延迟 | {card['speed']['latency_p95_s']} s |
| 云端单请求成本（估算） | ${cloud_per_request:.6f} |
| 本地单请求成本（按 P50 摊销） | {local_one} |
| {args.requests:,} 次云端成本（估算） | ${cloud_per_request * args.requests:.2f} |
| {args.requests:,} 次本地成本（估算） | {local_total} |

价格是假设而非模型测量值；更换供应商、硬件利用率或并发度后必须重跑本报告。
"""
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)
