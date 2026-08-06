"""对运行中的 CPU llama-server 做完整冻结集评估并写出分数卡。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.evaluation.runner import render_scorecard, run_eval
from techlens.inference.backends import LlamaServerBackend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="techlens")
    parser.add_argument("--eval", default="data/eval/real_test_v2.jsonl")
    parser.add_argument("--out", default="experiments/m4_gguf_cpu")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    eval_path = Path(args.eval)
    if not eval_path.exists():
        raise FileNotFoundError(f"frozen evaluation set not found: {eval_path}; run freeze_real_eval.py first")
    backend = LlamaServerBackend(base_url=args.url, model=args.model, timeout=args.timeout)
    card, _ = run_eval(eval_path, backend, out_dir=args.out)
    print(render_scorecard(card))
