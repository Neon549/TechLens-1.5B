# -*- coding: utf-8 -*-
"""评估入口。
python run_eval.py --backend mock-gold    # 链路自测
python run_eval.py --backend mock-noisy   # scorer灵敏度
python run_eval.py --backend llama --url http://127.0.0.1:8080 --out ../experiments/m0_base
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.inference.backends import create_backend
from techlens.evaluation.runner import run_eval, render_scorecard

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock-gold")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.backend == "mock-gold":
        backend = create_backend({"kind": "mock", "mode": "gold"})
    elif args.backend == "mock-noisy":
        backend = create_backend({"kind": "mock", "mode": "noisy", "noise_rate": 0.35})
    else:
        backend = create_backend({"kind": "llama_server", "base_url": args.url})
    root = Path(__file__).resolve().parents[1]
    card, _ = run_eval(root / "data" / "eval" / "test.jsonl", backend, out_dir=args.out, limit=args.limit)
    print(render_scorecard(card))
