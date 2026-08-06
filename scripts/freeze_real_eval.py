"""下载、规则标注并冻结一个真实 A 股行情评估集。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.datagen.real_eval import DEFAULT_SYMBOLS, freeze_stratified_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval/real_test_v2.jsonl")
    parser.add_argument("--per-ok-type", type=int, default=24)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    args = parser.parse_args()
    stats = freeze_stratified_eval(args.out, per_ok_type=args.per_ok_type,
                                   symbols=tuple(args.symbols), start=args.start, end=args.end)
    print(f"frozen real evaluation set: {args.out} | {stats}")
