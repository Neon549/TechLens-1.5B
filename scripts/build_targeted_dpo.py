"""由错误画像构建不含评估样本的定向 DPO 数据。"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.datagen.builder import build_targeted_dpo_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--train", default="data/train/clean.jsonl")
    parser.add_argument("--out", default="data/train/dpo_targeted_train.json")
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    pairs = build_targeted_dpo_pairs(args.train, profile["field_failures"], args.count)
    Path(args.out).write_text(json.dumps(pairs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(pairs)} targeted pairs: {dict(Counter(p['metadata']['targeted_field'] for p in pairs))}")
