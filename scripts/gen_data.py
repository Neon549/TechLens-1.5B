# -*- coding: utf-8 -*-
"""程序化生成全部数据（无需API key，本地秒级完成）。
用法: python gen_data.py            # 按默认配比生成 raw + 冻结评估集 + 训练集
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.datagen.engine import generate_sample
from techlens.datagen.builder import build_datasets

# 配比：价位克制和abort是DPO重点，给足量；总计~2000
PLAN = {"ok_bullish": 400, "ok_bearish": 400, "ok_neutral": 350,
        "ok_no_levels": 350, "ok_edge": 200, "abort_tool_error": 180, "abort_insufficient": 120}
EVAL_PER_TYPE = 30  # 每类切30条进冻结评估集

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "all_samples.jsonl"
    eval_path = root / "data" / "eval" / "test.jsonl"
    clean_path = root / "data" / "train" / "clean.jsonl"

    if eval_path.exists():
        print(f"REFUSE: {eval_path} 已存在，评估集冻结后不可覆盖。如确要重建请手动删除。")
        sys.exit(1)

    train_rows, eval_rows = [], []
    seed = 0
    for t, count in PLAN.items():
        for i in range(count):
            s = generate_sample(t, seed=seed)
            seed += 1
            if i < EVAL_PER_TYPE:
                eval_rows.append({"id": f"{t}-{i:04d}", **s})
            else:
                train_rows.append(s)
        print(f"[{t}] {count} generated")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in train_rows + eval_rows), encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in eval_rows), encoding="utf-8")
    clean_path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in train_rows), encoding="utf-8")

    stats = build_datasets(clean_path, root / "data" / "train")
    print(f"\nfrozen eval: {len(eval_rows)} | train: {len(train_rows)}")
    print(f"sft samples: {stats['sft']} | dpo pairs: {stats['dpo']}")
