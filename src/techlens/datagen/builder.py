# -*- coding: utf-8 -*-
"""训练集构造：SFT alpaca + DPO偏好对（程序化定向扰动）。
DPO扰动对齐痛点：编造价位(主力40%)、KDJ值漂移(25%)、该ABORT硬分析(25%)、markdown格式(10%)。
"""
import copy
import json
import random
from pathlib import Path

from techlens.prompts.template import build_system_prompt, serialize_input


def to_sft(sample: dict) -> dict:
    return {
        "instruction": build_system_prompt(),
        "input": serialize_input(sample["input"], sample["stock_code"]),
        "output": json.dumps(sample["expected"], ensure_ascii=False),
    }


def _perturb(sample: dict, rng: random.Random):
    out = copy.deepcopy(sample["expected"])
    r = rng.random()
    if out["status"] == "OK":
        if r < 0.45:  # 编造价位（最核心痛点）
            target = rng.choice(["support", "resistance"])
            base = out["kdj"]["K"] + 20
            out[target] = round(base * rng.uniform(0.8, 1.2), 2)
            return out
        if r < 0.75:  # KDJ复制漂移
            f = rng.choice(["K", "D", "J"])
            out["kdj"][f] = round(out["kdj"][f] + rng.uniform(0.5, 5), 2)
            return out
        if r < 0.9:  # 方向判反
            out["trend"] = {"bullish": "bearish", "bearish": "bullish",
                            "neutral": rng.choice(["bullish", "bearish"])}[out["trend"]]
            return out
        return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"  # 格式扰动
    else:  # ABORT样本 → 该中止却硬分析（危险行为）
        return {"status": "OK", "stock_code": sample["stock_code"], "trend": "neutral",
                "volume_price": "量价平稳", "support": "暂不设定", "resistance": "暂不设定",
                "kdj": {"K": 50.0, "D": 50.0, "J": 50.0, "signal": "waiting"},
                "confidence": "low", "summary": "基于有限数据的分析结论。"}


def build_dpo_pair(sample: dict, rng: random.Random) -> dict | None:
    rejected = _perturb(sample, rng)
    rejected_str = rejected if isinstance(rejected, str) else json.dumps(rejected, ensure_ascii=False)
    chosen_str = json.dumps(sample["expected"], ensure_ascii=False)
    if rejected_str == chosen_str:
        return None
    return {
        "instruction": build_system_prompt(),
        "input": serialize_input(sample["input"], sample["stock_code"]),
        "chosen": chosen_str, "rejected": rejected_str,
    }


def build_datasets(clean_jsonl, out_dir, dpo_count=400, seed=42):
    rng = random.Random(seed)
    samples = [json.loads(l) for l in Path(clean_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft = [to_sft(s) for s in samples]
    (out_dir / "sft_train.json").write_text(json.dumps(sft, ensure_ascii=False, indent=1), encoding="utf-8")

    # DPO优先：价位克制类 > abort类 > 其他
    def priority(s):
        if "价位克制" in s.get("tags", []):
            return 0
        if s["expected"]["status"] == "ABORT":
            return 1
        return 2
    pairs = []
    for s in sorted(samples, key=priority):
        if len(pairs) >= dpo_count:
            break
        p = build_dpo_pair(s, rng)
        if p:
            pairs.append(p)
    (out_dir / "dpo_train.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    info = {
        "techlens_sft": {"file_name": "sft_train.json",
                         "columns": {"prompt": "instruction", "query": "input", "response": "output"}},
        "techlens_dpo": {"file_name": "dpo_train.json", "ranking": True,
                         "columns": {"prompt": "instruction", "query": "input",
                                     "chosen": "chosen", "rejected": "rejected"}},
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"sft": len(sft), "dpo": len(pairs)}
