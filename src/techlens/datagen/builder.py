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


def _targeted_rejected(sample: dict, field: str) -> dict:
    """构造与已观测错误同型的 rejected，不引用评估样本的内容。"""
    out = copy.deepcopy(sample["expected"])
    if field == "volume_price":
        alternatives = ["放量上涨", "缩量上涨", "放量下跌", "缩量下跌", "量价平稳"]
        # 真实错误中模型常回退为“量价平稳”；其余情况选该回退值。
        out["volume_price"] = "量价平稳" if out["volume_price"] != "量价平稳" else "放量上涨"
    elif field == "confidence":
        alternatives = ["high", "medium", "low"]
        out["confidence"] = next(value for value in alternatives if value != out["confidence"])
    else:
        raise ValueError(f"unsupported targeted field: {field}")
    return out


def build_targeted_dpo_pairs(clean_jsonl, error_fields: dict[str, int], count=400) -> list[dict]:
    """由聚合错误画像选择 DPO 扰动类型，避免从评估集复制输入。"""
    samples = [json.loads(line) for line in Path(clean_jsonl).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    ok_samples = [sample for sample in samples if sample["expected"]["status"] == "OK"]
    supported = {field: n for field, n in error_fields.items()
                 if field in {"volume_price", "confidence"} and n > 0}
    if not supported:
        raise ValueError("error profile has no supported field failures")
    schedule = []
    total = sum(supported.values())
    for field, failures in sorted(supported.items()):
        schedule.extend([field] * round(count * failures / total))
    while len(schedule) < count:
        schedule.append(max(supported, key=supported.get))
    schedule = schedule[:count]

    pairs = []
    for index, field in enumerate(schedule):
        sample = ok_samples[index % len(ok_samples)]
        rejected = _targeted_rejected(sample, field)
        pairs.append({
            "instruction": build_system_prompt(),
            "input": serialize_input(sample["input"], sample["stock_code"]),
            "chosen": json.dumps(sample["expected"], ensure_ascii=False),
            "rejected": json.dumps(rejected, ensure_ascii=False),
            "metadata": {"targeted_field": field, "source": "train_only"},
        })
    return pairs


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
        "techlens_dpo_targeted": {"file_name": "dpo_targeted_train.json", "ranking": True,
                                   "columns": {"prompt": "instruction", "query": "input",
                                               "chosen": "chosen", "rejected": "rejected"}},
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"sft": len(sft), "dpo": len(pairs)}
