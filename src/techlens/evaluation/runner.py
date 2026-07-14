# -*- coding: utf-8 -*-
"""评估runner：冻结测试集 × backend → 判分 → 多维度分数卡（含按tag细分）。"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from techlens.evaluation.scorers.all_scorers import QUALITY_SCORERS, score_speed
from techlens.prompts.template import build_system_prompt, serialize_input
from techlens.schemas.output import parse_model_output


def run_eval(test_jsonl, backend, out_dir=None, limit=None):
    samples = [json.loads(l) for l in Path(test_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        samples = samples[:limit]
    system = build_system_prompt()

    records = []
    for sample in samples:
        user = serialize_input(sample["input"], sample["stock_code"])
        gen = backend.generate(system, user, meta={"expected": sample["expected"]})
        obj, parse_err, fenced = parse_model_output(gen["text"])
        scores = {n: fn(obj, parse_err, fenced, sample) for n, fn in QUALITY_SCORERS.items()}
        scores["speed"] = score_speed(obj, parse_err, fenced, sample, gen)
        records.append({"id": sample["id"], "tags": sample.get("tags", []),
                        "task_type": sample.get("task_type"),
                        "raw_output": gen["text"], "scores": scores})

    card = build_scorecard(records, backend.name, len(samples))
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
        (out_dir / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / "scorecard.txt").write_text(render_scorecard(card), encoding="utf-8")
    return card, records


def _rate(records, key, field="pass"):
    vals = [r["scores"][key].get(field) for r in records
            if not r["scores"][key].get("skip") and field in r["scores"][key]]
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


def build_scorecard(records, model_name, n):
    lat = [r["scores"]["speed"]["latency_s"] for r in records if r["scores"]["speed"]["latency_s"]]
    field_accs = [r["scores"]["fields"].get("acc") for r in records
                  if not r["scores"]["fields"].get("skip") and r["scores"]["fields"].get("acc") is not None]
    invented = [r["scores"]["levels"].get("invented") for r in records
                if not r["scores"]["levels"].get("skip")]
    dangerous = [r["scores"]["restraint"].get("dangerous") for r in records]

    tag_stats = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in records:
        for tag in r["tags"]:
            tag_stats[tag]["n"] += 1
            tag_stats[tag]["ok"] += int(bool(r["scores"]["status"]["pass"]))
    by_tag = {t: {"n": v["n"], "status_acc": round(v["ok"] / v["n"], 3)}
              for t, v in sorted(tag_stats.items()) if v["n"] >= 3}

    return {
        "model": model_name, "n": n,
        "quality": {
            "instruction_following": _rate(records, "instruction"),
            "status_accuracy": _rate(records, "status"),
            "copy_fidelity": _rate(records, "fidelity"),
            "fields_exact": _rate(records, "fields"),
            "fields_acc_avg": round(statistics.mean(field_accs), 4) if field_accs else None,
            "levels_discipline": _rate(records, "levels"),
            "invented_levels_rate": round(sum(bool(x) for x in invented) / len(invented), 4) if invented else None,
            "restraint": _rate(records, "restraint"),
            "dangerous_analysis_rate": round(sum(bool(x) for x in dangerous) / len(dangerous), 4) if dangerous else None,
        },
        "speed": {
            "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
            "latency_p95_s": round(sorted(lat)[int(len(lat) * 0.95) - 1], 3) if len(lat) >= 20 else None,
        },
        "by_tag_status_acc": by_tag,
    }


def render_scorecard(card) -> str:
    q, s = card["quality"], card["speed"]
    def pct(x):
        return f"{x*100:.1f}%" if x is not None else "  n/a"
    lines = [
        f"===== TechLens 评估分数卡  (model={card['model']} / n={card['n']}) =====",
        "质量角度",
        f"  格式遵循              {pct(q['instruction_following'])}",
        f"  OK/ABORT决策          {pct(q['status_accuracy'])}",
        f"  复制保真度            {pct(q['copy_fidelity'])}   (KDJ三值+代码逐字一致)",
        f"  分类字段全对率        {pct(q['fields_exact'])}   (字段级均值 {q['fields_acc_avg']})",
        f"  价位纪律              {pct(q['levels_discipline'])}   (编造价位率 {pct(q['invented_levels_rate'])})",
        f"  克制与安全            {pct(q['restraint'])}   (危险分析率 {pct(q['dangerous_analysis_rate'])})",
        "速度角度",
        f"  延迟 P50 {s['latency_p50_s']}s | P95 {s['latency_p95_s']}s",
        "按tag细分（status准确率，n>=3）",
    ]
    for tag, v in card["by_tag_status_acc"].items():
        lines.append(f"  {tag:<10} n={v['n']:<4} acc={v['status_acc']*100:.1f}%")
    return "\n".join(lines)
