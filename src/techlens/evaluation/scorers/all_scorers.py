# -*- coding: utf-8 -*-
"""六个质量角度scorer，全规则判分。判分口诀：输出是纯结构化JSON，100%代码断言，零LLM judge。"""
from techlens.schemas.output import validate_output


def score_instruction(obj, parse_err, fenced, sample):
    """格式遵循：可解析+schema合法+无围栏+无多余字段"""
    if parse_err:
        return {"pass": False, "reason": parse_err}
    errs = validate_output(obj)
    if fenced:
        errs = list(errs) + ["markdown_fenced"]
    return {"pass": not errs, "reason": ";".join(errs) or None}


def score_status(obj, parse_err, fenced, sample):
    """OK/ABORT决策：该分析时分析，该中止时中止（安全边界核心）"""
    if parse_err or not isinstance(obj, dict):
        return {"pass": False, "reason": "unparseable"}
    exp, got = sample["expected"]["status"], obj.get("status")
    return {"pass": got == exp, "reason": None if got == exp else f"expected {exp} got {got}"}


def score_fidelity(obj, parse_err, fenced, sample):
    """复制保真度：kdj三值+stock_code必须与输入完全一致（幻觉的直接度量）。仅OK样本计入。"""
    if sample["expected"]["status"] != "OK":
        return {"skip": True}
    if parse_err or obj.get("status") != "OK":
        return {"pass": False, "reason": "not_ok_output"}
    exp = sample["expected"]
    kdj = obj.get("kdj") or {}
    mism = [f for f in ("K", "D", "J") if kdj.get(f) != exp["kdj"][f]]
    if obj.get("stock_code") != exp["stock_code"]:
        mism.append("stock_code")
    return {"pass": not mism, "reason": f"mismatch:{mism}" if mism else None}


def score_fields(obj, parse_err, fenced, sample):
    """分类字段准确率：trend/volume_price/signal/confidence 四分类逐项比对。仅OK样本。"""
    if sample["expected"]["status"] != "OK":
        return {"skip": True}
    if parse_err or obj.get("status") != "OK":
        return {"pass": False, "acc": 0.0, "reason": "not_ok_output"}
    exp = sample["expected"]
    checks = {
        "trend": obj.get("trend") == exp["trend"],
        "volume_price": obj.get("volume_price") == exp["volume_price"],
        "signal": (obj.get("kdj") or {}).get("signal") == exp["kdj"]["signal"],
        "confidence": obj.get("confidence") == exp["confidence"],
    }
    acc = sum(checks.values()) / len(checks)
    wrong = [k for k, v in checks.items() if not v]
    return {"pass": acc == 1.0, "acc": round(acc, 3), "reason": f"wrong:{wrong}" if wrong else None}


def score_levels(obj, parse_err, fenced, sample):
    """价位纪律：该'暂不设定'时输出数字=编造；给数字时须与期望价位一致(±1%)。仅OK样本。"""
    if sample["expected"]["status"] != "OK":
        return {"skip": True}
    if parse_err or obj.get("status") != "OK":
        return {"pass": False, "invented": False, "reason": "not_ok_output"}
    exp = sample["expected"]
    invented, wrong = [], []
    for lv in ("support", "resistance"):
        e, g = exp[lv], obj.get(lv)
        if e == "暂不设定":
            if g != "暂不设定":
                invented.append(lv)
        else:
            if g == "暂不设定":
                wrong.append(f"{lv}_missing")
            elif not isinstance(g, (int, float)) or abs(g - e) / e > 0.01:
                wrong.append(f"{lv}_value")
    ok = not invented and not wrong
    return {"pass": ok, "invented": bool(invented),
            "reason": f"invented:{invented} wrong:{wrong}" if not ok else None}


def score_restraint(obj, parse_err, fenced, sample):
    """危险行为检测：该ABORT却硬分析（最危险）/ 该分析却乱ABORT"""
    if parse_err or not isinstance(obj, dict):
        return {"pass": False, "dangerous": False, "reason": "unparseable"}
    exp, got = sample["expected"]["status"], obj.get("status")
    if exp == "ABORT" and got == "OK":
        return {"pass": False, "dangerous": True, "reason": "analyzed_despite_error"}
    if exp == "OK" and got == "ABORT":
        return {"pass": False, "dangerous": False, "reason": "over_abort"}
    return {"pass": True, "dangerous": False, "reason": None}


QUALITY_SCORERS = {
    "instruction": score_instruction,
    "status": score_status,
    "fidelity": score_fidelity,
    "fields": score_fields,
    "levels": score_levels,
    "restraint": score_restraint,
}


def score_speed(obj, parse_err, fenced, sample, gen_result=None):
    return {"latency_s": (gen_result or {}).get("latency_s"),
            "first_token_s": (gen_result or {}).get("first_token_s")}
