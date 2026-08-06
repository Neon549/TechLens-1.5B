# -*- coding: utf-8 -*-
"""模型输出的结构校验。合成、评估、DPO 与服务端全部复用。"""
import json
from pathlib import Path
from functools import lru_cache

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "analysis_schema.json"
TRENDS = {"bullish", "bearish", "neutral"}
VOLUME_PRICE = {"放量上涨", "缩量上涨", "放量下跌", "缩量下跌", "量价平稳"}
SIGNALS = {"buy_ready", "waiting"}
CONFIDENCE = {"high", "medium", "low"}
OK_FIELDS = {"status", "stock_code", "trend", "volume_price", "support",
             "resistance", "kdj", "confidence", "summary"}
ABORT_FIELDS = {"status", "reason"}
TOOL_REQUEST_FIELDS = {"status", "tool_name", "arguments", "reason"}
TOOLS = {"get_stock_history", "get_stock_price", "get_kdj_signal"}


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def parse_model_output(text: str):
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    stripped = text.strip()
    fenced = False
    if stripped.startswith("```"):
        fenced = True
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        return json.loads(stripped), None, fenced
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {e}", fenced


def _is_price(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 10000


def validate_output(obj) -> list[str]:
    errors = []
    if not isinstance(obj, dict):
        return ["output_not_dict"]
    status = obj.get("status")
    if status not in ("OK", "ABORT", "TOOL_REQUEST"):
        return [f"invalid_status:{status!r}"]

    if status == "ABORT":
        if not isinstance(obj.get("reason"), str) or not obj.get("reason"):
            errors.append("abort_reason_empty")
        extra = set(obj) - ABORT_FIELDS
        if extra:
            errors.append(f"extra_fields:{sorted(extra)}")
        return errors

    if status == "TOOL_REQUEST":
        if obj.get("tool_name") not in TOOLS:
            errors.append(f"bad_tool_name:{obj.get('tool_name')!r}")
        arguments = obj.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != {"stock_code"}:
            errors.append("tool_request_arguments_must_be_stock_code")
        elif not (isinstance(arguments["stock_code"], str)
                  and len(arguments["stock_code"]) == 6
                  and arguments["stock_code"].isdigit()):
            errors.append("tool_request_bad_stock_code")
        if not isinstance(obj.get("reason"), str) or not obj["reason"]:
            errors.append("tool_request_reason_empty")
        extra = set(obj) - TOOL_REQUEST_FIELDS
        if extra:
            errors.append(f"extra_fields:{sorted(extra)}")
        return errors

    # status == OK
    code = obj.get("stock_code")
    if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
        errors.append(f"bad_stock_code:{code!r}")
    if obj.get("trend") not in TRENDS:
        errors.append(f"bad_trend:{obj.get('trend')!r}")
    if obj.get("volume_price") not in VOLUME_PRICE:
        errors.append(f"bad_volume_price:{obj.get('volume_price')!r}")
    for lv in ("support", "resistance"):
        v = obj.get(lv)
        if v != "暂不设定" and not _is_price(v):
            errors.append(f"bad_{lv}:{v!r}")
    kdj = obj.get("kdj")
    if not isinstance(kdj, dict):
        errors.append("kdj_not_dict")
    else:
        for f in ("K", "D", "J"):
            if not isinstance(kdj.get(f), (int, float)) or isinstance(kdj.get(f), bool):
                errors.append(f"kdj_{f}_not_number")
        if kdj.get("signal") not in SIGNALS:
            errors.append(f"bad_signal:{kdj.get('signal')!r}")
        extra_k = set(kdj) - {"K", "D", "J", "signal"}
        if extra_k:
            errors.append(f"kdj_extra_fields:{sorted(extra_k)}")
    if obj.get("confidence") not in CONFIDENCE:
        errors.append(f"bad_confidence:{obj.get('confidence')!r}")
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary:
        errors.append("summary_empty")
    elif len(summary) > 120:
        errors.append("summary_too_long")
    extra = set(obj) - OK_FIELDS
    if extra:
        errors.append(f"extra_fields:{sorted(extra)}")
    return errors
