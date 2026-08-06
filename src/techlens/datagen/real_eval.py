"""从 Yahoo Finance 的公开日线构建真实行情冻结评估集。

不将下载的行情混入训练集；每个样本保留来源、交易所代码和截断日期，便于复现。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from techlens.datagen.engine import calc_indicators, derive_label, render_tool_results


DEFAULT_SYMBOLS = (
    "000001.SZ", "000333.SZ", "000858.SZ", "002594.SZ", "300750.SZ",
    "600036.SS", "600519.SS", "601318.SS", "601888.SS", "603259.SS",
)


def _stock_code(symbol: str) -> str:
    return symbol.split(".", 1)[0]


def download_daily(symbol: str, start: str, end: str) -> list[dict]:
    """下载未经调整的公开日线；调用方负责把结果冻结到仓库。"""
    period1 = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp())
    period2 = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp())
    query = urllib.parse.urlencode({"period1": period1, "period2": period2,
                                    "interval": "1d", "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "TechLens-eval/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    rows = []
    for i, timestamp in enumerate(result["timestamp"]):
        values = {key: quote[key][i] for key in ("open", "high", "low", "close", "volume")}
        if any(value is None for value in values.values()) or values["volume"] < 0:
            continue
        rows.append({"date": datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat(), **values})
    return rows


def _to_sample(symbol: str, rows: list[dict], end_index: int) -> dict:
    window = rows[end_index - 89:end_index + 1]
    if len(window) != 90:
        raise ValueError("real evaluation windows require 90 trading days")
    kline = {field: np.asarray([row[field] for row in window])
             for field in ("open", "high", "low", "close", "volume")}
    kline["has_edge"] = any(
        row["volume"] == 0 or abs(row["close"] / window[i - 1]["close"] - 1) >= 0.095
        for i, row in enumerate(window) if i
    )
    code = _stock_code(symbol)
    indicators = calc_indicators(kline)
    expected = derive_label(kline, indicators, code)
    tools = render_tool_results(kline, indicators, code, dates=[row["date"] for row in window])
    if kline["has_edge"]:
        task_type = "ok_edge"
    elif "暂不设定" in (expected["support"], expected["resistance"]):
        task_type = "ok_no_levels"
    else:
        task_type = f"ok_{expected['trend']}"
    tags = ["real_market", task_type.removeprefix("ok_"), symbol]
    if "暂不设定" in (expected["support"], expected["resistance"]):
        tags.append("价位克制")
    return {"task_type": task_type, "stock_code": code, "input": tools, "expected": expected,
            "tags": tags, "source": {"provider": "Yahoo Finance", "symbol": symbol,
                                        "window_end": window[-1]["date"], "adjusted": False}}


def collect_candidates(symbols=DEFAULT_SYMBOLS, start="2018-01-01", end="2026-01-01") -> list[dict]:
    candidates = []
    for symbol in symbols:
        rows = download_daily(symbol, start, end)
        # 每月取一个窗口，降低高度相邻样本造成的伪重复。
        seen_months = set()
        for end_index in range(89, len(rows)):
            month = rows[end_index]["date"][:7]
            if month not in seen_months:
                candidates.append(_to_sample(symbol, rows, end_index))
                seen_months.add(month)
        time.sleep(0.2)
    return candidates


def freeze_stratified_eval(output_path: str | Path, *, per_ok_type=24,
                           symbols=DEFAULT_SYMBOLS, start="2018-01-01", end="2026-01-01") -> dict:
    """按真实行情类别均衡抽样；若类别不足则拒绝写出不平衡的评估集。"""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refuse to overwrite frozen evaluation set: {output_path}")
    buckets = defaultdict(list)
    for sample in collect_candidates(symbols, start, end):
        buckets[sample["task_type"]].append(sample)
    required = ("ok_bullish", "ok_bearish", "ok_neutral", "ok_no_levels", "ok_edge")
    short = {name: len(buckets[name]) for name in required if len(buckets[name]) < per_ok_type}
    if short:
        raise RuntimeError(f"insufficient real-market candidates: {short}; add symbols or extend date range")
    rows = []
    for task_type in required:
        # 按股票 round-robin 取样，避免某只股票的连续月份主导一个类别。
        by_symbol = defaultdict(list)
        for sample in sorted(buckets[task_type], key=lambda x: (x["source"]["symbol"], x["source"]["window_end"])):
            by_symbol[sample["source"]["symbol"]].append(sample)
        chosen, index = [], 0
        while len(chosen) < per_ok_type:
            progressed = False
            for symbol in sorted(by_symbol):
                if index < len(by_symbol[symbol]):
                    chosen.append(by_symbol[symbol][index])
                    progressed = True
                    if len(chosen) == per_ok_type:
                        break
            if not progressed:
                raise RuntimeError(f"unable to select {per_ok_type} diverse samples for {task_type}")
            index += 1
        rows.extend(chosen)
    for index, row in enumerate(rows):
        row["id"] = f"real-{row['task_type']}-{index:04d}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return {name: per_ok_type for name in required}
