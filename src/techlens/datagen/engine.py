# -*- coding: utf-8 -*-
"""程序化数据引擎：K线模拟 + 指标计算 + 标准答案推导。
核心优势（面试点）：ground truth完全由规则推导，不依赖强模型蒸馏，
不存在教师模型幻觉污染；且与StockMind线上_calc_kdj_signal公式逐行一致。
"""
import random

import numpy as np

REGIMES = {
    "bullish": {"drift": 0.004, "vol": 0.018},
    "bearish": {"drift": -0.004, "vol": 0.02},
    "neutral": {"drift": 0.0, "vol": 0.012},
    "volatile": {"drift": 0.0, "vol": 0.035},
}


def simulate_kline(regime: str, days: int = 90, seed: int | None = None,
                   with_limit_days: bool = False) -> dict:
    """随机游走模拟A股日K线。返回OHLCV数组与元信息。"""
    rng = np.random.default_rng(seed)
    p = REGIMES[regime]
    base = rng.uniform(5, 80)
    rets = rng.normal(p["drift"], p["vol"], days)
    if with_limit_days:  # 注入1-2个涨/跌停日
        for idx in rng.choice(range(days // 2, days), size=rng.integers(1, 3), replace=False):
            rets[idx] = rng.choice([0.0995, -0.0995])
    closes = base * np.cumprod(1 + rets)
    opens = closes / (1 + rets) * (1 + rng.normal(0, 0.004, days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.008, days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.008, days)))
    vol_base = rng.uniform(2e5, 5e6)
    vols = (vol_base * (1 + rng.normal(0, 0.3, days)).clip(0.2)).astype(int)
    if with_limit_days and rng.random() < 0.5:  # 偶发零成交（停牌）日
        vols[rng.integers(days // 2, days)] = 0
    return {"open": opens, "close": closes, "high": highs, "low": lows,
            "volume": vols, "regime": regime, "has_edge": with_limit_days}


def calc_indicators(k: dict) -> dict:
    """KDJ与MA20，公式与StockMind technical_analyst._calc_kdj_signal逐行一致。"""
    import pandas as pd
    df = pd.DataFrame({c: k[c] for c in ["open", "close", "high", "low", "volume"]})
    low_min = df["low"].rolling(9).min()
    high_max = df["high"].rolling(9).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-10) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["MA20"] = df["close"].rolling(20).mean()
    latest, prev_ma = df.iloc[-1], df.iloc[-6]["MA20"]
    return {
        "K": round(float(latest["K"]), 2), "D": round(float(latest["D"]), 2),
        "J": round(float(latest["J"]), 2), "close": round(float(latest["close"]), 2),
        "MA20": round(float(latest["MA20"]), 2),
        "ma20_up": bool(float(latest["MA20"]) > float(prev_ma)),
        "df": df,
    }


def derive_label(k: dict, ind: dict, stock_code: str) -> dict:
    """按analysis_schema.json的label_rules推导标准答案（单一事实来源）。"""
    close, ma20, ma20_up = ind["close"], ind["MA20"], ind["ma20_up"]
    df = ind["df"]

    if close > ma20 and ma20_up:
        trend = "bullish"
    elif close < ma20 and not ma20_up:
        trend = "bearish"
    else:
        trend = "neutral"

    v_recent, v_prev = df["volume"].iloc[-5:].mean(), df["volume"].iloc[-10:-5].mean()
    price_up = df["close"].iloc[-1] > df["close"].iloc[-6]
    if v_recent > v_prev * 1.15:
        volume_price = "放量上涨" if price_up else "放量下跌"
    elif v_recent < v_prev * 0.85:
        volume_price = "缩量上涨" if price_up else "缩量下跌"
    else:
        volume_price = "量价平稳"

    low10 = round(float(df["low"].iloc[-10:].min()), 2)
    high10 = round(float(df["high"].iloc[-10:].max()), 2)
    support = low10 if (close - low10) / close <= 0.08 else "暂不设定"
    resistance = high10 if (high10 - close) / close <= 0.08 else "暂不设定"

    buy_ready = ind["K"] < 25 and ind["J"] < 15 and close > ma20 and ma20_up

    vol_confirms = (trend == "bullish" and "上涨" in volume_price) or \
                   (trend == "bearish" and "下跌" in volume_price)
    if k.get("has_edge"):
        confidence = "low"
    elif trend != "neutral" and vol_confirms:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "status": "OK", "stock_code": stock_code, "trend": trend,
        "volume_price": volume_price, "support": support, "resistance": resistance,
        "kdj": {"K": ind["K"], "D": ind["D"], "J": ind["J"],
                "signal": "buy_ready" if buy_ready else "waiting"},
        "confidence": confidence,
        "summary": _template_summary(trend, volume_price, buy_ready),
    }


def _template_summary(trend, volume_price, buy_ready) -> str:
    t = {"bullish": "趋势向好", "bearish": "趋势偏弱", "neutral": "方向不明"}[trend]
    s = "KDJ满足超卖买入条件" if buy_ready else "KDJ暂不满足买入条件"
    return f"{t}，{volume_price}，{s}。"


# ---------- 渲染成StockMind工具返回的文本格式（输入侧） ----------

def render_tool_results(k: dict, ind: dict, stock_code: str,
                        inject_tool_error: str | None = None,
                        insufficient: bool = False) -> dict:
    """渲染成与akshare_tools输出一致的文本。返回三段工具结果。"""
    df = ind["df"]
    n = len(df)
    dates = [f"2026-{4 + i // 22:02d}-{(i % 22) + 1:02d}" for i in range(n)]

    if inject_tool_error == "history":
        history = f"[TOOL_ERROR]\ntool=get_stock_history\nsymbol={stock_code}\nAKShare 与 yfinance 均未获取到历史数据"
    else:
        rows = []
        for i in range(max(0, n - 10), n):
            chg = (df['close'].iloc[i] / df['close'].iloc[i-1] - 1) * 100 if i > 0 else 0.0
            rows.append(f"{dates[i]}  {df['open'].iloc[i]:.2f}  {df['close'].iloc[i]:.2f}  "
                        f"{df['high'].iloc[i]:.2f}  {df['low'].iloc[i]:.2f}  "
                        f"{int(df['volume'].iloc[i])}  {chg:.2f}")
        history = (f"[TOOL_OK]\ntool=get_stock_history\nsymbol={stock_code}\n"
                   f"最近{n}天K线数据\n"
                   f"期间最高价：{df['high'].max():.2f}\n"
                   f"期间最低价：{df['low'].min():.2f}\n"
                   f"最新收盘价：{ind['close']:.2f}\n"
                   f"数据来源：AKShare/东方财富历史行情\n\n"
                   f"最近10日明细：\n日期  开盘  收盘  最高  最低  成交量  涨跌幅\n" + "\n".join(rows))

    if inject_tool_error == "price":
        price = f"[TOOL_ERROR]\ntool=get_stock_price\nsymbol={stock_code}\nAKShare 与 yfinance 均未获取到行情数据"
    else:
        chg = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
        price = (f"[TOOL_OK]\ntool=get_stock_price\nsymbol={stock_code}\n"
                 f"最新价：{ind['close']:.2f}\n涨跌幅：{chg:.2f}%\n"
                 f"成交量：{int(df['volume'].iloc[-1])}\n数据来源：AKShare实时行情")

    if insufficient:
        kdj_text = "KDJ数据：K线数据不足60根，无法可靠计算KDJ和MA60。"
    else:
        k_ok, j_ok = ind["K"] < 25, ind["J"] < 15
        above, up = ind["close"] > ind["MA20"], ind["ma20_up"]
        all_ok = k_ok and j_ok and above and up
        kdj_text = (
            f"数据日期：{dates[-1]}\n当前价格：¥{ind['close']}\n"
            f"MA20：¥{ind['MA20']}  {'📈 向上' if up else '📉 向下'}\n\n"
            f"KDJ超卖策略条件检测（需全部满足才触发买入）：\n"
            f"  K={ind['K']}  {'✅ 满足(K<25)' if k_ok else '❌ 不满足(需<25)'}\n"
            f"  D={ind['D']}  {'✅ 满足(D<30)' if ind['D'] < 30 else '❌ 不满足(需<30)'}\n"
            f"  J={ind['J']}  {'✅ 满足(J<15)' if j_ok else '❌ 不满足(需<15)'}\n"
            f"  价格在MA20上方  {'✅ 满足' if above else '❌ 不满足'}\n"
            f"  MA20向上  {'✅ 满足' if up else '❌ 不满足'}\n"
            f"综合信号：{'🟢 当前满足KDJ超卖买入条件！可关注买入机会。' if all_ok else '⏳ 当前不满足买入条件，需继续等待KDJ回落至超卖区域。'}"
        )

    return {"history": history, "price": price, "kdj": kdj_text}


def random_stock_code(rng: random.Random) -> str:
    prefix = rng.choice(["600", "601", "000", "002", "003", "300"])
    return prefix + f"{rng.randint(0, 999):03d}"


def generate_sample(task_type: str, seed: int) -> dict:
    """生成一条完整样本：{task_type, input:{...三段工具文本}, expected, tags}"""
    rng = random.Random(seed)
    stock_code = random_stock_code(rng)

    regime_map = {"ok_bullish": "bullish", "ok_bearish": "bearish",
                  "ok_neutral": "neutral", "ok_no_levels": "volatile",
                  "ok_edge": rng.choice(["bullish", "bearish"])}

    if task_type.startswith("ok"):
        k = simulate_kline(regime_map[task_type], days=90, seed=seed,
                           with_limit_days=(task_type == "ok_edge"))
        ind = calc_indicators(k)
        # ok_no_levels：重采样直到至少一个价位是'暂不设定'
        if task_type == "ok_no_levels":
            tries = 0
            expected = derive_label(k, ind, stock_code)
            while expected["support"] != "暂不设定" and expected["resistance"] != "暂不设定" and tries < 20:
                seed += 100000
                k = simulate_kline("volatile", days=90, seed=seed)
                ind = calc_indicators(k)
                expected = derive_label(k, ind, stock_code)
                tries += 1
        else:
            expected = derive_label(k, ind, stock_code)
        tools = render_tool_results(k, ind, stock_code)
        tags = [task_type.replace("ok_", "")]
        if expected["kdj"]["signal"] == "buy_ready":
            tags.append("买入信号")
        if "暂不设定" in (expected["support"], expected["resistance"]):
            tags.append("价位克制")

    elif task_type == "abort_tool_error":
        k = simulate_kline("neutral", days=90, seed=seed)
        ind = calc_indicators(k)
        which = rng.choice(["history", "price"])
        tools = render_tool_results(k, ind, stock_code, inject_tool_error=which)
        expected = {"status": "ABORT", "reason": f"{'历史K线' if which == 'history' else '行情核验'}工具返回错误，无法完成技术面分析"}
        tags = ["工具错误", which]

    elif task_type == "abort_insufficient":
        k = simulate_kline("neutral", days=90, seed=seed)
        ind = calc_indicators(k)
        tools = render_tool_results(k, ind, stock_code, insufficient=True)
        expected = {"status": "ABORT", "reason": "K线数据不足60根，无法可靠计算KDJ指标"}
        tags = ["数据不足"]
    else:
        raise ValueError(task_type)

    return {"task_type": task_type, "stock_code": stock_code,
            "input": tools, "expected": expected, "tags": tags}
