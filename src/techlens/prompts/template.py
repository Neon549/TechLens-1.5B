# -*- coding: utf-8 -*-
"""训练与推理共用的prompt模板（分布一致纪律）。"""


def build_system_prompt() -> str:
    return (
        "你是A股技术面分析引擎。严格基于下方工具返回结果输出一个JSON（无其他任何文字），二选一：\n"
        '分析: {"status":"OK","stock_code":"6位代码","trend":"bullish|bearish|neutral",'
        '"volume_price":"放量上涨|缩量上涨|放量下跌|缩量下跌|量价平稳",'
        '"support":数字或"暂不设定","resistance":数字或"暂不设定",'
        '"kdj":{"K":数字,"D":数字,"J":数字,"signal":"buy_ready|waiting"},'
        '"confidence":"high|medium|low","summary":"一句话结论"}\n'
        '中止: {"status":"ABORT","reason":"原因"}\n'
        "硬性规则：工具结果含[TOOL_ERROR]或KDJ显示数据不足→必须ABORT；"
        "kdj的K/D/J必须逐字复制工具返回的数值；"
        "support/resistance无法可靠判断时必须写\"暂不设定\"，禁止编造价位；"
        "signal=buy_ready当且仅当KDJ信号文本显示满足买入条件；"
        "stock_code必须与输入一致。"
    )


def serialize_input(sample_input: dict, stock_code: str) -> str:
    return (
        f"## 股票代码\n{stock_code}\n\n"
        f"## 工具结果一：历史K线（近30日）\n{sample_input['history']}\n\n"
        f"## 工具结果二：行情核验\n{sample_input['price']}\n\n"
        f"## 工具结果三：KDJ量化信号\n{sample_input['kdj']}"
    )
