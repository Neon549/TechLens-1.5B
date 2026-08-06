from techlens.prompts.template import serialize_input
from techlens.schemas.output import validate_output


def test_tool_request_contract_is_valid():
    result = {
        "status": "TOOL_REQUEST",
        "tool_name": "get_stock_history",
        "arguments": {"stock_code": "600519"},
        "reason": "历史行情缺失",
    }
    assert validate_output(result) == []


def test_tool_request_rejects_extra_arguments():
    result = {
        "status": "TOOL_REQUEST",
        "tool_name": "get_stock_history",
        "arguments": {"stock_code": "600519", "days": 90},
        "reason": "历史行情缺失",
    }
    assert "tool_request_arguments_must_be_stock_code" in validate_output(result)


def test_missing_tool_results_are_explicit_in_prompt():
    prompt = serialize_input({}, "600519")
    assert "[TOOL_MISSING] tool=get_stock_history" in prompt
    assert "[TOOL_MISSING] tool=get_stock_price" in prompt
    assert "[TOOL_MISSING] tool=get_kdj_signal" in prompt
