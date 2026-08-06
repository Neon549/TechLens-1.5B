import json
from collections import Counter
from pathlib import Path

from techlens.datagen.builder import build_targeted_dpo_pairs
from techlens.datagen.engine import generate_sample
from techlens.evaluation.diagnostics import diagnose_records
from techlens.evaluation.runner import run_eval
from techlens.inference.backends import MockBackend


def test_tool_request_samples_validate_and_mock_scores_perfectly(tmp_path):
    rows = []
    for index, task_type in enumerate(("request_history", "request_price", "request_kdj")):
        row = generate_sample(task_type, 100 + index)
        row["id"] = task_type
        rows.append(row)
    path = tmp_path / "tool_request.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    card, _ = run_eval(path, MockBackend("gold"))
    assert card["quality"]["tool_request_accuracy"] == 1.0
    assert card["quality"]["status_accuracy"] == 1.0


def test_targeted_dpo_uses_train_rows_only_and_preserves_chosen():
    pairs = build_targeted_dpo_pairs("data/train/clean.jsonl", {"volume_price": 15, "confidence": 6}, count=21)
    assert len(pairs) == 21
    assert {pair["metadata"]["targeted_field"] for pair in pairs} == {"volume_price", "confidence"}
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)


def test_diagnostics_reports_observed_m2_field_errors():
    profile = diagnose_records("experiments/m2_dpo/records.jsonl")
    assert profile["n_records"] == 30
    assert profile["field_failures"] == {"volume_price": 15, "confidence": 6}


def test_real_frozen_eval_is_balanced_and_diverse():
    rows = [json.loads(line) for line in Path("data/eval/real_test_v2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert Counter(row["task_type"] for row in rows) == {
        "ok_bullish": 24, "ok_bearish": 24, "ok_neutral": 24,
        "ok_no_levels": 24, "ok_edge": 24,
    }
    for task_type in {row["task_type"] for row in rows}:
        assert len({row["source"]["symbol"] for row in rows if row["task_type"] == task_type}) == 10
