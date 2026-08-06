"""构造工具缺失时的请求动作 SFT 数据，不与冻结评估集重叠。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.datagen.builder import to_sft
from techlens.datagen.engine import generate_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=150)
    parser.add_argument("--out-dir", default="data/train")
    args = parser.parse_args()
    rows = []
    for offset, task_type in enumerate(("request_history", "request_price", "request_kdj")):
        for index in range(args.per_type):
            rows.append(generate_sample(task_type, seed=50_000 + offset * 10_000 + index))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tool_action_clean.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (output_dir / "sft_tool_actions.json").write_text(
        json.dumps([to_sft(row) for row in rows], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} train-only tool-action examples")
