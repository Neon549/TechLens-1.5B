"""冻结工具请求评估集；输出存在时拒绝覆盖。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.datagen.engine import generate_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--out", default="data/eval/tool_request_test.jsonl")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite frozen evaluation set: {output}")
    rows = []
    for offset, task_type in enumerate(("request_history", "request_price", "request_kdj")):
        for index in range(args.per_type):
            row = generate_sample(task_type, seed=90_000 + offset * 10_000 + index)
            row["id"] = f"{task_type}-{index:04d}"
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"frozen {len(rows)} tool-request samples: {output}")
