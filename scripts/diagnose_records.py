"""从一次评估运行生成错误画像，不修改训练数据。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from techlens.evaluation.diagnostics import diagnose_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("records")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    profile = diagnose_records(args.records)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, ensure_ascii=False, indent=2))
