"""从评估 records 生成可审计的聚合错误画像。"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def diagnose_records(records_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(records_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    errors = Counter()
    task_types = Counter()
    for row in rows:
        task_types[row.get("task_type", "unknown")] += 1
        reason = row.get("scores", {}).get("fields", {}).get("reason") or ""
        if reason.startswith("wrong:"):
            for field in reason.removeprefix("wrong:").strip("[]").replace("'", "").split(","):
                if field.strip():
                    errors[field.strip()] += 1
    return {
        "records_path": str(records_path),
        "n_records": len(rows),
        "task_type_counts": dict(sorted(task_types.items())),
        "field_failures": dict(errors.most_common()),
        "limitations": [
            "该画像只表示被评估模型在该 records 文件中的失败模式。",
            "若 n_records 或任务类型覆盖不足，必须先运行完整冻结评估再作为训练依据。",
            "画像只提供字段级聚合统计；定向 DPO 只能使用训练集样本。",
        ],
    }
