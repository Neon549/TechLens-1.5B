"""准备 llama.cpp 转换暂存目录，不改动原始合并模型。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def prepare(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refuse to overwrite conversion staging directory: {destination}")
    destination.mkdir(parents=True)
    for path in source.iterdir():
        target = destination / path.name
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            os.link(path, target)  # 同一 NTFS 卷内不复制数 GB 权重。
        elif path.name == "tokenizer_config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            removed = config.pop("extra_special_tokens", None)
            target.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (destination / "conversion_patch.json").write_text(
                json.dumps({"removed_extra_special_tokens": removed is not None,
                            "reason": "Transformers compatibility for llama.cpp conversion staging"}, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(path, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/merged/techlens-1.7b")
    parser.add_argument("--out", default="models/gguf/techlens-convert-staging")
    args = parser.parse_args()
    prepare(Path(args.source), Path(args.out))
    print(args.out)
