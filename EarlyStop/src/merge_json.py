import os, json, argparse
from pathlib import Path
import re

def extract_id(fp: Path) -> int:
    """从文件名里提取数字 ID，比如 item_00005.json -> 5"""
    match = re.search(r"(\d+)", fp.stem)
    if match:
        return int(match.group(1))
    return -1  # 没有数字时放到最前

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="tmp_output_json 目录")
    ap.add_argument("--output_json", required=True, help="合并后的 JSON 路径")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.json"), key=extract_id)

    out = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            out.extend(data)
        else:
            raise ValueError(f"{fp} 顶层不是 list")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] merged {len(files)} files -> {args.output_json} (items={len(out)})")

if __name__ == "__main__":
    main()



