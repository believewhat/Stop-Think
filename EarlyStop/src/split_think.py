# split_json.py
import os, json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default="/home/azureuser/cloudfiles/code/Users/junda.wang/project/AdaptThink/data/train/medicalqa/medicalqa_with_correct_think.json")
    ap.add_argument("--out_dir", default="tmp_input_json")
    ap.add_argument("--prefix", default="item", help="文件名前缀")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(in_path, "r", encoding="utf-8"))
    assert isinstance(data, list), "输入顶层必须是 list"

    digits = max(4, len(str(len(data))))
    for i, item in enumerate(data):
        # 每个文件存一个元素的 list，保持你现有 condense 脚本兼容
        fn = f"{args.prefix}_{i:0{digits}d}.json"
        with open(out_dir / fn, "w", encoding="utf-8") as f:
            json.dump([item], f, ensure_ascii=False)

    print(f"[done] wrote {len(data)} files to {out_dir}")

if __name__ == "__main__":
    main()
