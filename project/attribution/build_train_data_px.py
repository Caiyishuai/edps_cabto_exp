#!/usr/bin/env python3
"""把 collect_epm_px.py 产出的 samples.json 转成 mlx-vlm LoRA 训练格式（单图）。

question : 固定 prompt，要求输出物体像素坐标 (u, v)
answer   : 「物体名: (u, v)」逐行（绝对像素）
images   : [overview.png]（单视角）

输出 train.jsonl + val.jsonl，并转成 parquet（train/data.parquet + val/data.parquet）。
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

PROMPT = (
    "这是桌面场景的俯视图。输出桌面上每个物体在图像中的像素坐标 (u, v)，"
    "u 是水平像素坐标（从左到右），v 是垂直像素坐标（从上到下），"
    "格式为「物体名: (u, v)」，每行一个物体，不要输出其他内容。"
)


def answer_text(objects):
    return "\n".join(f"{name}: ({v[0]:.1f}, {v[1]:.1f})" for name, v in objects.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True, help="samples.json 路径")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.samples.read_text())
    samples = data["samples"]
    base = args.samples.parent

    n_val = max(1, int(len(samples) * args.val_ratio))
    rng = random.Random(args.seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    val_idx = set(idx[:n_val])

    train, val = [], []
    for i, s in enumerate(samples):
        images = [str((base / s["image"]).resolve())]
        row = {"images": images, "question": PROMPT, "answer": answer_text(s["objects"])}
        (val if i in val_idx else train).append(row)

    args.output.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "val.jsonl", val)

    # 转 parquet（mlx-vlm 的 load_dataset 用目录加载）
    for split, rows in (("train", train), ("val", val)):
        d = args.output / "parquet" / split
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df["images"] = df["images"].apply(lambda x: x)  # list of str
        df.to_parquet(d / "data.parquet", index=False)

    print(f"train: {len(train)} 条 -> {args.output / 'train.jsonl'}")
    print(f"val:   {len(val)} 条 -> {args.output / 'val.jsonl'}")
    print("\n样例 answer:")
    print(train[0]["answer"])


if __name__ == "__main__":
    main()
