#!/usr/bin/env python3
"""把 collect_epm_data.py 产出的 samples.json 转成 mlx-vlm LoRA 训练格式。

mlx-vlm 的 VisionDataset 接受 question / answer / images 列：
    question : 固定 prompt，要求输出物体 3D 坐标
    answer   : 结构化文本「物体名: (x, y, z)」逐行
    images   : [overview.png, front.png]（两视角）

输出 train.jsonl + val.jsonl（按比例划分）。
"""
import argparse
import json
from pathlib import Path

PROMPT = (
    "这是桌面场景的两张图片（第一张 overview 俯视图，第二张 front 正视图）。"
    "输出桌面上每个物体的 3D 世界坐标 (x, y, z)，单位米，格式为「物体名: (x, y, z)」，"
    "每行一个物体，不要输出其他内容。"
)


def answer_text(objects):
    return "\n".join(f"{name}: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})" for name, v in objects.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True, help="samples.json 路径")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.samples.read_text())
    samples = data["samples"]
    base = args.samples.parent  # 图片的相对路径基准

    n_val = max(1, int(len(samples) * args.val_ratio))
    import random
    rng = random.Random(args.seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    val_idx = set(idx[:n_val])

    train, val = [], []
    for i, s in enumerate(samples):
        # 图片绝对路径
        images = [str((base / s["views"][cam]).resolve()) for cam in ("overview", "front")]
        row = {"images": images, "question": PROMPT, "answer": answer_text(s["objects"])}
        (val if i in val_idx else train).append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write(args.output / "train.jsonl", train)
    write(args.output / "val.jsonl", val)

    print(f"train: {len(train)} 条 -> {args.output / 'train.jsonl'}")
    print(f"val:   {len(val)} 条 -> {args.output / 'val.jsonl'}")
    print("\n样例 answer:")
    print(train[0]["answer"])


if __name__ == "__main__":
    main()
