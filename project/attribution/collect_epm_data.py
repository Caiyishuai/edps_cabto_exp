#!/usr/bin/env python3
"""采集最小 EPM 切片训练数据：cover 场景 (RGB 观测 → 物体 3D 坐标) 对。

目标：训练 VLM 从 RGB 图直接输出物体 3D 坐标，替代「Qwen 打点 + 已知高度反投影」。

数据对：
    输入  = 图片（overview + front 两视角）
    输出  = 物体清单（每个物体：name + 3D 世界坐标）

难例：随机 jitter 物体位置（比 config 的 2mm jitter 大得多），让物体在桌面随机
分布，自然产生「物体靠得很近」等较难定位的样本。遮挡 / 空中 / 动态等难例后续加。

用法：
    python3 collect_epm_data.py --num 200 --jitter 0.03 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="采集 EPM 训练数据")
    ap.add_argument("--num", type=int, default=200, help="样本数")
    ap.add_argument("--jitter", type=float, default=0.03, help="物体 xy 扰动幅度(米)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=PACKAGE / "outputs/epm_data")
    args = ap.parse_args()

    config = json.loads((ROOT / "scenes/cover_kitchen_sort_v2.json").read_text())
    items = list(config["items"].keys())             # shrimp / apple / pepper（free 物体）
    rng = np.random.default_rng(args.seed)

    runtime = RebuiltSingleArm(config, seed=0, render=True)
    env = runtime.env
    initial = {name: env.get_object_pose(name)[0].copy() for name in items}

    out = args.output
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    try:
        for i in range(args.num):
            # 随机 jitter 物体 xy（z 保持桌面高度）
            for name in items:
                b = env.obj_qadr[name]
                x = initial[name][0] + rng.uniform(-args.jitter, args.jitter)
                y = initial[name][1] + rng.uniform(-args.jitter, args.jitter)
                env.d.qpos[b:b + 3] = [x, y, initial[name][2]]
            mujoco.mj_forward(env.m, env.d)

            # 渲染两视角
            views = {}
            for cam in ("overview", "front"):
                img = env.render_cam(cam)
                p = img_dir / f"{i:04d}_{cam}.png"
                Image.fromarray(img).save(p)
                views[cam] = str(p.relative_to(out))

            # 真值 3D 坐标
            objects = {
                name: [round(float(v), 4) for v in env.get_object_pose(name)[0]]
                for name in items
            }
            samples.append({"id": i, "views": views, "objects": objects})

    finally:
        runtime.close()

    meta = {
        "task": "cover", "items": items, "num": len(samples),
        "jitter": args.jitter, "seed": args.seed,
        "views": ["overview", "front"],
        "label": "物体 3D 世界坐标（free joint 中心，真值）",
        "config_sha256": json.loads((ROOT / "scenes/cover_kitchen_sort_v2.json").read_text()) and
            __import__("hashlib").sha256((ROOT / "scenes/cover_kitchen_sort_v2.json").read_bytes()).hexdigest(),
    }
    (out / "samples.json").write_text(json.dumps({"meta": meta, "samples": samples}, indent=2, ensure_ascii=False))

    print(f"采集 {len(samples)} 个样本 -> {out / 'samples.json'}")
    print(f"物体: {items}")
    print(f"每个样本两视角图片: {img_dir}")


if __name__ == "__main__":
    main()
