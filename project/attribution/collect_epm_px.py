#!/usr/bin/env python3
"""采集 EPM 训练数据 v2：cover 场景 (RGB 观测 → 物体像素坐标) 对。

目标：训练 VLM 从 RGB 俯视图输出物体**像素坐标（2D）**，反投影（像素→3D）交给几何代码
（core.cover_cabto_api.intersect_plane + 已知物体高度）。这是 EPM 论文的原做法，也是
CABTO 现有 cover_visual_localizer 的做法——VLM 只做视觉定位（像素），不做相机几何。

数据对：
    输入  = overview 俯视图（单视角）
    输出  = 物体清单（每个物体：name + 像素坐标 (u, v)，绝对像素，图 720x720）

难例：随机 jitter 物体位置，让物体在桌面随机分布。

用法：
    python3 collect_epm_px.py --num 400 --jitter 0.04 --seed 0
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm  # noqa: E402
from core.cover_cabto_api import project  # noqa: E402  3D -> 像素投影

CAM = "overview"


def main():
    ap = argparse.ArgumentParser(description="采集 EPM 像素坐标训练数据")
    ap.add_argument("--num", type=int, default=400, help="样本数")
    ap.add_argument("--jitter", type=float, default=0.04, help="物体 xy 扰动幅度(米)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=PACKAGE / "outputs/epm_px_data")
    args = ap.parse_args()

    config = json.loads((ROOT / "scenes/cover_kitchen_sort_v2.json").read_text())
    items = list(config["items"].keys())             # shrimp / apple / pepper
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

            # 渲染 overview 俯视图
            img = env.render_cam(CAM)
            size = img.shape[0]
            p = img_dir / f"{i:04d}_{CAM}.png"
            Image.fromarray(img).save(p)

            # 真值：每个物体的像素坐标（物体中心 3D -> 投影到像素）+ 3D 坐标（用于反投影验证）
            objects = {}
            objects_3d = {}
            for name in items:
                pos3d = env.get_object_pose(name)[0]
                uv = project(env, pos3d, CAM, size)
                objects[name] = [round(float(uv[0]), 1), round(float(uv[1]), 1)]
                objects_3d[name] = [round(float(v), 4) for v in pos3d]

            samples.append({
                "id": i,
                "image": str(p.relative_to(out)),
                "objects": objects,
                "objects_3d": objects_3d,
            })

    finally:
        runtime.close()

    meta = {
        "task": "cover", "items": items, "num": len(samples),
        "jitter": args.jitter, "seed": args.seed,
        "camera": CAM, "image_size": 720,
        "label": "物体中心的像素坐标 (u, v)，绝对像素，原点左上角",
    }
    (out / "samples.json").write_text(
        json.dumps({"meta": meta, "samples": samples}, indent=2, ensure_ascii=False))

    print(f"采集 {len(samples)} 个样本 -> {out / 'samples.json'}")
    print(f"物体: {items}")


if __name__ == "__main__":
    main()
